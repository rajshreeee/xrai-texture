import argparse
import os
import time
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# We use standard sklearn split as fallback if skmultilearn is missing
try:
    from skmultilearn.model_selection import IterativeStratification
    HAS_ITERATIVE = True
except ImportError:
    from sklearn.model_selection import train_test_split
    HAS_ITERATIVE = False
    
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# ================= ARGUMENTS =================
parser = argparse.ArgumentParser(description='Train Multi-Task Swin (Fixed)')
parser.add_argument('--train_dir', type=str, required=True, help='Path to augmented masked TRAINING images')
parser.add_argument('--test_dir', type=str, required=True, help='Path to augmented masked TEST images')
parser.add_argument('--csv_file', type=str, required=True, help='Path to annotations CSV')
parser.add_argument('--output_name', type=str, default='swin_fine_tuned', help='Name for saved model')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOSS FUNCTIONS =================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return bce_loss + (1 - dice)

# ================= DATASET CLASS =================
class CBISDDSMDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        
        # Cleanup and Filtering
        if 'pathology' in self.data.columns:
            self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')
            self.data = self.data[
                (self.data['pathology'].isin({"malignant", "benign"})) &
                (self.data['mass shape'].str.lower().isin({"round", "oval", "lobulated", "irregular"})) &
                (self.data['assessment'].isin({2, 3, 4, 5}))
            ].reset_index(drop=True)
        
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.pathology_map = {"benign": 0, "malignant": 1}
        self.shape_map = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_map = {2: 0, 3: 1, 4: 2, 5: 3}

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0].split('#')[0]
        
        row = self.data[self.data['image_name'] == base_name]
        if row.empty: return self.__getitem__((idx + 1) % len(self))
        row = row.iloc[0]

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform: img = self.transform(img)

        # FIX 1: Change default from 4 to 2 (valid index)
        labels = torch.tensor([
            self.pathology_map.get(row['pathology'], 1),
            self.shape_map.get(str(row['mass shape']).lower(), 3),
            self.birads_map.get(int(row['assessment']), 2) 
        ], dtype=torch.long)
        return img, labels

# ================= MODEL SETUP =================
def load_swin_model(model_name="swin_tiny_patch4_window7_224"):
    model = timm.create_model(model_name, pretrained=True, features_only=False, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "norm" in name or "head" in name or "layers.3" in name:
            param.requires_grad = True
    return model

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = load_swin_model()
        self.shared_dim = 768
        self.fc_pathology = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
        # This outputs 4 classes (indices 0, 1, 2, 3)
        self.fc_birads = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 4))
        self.fc_shape = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 4))

    def forward(self, x):
        f = self.swin(x)
        return self.fc_pathology(f), self.fc_birads(f), self.fc_shape(f)

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print(f"[INFO] Training with dynamic paths on {args.train_dir}")
    start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 1. Load Datasets
    print("[INFO] Loading Datasets...")
    train_ds_initial = CBISDDSMDataset(args.train_dir, args.csv_file, transform)
    test_ds_initial = CBISDDSMDataset(args.test_dir, args.csv_file, transform)

    # 2. Extract Labels (OPTIMIZED + FIXED)
    print("[INFO] Extracting labels for stratification (Fast Mode)...")
    
    def get_labels_fast(dataset):
        labels_list = []
        unique_data = dataset.data.drop_duplicates(subset=['image_name'])
        data_lookup = unique_data.set_index('image_name').to_dict('index')
        
        for img_name in dataset.image_files:
            base_name = os.path.splitext(img_name)[0].split('#')[0]
            
            if base_name in data_lookup:
                row = data_lookup[base_name]
                # FIX 2: Change default from 4 to 2
                label = [
                    dataset.pathology_map.get(row['pathology'], 1),
                    dataset.shape_map.get(str(row['mass shape']).lower(), 3),
                    dataset.birads_map.get(int(row['assessment']), 2)
                ]
                labels_list.append(label)
            else:
                # FIX 3: Change fallback from [1, 3, 4] to [1, 3, 2]
                labels_list.append([1, 3, 2]) 
        return labels_list

    all_labels_list = []
    all_labels_list.extend(get_labels_fast(train_ds_initial))
    all_labels_list.extend(get_labels_fast(test_ds_initial))
    all_labels_np = np.array(all_labels_list)

    # 3. Calculate Weights & Split
    combined_dataset = ConcatDataset([train_ds_initial, test_ds_initial])
    
    print("[INFO] Calculating Class Weights...")
    shape_weights = compute_class_weight('balanced', classes=np.unique(all_labels_np[:, 1]), y=all_labels_np[:, 1])
    shape_weights = torch.tensor(shape_weights, dtype=torch.float).to(device)
    
    birads_weights = compute_class_weight('balanced', classes=np.unique(all_labels_np[:, 2]), y=all_labels_np[:, 2])
    birads_weights = torch.tensor(birads_weights, dtype=torch.float).to(device)
    
    # Confirm shape is correct (should be 4)
    if len(birads_weights) != 4:
        print(f"[WARN] Expected 4 BI-RADS weights, got {len(birads_weights)}. Padding or truncating to avoid crash.")
        # Fallback to avoid crash if data is missing classes
        # This manually sets weights to 1.0 if calculation is weird, to let training proceed
        if len(birads_weights) < 4:
             birads_weights = torch.ones(4).to(device)
        elif len(birads_weights) > 4:
             birads_weights = birads_weights[:4]

    print("[INFO] Splitting Data...")
    
    if HAS_ITERATIVE:
        stratifier = IterativeStratification(n_splits=5, order=1)
        train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))
        
        temp_labels = all_labels_np[temp_idx]
        stratifier_val_test = IterativeStratification(n_splits=2, order=1)
        val_sub_idx, test_sub_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels)), temp_labels))
        
        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]
    else:
        print("[WARN] skmultilearn not found. Using standard sklearn split.")
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(all_labels_np))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=all_labels_np[:,0])
        
        temp_labels = all_labels_np[temp_idx]
        val_sub_idx, test_sub_idx = train_test_split(np.arange(len(temp_labels)), test_size=0.5, random_state=42, stratify=temp_labels[:,0])
        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

    print(f"[INFO] Split Sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Setup
    model = SwinClassifier().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    criterion_path = BCEDiceLoss()
    criterion_shape = FocalLoss(gamma=2.0, weight=shape_weights)
    criterion_birads = FocalLoss(gamma=2.0, weight=birads_weights)

    # 5. Training Loop
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("[INFO] Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            p_target = labels[:, 0].float().unsqueeze(1)
            s_target = labels[:, 1]
            b_target = labels[:, 2]

            optimizer.zero_grad()
            p_out, b_out, s_out = model(images)
            
            loss = criterion_path(p_out, p_target) + \
                   criterion_birads(b_out, b_target) + \
                   criterion_shape(s_out, s_target)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                p_target = labels[:, 0].float().unsqueeze(1)
                s_target = labels[:, 1]
                b_target = labels[:, 2]

                p_out, b_out, s_out = model(images)
                loss = criterion_path(p_out, p_target) + \
                       criterion_birads(b_out, b_target) + \
                       criterion_shape(s_out, s_target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{args.output_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break

    end_time = time.time()
    print(f"\n[INFO] Training Time: {(end_time - start_time)/60:.2f} minutes")
    print(f"[INFO] Model saved as {args.output_name}.pth")

    # 6. Evaluation
    print("\n[INFO] Starting Final Evaluation...")
    try:
        model.load_state_dict(torch.load(f"{args.output_name}.pth"))
    except RuntimeError:
        print("[INFO] Loading with strict=False to handle buffer mismatch...")
        model.load_state_dict(torch.load(f"{args.output_name}.pth"), strict=False)
        
    model.eval()

    preds_p, targs_p = [], []
    preds_s, targs_s = [], []
    preds_b, targs_b = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            p_out, b_out, s_out = model(images)

            preds_p.extend((torch.sigmoid(p_out) > 0.5).long().cpu().numpy().flatten())
            targs_p.extend(labels[:, 0].cpu().numpy())

            preds_s.extend(s_out.argmax(dim=1).cpu().numpy())
            targs_s.extend(labels[:, 1].cpu().numpy())

            preds_b.extend(b_out.argmax(dim=1).cpu().numpy())
            targs_b.extend(labels[:, 2].cpu().numpy())

    print("\n--- Pathology ---")
    print(classification_report(targs_p, preds_p))
    print("Confusion Matrix:\n", confusion_matrix(targs_p, preds_p))

    print("\n--- Shape ---")
    print(classification_report(targs_s, preds_s))
    print("Confusion Matrix:\n", confusion_matrix(targs_s, preds_s))

    print("\n--- BI-RADS ---")
    print(classification_report(targs_b, preds_b))
    print("Confusion Matrix:\n", confusion_matrix(targs_b, preds_b))