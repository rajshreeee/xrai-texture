import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification
from torchvision import models
from collections import Counter
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms



train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32
num_epochs = 30
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # pass class weights if needed

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
        dice_loss = 1 - dice

        return bce_loss + dice_loss


class CBISDDSMDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")

        # Treat 'benign_without_callback' as 'benign'
        self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')

        # Filter valid entries
        valid_pathologies = {"malignant", "benign"}
        valid_shapes = {"round", "oval", "lobulated", "irregular"}
        valid_birads = {2, 3, 4, 5}

        self.data = self.data[
            (self.data['pathology'].isin(valid_pathologies)) &
            (self.data['mass shape'].str.lower().isin(valid_shapes)) &
            (self.data['assessment'].isin(valid_birads))
        ].reset_index(drop=True)

        print(f"[INFO] Filtered rows: {len(self.data)}")

        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[INFO] Found {len(self.image_files)} images.")

        # Define mappings
        self.pathology_classes = {"benign": 0, "malignant": 1}
        self.shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_mapping = {2: 0, 3: 1, 4: 2, 5: 3}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        base_name = os.path.splitext(image_name)[0].split('#')[0]

        image = Image.open(image_path).convert("RGB")
        matching_rows = self.data[self.data['image_name'] == base_name]

        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]

        pathology = self.pathology_classes.get(row['pathology'], 1)
        shape = self.shape_classes.get(str(row['mass shape']).lower(), 3)

        birads_raw = int(row['assessment'])
        birads = self.birads_mapping.get(birads_raw, None)
        if birads is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([pathology, shape, birads], dtype=torch.long)

        return image, labels

    def get_all_labels(self):
        labels = []
        for _, row in self.data.iterrows():
            pathology = self.pathology_classes.get(row["pathology"], 1)
            mass_shape = self.shape_classes.get(str(row["mass shape"]).lower(), 3)
            birads_raw = int(row["assessment"])
            birads = self.birads_mapping.get(birads_raw, 4)  # Map BIRADS correctly
            labels.append(torch.tensor([pathology, mass_shape, birads]))
        return labels

def load_swin_model(model_name="swin_tiny_patch4_window7_224"):
    model = timm.create_model(model_name, pretrained=True, features_only=False, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layers.2" in name or "layers.3" in name or "norm" in name or "head" in name:
            param.requires_grad = True
    return model


class StackedEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = load_swin_model()  # 768-d output from timm
        self.shared_dim = 768

        self.fc_pathology = nn.Sequential(
            nn.Linear(self.shared_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 1)
        )

        self.fc_birads = nn.Sequential(
            nn.Linear(self.shared_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 4)
        )

        self.fc_shape = nn.Sequential(
            nn.Linear(self.shared_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        f = self.swin(x)  # [B, 768]

        out_pathology = self.fc_pathology(f)
        out_birads    = self.fc_birads(f)
        out_shape     = self.fc_shape(f)
        return out_pathology, out_birads, out_shape


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ========= Prepare Data =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CBISDDSMDataset(train_path, csv_file, transform=train_transform)
test_dataset = CBISDDSMDataset(test_path, csv_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

print(f"[INFO] Training samples: {len(train_dataset)}")
print(f"[INFO] Testing samples:  {len(test_dataset)}")

train_labels = train_dataset.get_all_labels()  # list of tensors
train_labels_np = torch.stack(train_labels).numpy()  # shape: (N, 3)


# === Shape Class Weights ===
shape_labels = train_labels_np[:, 1]
shape_classes = np.unique(shape_labels)
shape_weights = compute_class_weight(class_weight='balanced', classes=shape_classes, y=shape_labels)
shape_weights = torch.tensor(shape_weights, dtype=torch.float).to(device)

# === BIRADS Class Weights ===
birads_labels = train_labels_np[:, 2]
birads_classes = np.unique(birads_labels)
birads_weights = compute_class_weight(class_weight='balanced', classes=birads_classes, y=birads_labels)
birads_weights = torch.tensor(birads_weights, dtype=torch.float).to(device)


# ==== Model, Losses, Optimizer ====
model = StackedEnsemble().to(device)

if torch.cuda.device_count() > 1:
    print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

loss_pathology = BCEDiceLoss()
loss_shape = FocalLoss(gamma=2.0, weight=shape_weights)
loss_birads = FocalLoss(gamma=2.0, weight=birads_weights)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

train_losses = []
train_accuracies_pathology, train_accuracies_shape, train_accuracies_birads = [], [], []

# Early stopping settings
early_stopping_patience = 5
best_train_loss = float('inf')
epochs_no_improve = 0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    preds_path, targets_path = [], []
    preds_shape, targets_shape = [], []
    preds_birads, targets_birads = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        pathology_targets = labels[:, 0].float().unsqueeze(1)
        shape_targets = labels[:, 1].long()
        birads_targets = labels[:, 2].long()

        optimizer.zero_grad()
        out_pathology, out_birads, out_shape = model(images)

        loss1 = loss_pathology(out_pathology, pathology_targets)
        loss2 = loss_birads(out_birads, birads_targets)
        loss3 = loss_shape(out_shape, shape_targets)
        total_loss = loss1 + loss2 + loss3

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

        # Track predictions
        probs_pathology = torch.sigmoid(out_pathology)
        pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
        preds_path.extend(pred_labels_path.flatten())
        targets_path.extend(pathology_targets.cpu().numpy().flatten())

        pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
        preds_shape.extend(pred_labels_shape.flatten())
        targets_shape.extend(shape_targets.cpu().numpy().flatten())

        pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
        preds_birads.extend(pred_labels_birads.flatten())
        targets_birads.extend(birads_targets.cpu().numpy().flatten())

    avg_train_loss = running_loss / len(train_loader)
    acc_path = accuracy_score(targets_path, preds_path)
    acc_shape = accuracy_score(targets_shape, preds_shape)
    acc_birads = accuracy_score(targets_birads, preds_birads)

    train_losses.append(avg_train_loss)
    train_accuracies_pathology.append(acc_path)
    train_accuracies_shape.append(acc_shape)
    train_accuracies_birads.append(acc_birads)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss={avg_train_loss:.4f} | "
          f"Pathology Acc={acc_path:.4f} | Shape Acc={acc_shape:.4f} | BIRADS Acc={acc_birads:.4f}")

    # === Early Stopping Logic ===
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            break

end_time = time.time()
print(f"[INFO] Training completed in {(end_time - start_time)/60:.2f} minutes.")


model.eval()

# Reset prediction lists
preds_path, targets_path = [], []
preds_shape, targets_shape = [], []
preds_birads, targets_birads = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out_pathology, out_birads, out_shape = model(images)

        # Pathology
        probs_pathology = torch.sigmoid(out_pathology)
        pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
        preds_path.extend(pred_labels_path.flatten())
        targets_path.extend(labels[:, 0].cpu().numpy().flatten())

        # Shape
        pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
        preds_shape.extend(pred_labels_shape.flatten())
        targets_shape.extend(labels[:, 1].cpu().numpy().flatten())

        # BIRADS
        pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
        preds_birads.extend(pred_labels_birads.flatten())
        targets_birads.extend(labels[:, 2].cpu().numpy().flatten())

from sklearn.metrics import ConfusionMatrixDisplay

print("\n========= [INFO] Final Test Results =========")

# --- Pathology ---
print("\nPathology Classification")
print(f"Accuracy: {accuracy_score(targets_path, preds_path):.4f}")
print("Classification Report:")
print(classification_report(targets_path, preds_path, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:")
print(confusion_matrix(targets_path, preds_path))

# --- Shape ---
print("\nShape Classification")
print(f"Accuracy: {accuracy_score(targets_shape, preds_shape):.4f}")
print("Classification Report:")
print(classification_report(
    targets_shape, preds_shape,
    target_names=["Round", "Oval", "Lobulated", "Irregular"]
))
print("Confusion Matrix:")
print(confusion_matrix(targets_shape, preds_shape))

# --- BIRADS ---
print("\nBIRADS Classification")
print(f"Accuracy: {accuracy_score(targets_birads, preds_birads):.4f}")
print("Classification Report:")
print(classification_report(
    targets_birads, preds_birads,
    target_names=["BIRADS 2", "BIRADS 3", "BIRADS 4", "BIRADS 5"]
))
print("Confusion Matrix:")
print(confusion_matrix(targets_birads, preds_birads))

# ==== Save Training Curves ====
os.makedirs("tuned-stacked-ensemble-plots-all-one-swin-no-shuffle", exist_ok=True)
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(18, 8))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies_pathology, label='Pathology Acc')
plt.plot(epochs_range, train_accuracies_shape, label='Shape Acc')
plt.plot(epochs_range, train_accuracies_birads, label='BIRADS Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracies Curve')
plt.legend()

plt.tight_layout()
plt.savefig("tuned-stacked-ensemble-plots-all-one-swin-no-shuffle/training_curves.png")
plt.close()

print("[INFO] Training plots saved to tuned-stacked-ensemble-plots-all-one-swin-no-shuffle/training_curves.png")

torch.save(model.state_dict(), "tuned-stacked-ensemble-plots-all-one-swin_stacked_ensemble_model-no-shuffle.pth")
print("[INFO] Model saved as tuned-stacked-ensemble-plots-all-one-swin_stacked_ensemble_model-no-shuffle.pth")
