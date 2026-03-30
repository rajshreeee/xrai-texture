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
from torchvision.transforms import Resize, ToTensor

train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_aug_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_aug_masked_images"
masked_train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masks"
masked_test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masks"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32 # 16 maybe?
num_epochs = 100 # around 50. 
learning_rate = 1e-2 #make low.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss for multi-class classification.

        Args:
            alpha: weight for false positives
            beta: weight for false negatives
            smooth: small constant to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (integer class labels)
        """
        num_classes = logits.size(1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        probs = F.softmax(logits, dim=1)

        # Compute Tversky index
        TP = (probs * targets_one_hot).sum(dim=0)
        FP = ((1 - targets_one_hot) * probs).sum(dim=0)
        FN = (targets_one_hot * (1 - probs)).sum(dim=0)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1.0 - tversky.mean()

        return loss

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

# class BCEDiceLoss(torch.nn.Module):
#     def __init__(self, smooth=1.0):
#         super().__init__()
#         self.smooth = smooth
#         self.bce = torch.nn.BCEWithLogitsLoss()

#     def forward(self, logits, targets):
#         bce_loss = self.bce(logits, targets)

#         probs = torch.sigmoid(logits)
#         targets = targets.float()

#         intersection = (probs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
#         dice_loss = 1 - dice

#         return bce_loss + dice_loss
class MaskDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits and targets: [B, H, W]
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2))  # [B]
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + self.smooth
        )
        dice_loss = 1 - dice.mean()
        return bce_loss + dice_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1,2,3))
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth
        )
        return bce_loss + (1 - dice).mean()

# === Updated Dataset Loader with Auxiliary Mask Support ===
class CBISDDSMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, csv_file, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.data = pd.read_csv(csv_file)
        self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')

        valid_pathologies = {"malignant", "benign"}
        valid_shapes = {"round", "oval", "lobulated", "irregular"}
        valid_birads = {2, 3, 4, 5}

        self.data = self.data[
            (self.data['pathology'].isin(valid_pathologies)) &
            (self.data['mass shape'].str.lower().isin(valid_shapes)) &
            (self.data['assessment'].isin(valid_birads))
        ].reset_index(drop=True)

        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.pathology_classes = {"benign": 0, "malignant": 1}
        self.shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_mapping = {2: 0, 3: 1, 4: 2, 5: 3}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_name = os.path.splitext(image_name)[0].split('#')[0]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)  # same name assumed

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # binary mask
        mask = Resize((224, 224))(mask)
        mask = transforms.ToTensor()(mask)

        matching_rows = self.data[self.data['image_name'] == base_name]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(row['pathology'], 1)
        shape = self.shape_classes.get(str(row['mass shape']).lower(), 3)
        birads = self.birads_mapping.get(int(row['assessment']), 4)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        labels = torch.tensor([pathology, shape, birads], dtype=torch.long)
        return image, labels, mask

# Helper function to load backbone without final FC
def load_resnet_model(model_name):
    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    modules = list(model.children())[:-2]  # Keep up to the last conv layer
    model = nn.Sequential(*modules)

    # Unfreeze last few layers
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

#=== Stacked Ensemble Model ===
# === Updated Model with Auxiliary Decoder ===
class StackedEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = load_resnet_model("resnet50")
        self.resnet101 = load_resnet_model("resnet101")
        self.resnet152 = load_resnet_model("resnet152")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_dim = 3 * 2048

        self.fc = nn.Linear(self.shared_dim, 1 + 4 + 4)

        # Auxiliary decoder for shape mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3 * 2048, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=1)  # Output single-channel binary mask
        )

    def forward(self, x):
        f1 = self.resnet50(x)  # [B, 2048, 7, 7]
        f2 = self.resnet101(x)
        f3 = self.resnet152(x)

        # Concat for decoder
        features_3d = torch.cat([f1, f2, f3], dim=1)  # [B, 6144, 7, 7]
        aux_mask = self.decoder(features_3d)  # [B, 1, H, W]

        # Pool and classify
        x1 = self.pool(f1).flatten(start_dim=1)
        x2 = self.pool(f2).flatten(start_dim=1)
        x3 = self.pool(f3).flatten(start_dim=1)
        combined = torch.cat([x1, x2, x3], dim=1)

        output = self.fc(combined)
        out_path = output[:, :1]
        out_birads = output[:, 1:5]
        out_shape = output[:, 5:]

        return out_path, out_birads, out_shape, aux_mask.squeeze(1)  # [B, H, W]



# class StackedEnsemble(nn.Module):
#     def __init__(self):
#         super(StackedEnsemble, self).__init__()
#         self.resnet50 = load_resnet_model("resnet50")
#         self.resnet101 = load_resnet_model("resnet101")
#         self.resnet152 = load_resnet_model("resnet152")

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 2048, 1, 1]
#         self.feature_dim = 2048  # Output dim from each ResNet

#         # Attention Layer over 3 backbones
#         self.attn_layer = nn.Sequential(
#             nn.Linear(self.feature_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)  # One score per backbone
#         )

#         # Final task heads
#         self.fc_pathology = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1)
#         )

#         self.fc_birads = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 4)
#         )

#         self.fc_shape = nn.Sequential(
#             nn.Linear(self.feature_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 4)
#         )

#     def forward(self, x):
#         # Backbone features: [B, 2048]
#         x1 = self.pool(self.resnet50(x)).flatten(1)
#         x2 = self.pool(self.resnet101(x)).flatten(1)
#         x3 = self.pool(self.resnet152(x)).flatten(1)

#         # Stack features: [B, 3, 2048]
#         features = torch.stack([x1, x2, x3], dim=1)

#         # Attention scores: softmax over 3 backbones → [B, 3]
#         #attn_scores = self.attn_layer(features.mean(dim=2))  # [B, 3]
#         attn_scores = self.attn_layer(features)              # [B, 3, 2048] → [B, 3, 1]
#         attn_scores = attn_scores.squeeze(-1)                # [B, 3]
#         attn_weights = torch.softmax(attn_scores, dim=1)     # [B, 3]

#         # Weighted sum → [B, 2048]
#         attn_weights = attn_weights.unsqueeze(2)             # [B, 3, 1]
#         fused_features = (features * attn_weights).sum(dim=1)

#         # Task heads
#         out_pathology = self.fc_pathology(fused_features)
#         out_birads    = self.fc_birads(fused_features)
#         out_shape     = self.fc_shape(fused_features)

#         return out_pathology, out_birads, out_shape
    
# ========= Prepare Data =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CBISDDSMDataset(train_path, masked_train_path, csv_file, transform=transform)
test_dataset_extra = CBISDDSMDataset(test_path, masked_test_path, csv_file, transform=transform)
combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])


# 1. Prepare multilabel matrix
all_labels = []
for i in range(len(combined_dataset)):
    _, labels, _ = combined_dataset[i]
    all_labels.append(labels)

# Stack into (n_samples, 3) [pathology, shape, birads]
all_labels_np = torch.stack(all_labels).numpy()

# ========== Print Overall Distribution Before Splitting ==========
print("[INFO] Overall dataset size:", len(all_labels_np))

pathology_counts = Counter(all_labels_np[:, 0])
shape_counts = Counter(all_labels_np[:, 1])
birads_counts = Counter(all_labels_np[:, 2])

print("Pathology distribution (0=benign,1=malignant):", dict(pathology_counts))
print("Shape distribution (0=round,1=oval,2=lobulated,3=irregular):", dict(shape_counts))
print("BIRADS distribution (0->B2,1->B3,2->B4,3->B5):", dict(birads_counts))

# 2. Setup stratifier
stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# 3. Get train/test split
train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# 4. Further split temp into val and test (50-50)
temp_labels_np = all_labels_np[temp_idx]
stratifier_val_test = IterativeStratification(n_splits=2, order=1)
val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# Need to adjust temp_idx mapping
val_idx = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

# 5. Print distributions after split
def print_split_distribution(indices, name):
    labels = all_labels_np[indices]
    pathology_counts = Counter(labels[:, 0])
    shape_counts = Counter(labels[:, 1])
    birads_counts = Counter(labels[:, 2])
    print(f"\n[INFO] {name} size: {len(indices)}")
    print("Pathology distribution:", dict(pathology_counts))
    print("Shape distribution:", dict(shape_counts))
    print("BIRADS distribution:", dict(birads_counts))

print_split_distribution(train_idx, "TRAIN")
print_split_distribution(val_idx,   "VALID")
print_split_distribution(test_idx,  "TEST")


# 6. Final DataLoaders
train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

print(f"\nFinal Split Sizes --> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


def get_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.sum()

shape_weights = get_class_weights(all_labels_np[:, 1], 4).to(device)
#birads_weights = get_class_weights(all_labels_np[:, 2], 4).to(device)

from sklearn.utils.class_weight import compute_class_weight

birads_labels = [lbl[2] for lbl in all_labels_np]
class_weights_birads = compute_class_weight('balanced', classes=np.unique(birads_labels), y=birads_labels)
class_weights_birads = torch.tensor(class_weights_birads, dtype=torch.float).to(device)

# ==== Model, Losses, Optimizer ====
model = StackedEnsemble().to(device)

if torch.cuda.device_count() > 1:
    print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

loss_pathology = BCEDiceLoss()
loss_birads = FocalLoss(gamma=2.0)
loss_shape = FocalLoss(gamma=2.0)
loss_aux = MaskDiceLoss()
#loss_shape = TverskyLoss(alpha=0.3, beta=0.7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
val_accuracies_pathology, val_accuracies_shape, val_accuracies_birads = [], [], []

start_time = time.time()
# ==== Training Loop ====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels, masks in train_loader:
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)

        pathology_labels = labels[:, 0]
        shape_labels = labels[:, 1]
        birads_labels = labels[:, 2:].long()


        pathology_targets = labels[:, 0].float().unsqueeze(1)  # float + unsqueeze for BCE
        shape_targets = labels[:, 1].long()
        birads_targets = labels[:, 2].long()

        optimizer.zero_grad()
        out_pathology, out_birads, out_shape, out_mask = model(images)

        # loss1 = loss_pathology(out_pathology, pathology_targets)
        # loss2 = loss_birads(out_birads, birads_targets)
        # loss3 = loss_shape(out_shape, shape_targets)

        # w1 = 1.0  # pathology
        # w2 = 1.0  # birads
        # w3 = 1.5  # shape (boost its contribution)

        # total_loss = w1 * loss1 + w2 * loss2 + w3 * loss3
        # #total_loss = loss1 + loss2 + loss3

        # total_loss.backward()
        # optimizer.step()
        l1 = loss_pathology(out_pathology, pathology_targets)
        l2 = loss_birads(out_birads, birads_targets)
        l3 = loss_shape(out_shape, shape_targets)
        l4 = loss_aux(out_mask, masks.squeeze(1))

        total_loss = l1 + l2 + l3 + 0.5 * l4  # 0.5 is auxiliary loss weight
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ==== Validation ====
    model.eval()
    val_loss = 0

    preds_path, targets_path = [], []
    preds_shape, targets_shape = [], []
    preds_birads, targets_birads = [], []

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)

            pathology_targets = labels[:, 0].unsqueeze(1).float()
            shape_targets = labels[:, 1].long()
            birads_targets = labels[:, 2].long()

            out_pathology, out_birads, out_shape, out_mask = model(images)

            loss1 = loss_pathology(out_pathology, pathology_targets)
            loss2 = loss_birads(out_birads, birads_targets)
            loss3 = loss_shape(out_shape, shape_targets)
            loss4 = loss_aux(out_mask, masks.squeeze(1))

            total_val = loss1 + loss2 + loss3 + 0.5 * loss4  # Weighted aux loss
            val_loss += total_val.item()

            # Pathology
            probs_path = torch.sigmoid(out_pathology)
            preds_path.extend((probs_path > 0.5).long().cpu().numpy().flatten())
            targets_path.extend(pathology_targets.cpu().numpy().flatten())

            # Shape
            preds_shape.extend(out_shape.argmax(1).cpu().numpy())
            targets_shape.extend(shape_targets.cpu().numpy())

            # BI-RADS
            preds_birads.extend(out_birads.argmax(1).cpu().numpy())
            targets_birads.extend(birads_targets.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc_path = accuracy_score(targets_path, preds_path)
    val_acc_shape = accuracy_score(targets_shape, preds_shape)
    val_acc_birads = accuracy_score(targets_birads, preds_birads)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save all accuracies separately
    val_accuracies_pathology.append(val_acc_path)
    val_accuracies_shape.append(val_acc_shape)
    val_accuracies_birads.append(val_acc_birads)

    # --- Print all ---
    print(f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
        f"Pathology Acc={val_acc_path:.4f} | Shape Acc={val_acc_shape:.4f} | BIRADS Acc={val_acc_birads:.4f}")

    # === Early Stopping ===
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     epochs_no_improve = 0
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve >= early_stopping_patience:
    #         print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
    #         break

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60

print(f"Training time: {elapsed_time_minutes:.2f} minutes")

# ==== Test Evaluation (Pathology, Shape, BIRADS) ====
model.eval()

preds_path, targets_path = [], []
preds_shape, targets_shape = [], []
preds_birads, targets_birads = [], []

with torch.no_grad():
    for images, labels, _ in test_loader:  # Masks are not needed during test
        images, labels = images.to(device), labels.to(device)

        out_pathology, out_birads, out_shape, _ = model(images)

        # Pathology
        probs_path = torch.sigmoid(out_pathology)
        preds_path.extend((probs_path > 0.5).long().cpu().numpy().flatten())
        targets_path.extend(labels[:, 0].cpu().numpy().flatten())

        # Shape
        preds_shape.extend(out_shape.argmax(1).cpu().numpy())
        targets_shape.extend(labels[:, 1].cpu().numpy())

        # BI-RADS
        preds_birads.extend(out_birads.argmax(1).cpu().numpy())
        targets_birads.extend(labels[:, 2].cpu().numpy())

# === Print Final Metrics ===

print("\n========= [INFO] Final Test Results =========")

print("\n--- Pathology Classification ---")
print("Accuracy:", accuracy_score(targets_path, preds_path))
print("Classification Report:\n", classification_report(targets_path, preds_path))
print("Confusion Matrix:\n", confusion_matrix(targets_path, preds_path))

print("\n--- Shape Classification ---")
print("Accuracy:", accuracy_score(targets_shape, preds_shape))
print("Classification Report:\n", classification_report(targets_shape, preds_shape))
print("Confusion Matrix:\n", confusion_matrix(targets_shape, preds_shape))

print("\n--- BIRADS Classification ---")
print("Accuracy:", accuracy_score(targets_birads, preds_birads))
print("Classification Report:\n", classification_report(targets_birads, preds_birads))
print("Confusion Matrix:\n", confusion_matrix(targets_birads, preds_birads))

# ==== Save Training Curves ====
os.makedirs("tuned-stacked-ensemble-plots", exist_ok=True)
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(18, 8))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies_pathology, label='Pathology Acc')
plt.plot(epochs_range, val_accuracies_shape, label='Shape Acc')
plt.plot(epochs_range, val_accuracies_birads, label='BIRADS Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracies Curve')
plt.legend()

plt.tight_layout()
plt.savefig("tuned-stacked-ensemble-plots/training_curves.png")
plt.close()

print("[INFO] Training plots saved to final-stacked-ensemble-plots")

torch.save(model.state_dict(), "saads_tuned_stacked_ensemble_model.pth")
print("[INFO] Model saved as tuned_stacked_ensemble_model.pth")
