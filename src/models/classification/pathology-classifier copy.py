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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter


# ========= Config =========
train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
radimagenet_weights_path = None  # not using RadImageNet for now
batch_size = 32
num_epochs = 30
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Dataset =========
class PathologyDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        print(f"[INFO] Initializing dataset for: {image_dir}")
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")
        
        # Treat 'benign_without_callback' as 'benign'
        self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')

        valid_pathologies = {"malignant", "benign"}
        self.data = self.data[self.data["pathology"].isin(valid_pathologies)].reset_index(drop=True)
        print(f"[INFO] Filtered data rows (only benign/malignant): {len(self.data)}")
        
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"[INFO] Found {len(all_files)} images in directory.")

        valid_names = set(self.data["image_name"])
        self.image_files = [f for f in all_files if os.path.splitext(f)[0].split('#')[0] in valid_names]
        print(f"[INFO] Valid images (matched with CSV): {len(self.image_files)}")

        self.transform = transform
        self.pathology_classes = {"benign": 0, "malignant": 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        base_name = os.path.splitext(image_name)[0].split('#')[0]

        image = Image.open(image_path).convert("RGB")

        matching_rows = self.data[self.data["image_name"] == base_name]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(row["pathology"], 1)

        if self.transform:
            #image = self.transform(image)
            augmented = self.transform(image=np.array(image))
            image = augmented["image"]

        return image, pathology

    def get_all_labels(self):
        return [self.pathology_classes.get(row["pathology"], 1)
                for _, row in self.data.iterrows()]

# ========= Model =========
class BIRADSClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ========= Prepare Data =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# For validation/test, only resize + normalize
val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# FOR TRAIN TEST CONCATENATION

train_dataset = PathologyDataset(train_path, csv_file, transform=train_transform)
test_dataset_extra = PathologyDataset(test_path, csv_file, transform=val_test_transform)
combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])

# Manually stratified split
all_labels = []
for idx in range(len(combined_dataset)):
    _, label = combined_dataset[idx]
    all_labels.append(label)

indices = np.arange(len(combined_dataset))

# 80% train, 20% temp (val+test)
train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, all_labels, stratify=all_labels, test_size=0.2, random_state=42
)

# Split temp into 50%-50% val and test (i.e., 10%-10% overall)
val_idx, test_idx = train_test_split(
    temp_idx, stratify=temp_labels, test_size=0.5, random_state=42
)

# ========= 4. Create DataLoaders =========
train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")


# FOR TRAIN ONLY SPLIT

# dataset = PathologyDataset(train_path, csv_file, transform=train_transform)
 
# # Manually stratified split
# all_labels = []
# for idx in range(len(dataset)):
#     _, label = dataset[idx]
#     all_labels.append(label)
 
# indices = np.arange(len(dataset))
# train_idx, temp_idx, _, temp_labels = train_test_split(
#     indices, all_labels, stratify=all_labels, test_size=0.2, random_state=42)
 
# val_idx, test_idx = train_test_split(
#     temp_idx, stratify=temp_labels, test_size=0.5, random_state=42)
 
# train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
 
# print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")


def get_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.sum()

shape_weights = get_class_weights(all_labels[:, 1], 4).to(device)
birads_weights = get_class_weights(all_labels[:, 2], 4).to(device)

# ========= Training =========
train_losses = []
val_losses = []
val_accuracies = []

model = BIRADSClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Lists to save losses and accuracies
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Compute training accuracy
    train_preds, train_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            pred_labels = (probs > 0.5).long().cpu().numpy()
            train_preds.extend(pred_labels.flatten())
            train_targets.extend(labels.cpu().numpy().flatten())
    train_acc = accuracy_score(train_targets, train_preds)

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            pred_labels = (probs > 0.5).long().cpu().numpy()
            val_preds.extend(pred_labels.flatten())
            val_targets.extend(labels.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)

    # Step scheduler
    scheduler.step(avg_val_loss)

    # Save stats
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print stats
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"[INFO] No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stopping_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}!")
            break

# ========= Final Testing =========
model.eval()
preds, targets = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        pred_labels = (probs > 0.5).long().cpu().numpy()
        preds.extend(pred_labels.flatten())
        targets.extend(labels.cpu().numpy().flatten())

print("\nFinal Test Accuracy:", accuracy_score(targets, preds))
print("Classification Report:\n", classification_report(targets, preds))
print("Confusion Matrix:\n", confusion_matrix(targets, preds))

# ========= Save Training Curves =========
os.makedirs("pathology-classifier-plots", exist_ok=True)  # Folder to save plots

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("pathology-classifier-plots/training_curves.png")
plt.close()

print("[INFO] Training curves saved to training_curves.png")
