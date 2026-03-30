import time
import os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torchvision.models as models
import timm


class PathologyDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        print(f"[INFO] Initializing dataset for: {image_dir}")
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")
        
        valid_pathologies = {"malignant", "benign"}
        self.data = self.data[self.data["pathology"].str.lower().isin(valid_pathologies)].reset_index(drop=True)
        print(f"[INFO] Filtered data rows: {len(self.data)}")
        
        # List all image files from the directory
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"[INFO] Found {len(all_files)} images in directory.")
        
        # Compute base names (remove extension and anything after '#')
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
        if self.transform:
            image = self.transform(image)
        
        matching_rows = self.data[self.data["image_name"] == base_name]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))
        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(str(row["pathology"]).lower(), 1)
        return image, pathology

    def get_all_labels(self):
        return [self.pathology_classes.get(str(row["pathology"]).lower(), 1)
                for _, row in self.data.iterrows()]
    
class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes=2, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, pre_logits: bool = False):
        x = self.pool(x)       # <<< FIX: Pool spatial dimensions
        x = torch.flatten(x, 1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if pre_logits:
            return x
        x = self.fc2(x)
        return x

# ========== 1. Paths and Constants ==========
train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32
num_epochs = 30
image_size = 224
save_dir = "papertraining_outputs"  # Directory to save plots
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. Data Loading ==========

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = PathologyDataset(train_path, csv_file, transform=transform)
test_dataset = PathologyDataset(test_path, csv_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========== 3. Model (ResNet50V2 style) ==========

model = timm.create_model('resnetv2_50x1_bitm', pretrained=True, num_classes=0)

# Freeze all except BatchNorm layers
for name, param in model.named_parameters():
    if 'bn' not in name and 'downsample.1' not in name:
        param.requires_grad = False


model.head = CustomHead(in_features=model.num_features, num_classes=2, dropout=0.3)

model = model.to(device)

# ========== 4. Optimizer, Loss, Scheduler ==========

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

# ========== 5. Training ==========

train_losses, train_accs = [], []
best_train_acc = 0.0
best_model_path = os.path.join(save_dir, "resnet50v2_best_model.pth")

print("[INFO] Starting Training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    start = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accs.append(train_accuracy)

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

    scheduler.step(train_accuracy)

    # Save best model based on training accuracy
    if train_accuracy > best_train_acc:
        torch.save(model.state_dict(), best_model_path)
        best_train_acc = train_accuracy
        print("[INFO] Best model saved!")

    end = time.time()
    print(f"[INFO] Epoch duration: {(end-start)/60:.2f} mins")

# ========== 6. Testing ==========

print("\n[INFO] Loading best model for final testing...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\n[FINAL] Test Accuracy: {test_acc:.4f}")

# Classification report
print("\n[INFO] Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close