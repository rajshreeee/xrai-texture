import os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


print("[INFO] Libraries imported successfully.")

# -------------------- Dataset --------------------
class ShapeMaskDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        print(f"[INFO] Initializing dataset for: {image_dir}")
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")

        valid_pathologies = {"malignant", "benign"}
        self.data = self.data[self.data["pathology"].str.lower().isin(valid_pathologies)].reset_index(drop=True)
        print(f"[INFO] Filtered data rows: {len(self.data)}")

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

        image = Image.open(image_path).convert("L")  # Binary mask

        matching_rows = self.data[self.data["image_name"] == base_name]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(str(row["pathology"]).lower(), 1)

        if self.transform:
            image = self.transform(image)
        return image, pathology

    def get_all_labels(self):
        return [self.pathology_classes.get(str(row["pathology"]).lower(), 1)
                for _, row in self.data.iterrows()]

# -------------------- Simple Shape CNN --------------------
class ShapeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        x = self.flatten(x)
        features = self.relu(self.fc1(x))
        if return_features:
            return features
        x = self.dropout(features)
        return self.fc2(x)

# -------------------- Evaluation --------------------
def evaluate_model(model, loader, device):
    print("[INFO] Starting model evaluation.")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true)
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# -------------------- Main --------------------
def main():
    train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masks"
    test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masks"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = ShapeMaskDataset(train_path, csv_file, transform)
    test_set = ShapeMaskDataset(test_path, csv_file, transform)

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeCNN().to(device)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_set.get_all_labels())
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(20):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "shape_best_model.pth")
            print("[INFO] Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Shape-based CNN Loss")
    plt.savefig("shape_loss_curve.png")
    print("[INFO] Loss curve saved.")

    print("[INFO] Evaluating on test set...")
    evaluate_model(model, test_loader, device)

    features, labels = [], []

    model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(val_loader):
            images = images.to(device)
            feats = model(images, return_features=True)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    features = np.vstack(features)
    labels = np.array(labels)

   
    features_scaled = StandardScaler().fit_transform(features)

    # PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features_scaled)

    plt.figure(figsize=(7, 5))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("PCA of ShapeCNN Features")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.colorbar(label="Pathology (0=Benign, 1=Malignant)")
    plt.tight_layout()
    plt.savefig("shape_pca.png")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(7, 5))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("t-SNE of ShapeCNN Features")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.colorbar(label="Pathology (0=Benign, 1=Malignant)")
    plt.tight_layout()
    plt.savefig("shape_tsne.png")
    plt.show()

    print("Silhouette Score:", silhouette_score(features_scaled, labels))
    print("Davies-Bouldin Index:", davies_bouldin_score(features_scaled, labels))

if __name__ == "__main__":
    main()
