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
import torchvision.models as models


print("[INFO] Libraries imported successfully.")

# -------------------- Dataset --------------------
class ShapeMaskDataset(Dataset):
    def __init__(self, mask_dir, masked_img_dir, csv_file, transform_mask=None, transform_masked=None):
        print(f"[INFO] Initializing ShapeMaskDataset with:")
        print(f"       Masks dir: {mask_dir}")
        print(f"       Masked images dir: {masked_img_dir}")

        self.mask_dir = mask_dir
        self.masked_img_dir = masked_img_dir
        self.data = pd.read_csv(csv_file)

        valid_pathologies = {"malignant", "benign"}
        self.data = self.data[self.data["pathology"].str.lower().isin(valid_pathologies)].reset_index(drop=True)
        print(f"[INFO] Filtered data rows: {len(self.data)}")

        # Match on base name (e.g., 'Mass-Test_P_00016_LEFT_CC_1')
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = []
        for f in mask_files:
            base = os.path.splitext(f)[0].split('#')[0]
            if base in set(self.data["image_name"]):
                self.image_files.append(f)
        print(f"[INFO] Matched image files: {len(self.image_files)}")

        self.transform_mask = transform_mask
        self.transform_masked = transform_masked
        self.pathology_classes = {"benign": 0, "malignant": 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        mask_name = self.image_files[idx]
        base_name = os.path.splitext(mask_name)[0].split('#')[0]

        mask_path = os.path.join(self.mask_dir, mask_name)
        masked_img_path = os.path.join(self.masked_img_dir, mask_name)

        mask = Image.open(mask_path).convert("L")
        masked_img = Image.open(masked_img_path).convert("RGB")

        if self.transform_mask:
            mask = self.transform_mask(mask)
        if self.transform_masked:
            masked_img = self.transform_masked(masked_img)

        row = self.data[self.data["image_name"] == base_name].iloc[0]
        label = self.pathology_classes.get(str(row["pathology"]).lower(), 1)

        return (masked_img, mask), label

    def get_all_labels(self):
        return [self.pathology_classes.get(str(row["pathology"]).lower(), 1)
                for _, row in self.data.iterrows()]

# -------------------- Simple Shape CNN --------------------
class DualBranchCNN(nn.Module):
    def __init__(self, shape_input_channels=1, num_classes=2):
        super(DualBranchCNN, self).__init__()

        # Shape Branch (simple custom CNN)
        self.shape_branch = nn.Sequential(
            nn.Conv2d(shape_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Masked Image Branch (ResNet18 backbone)
        self.texture_branch = models.resnet18(pretrained=True)
        self.texture_branch.fc = nn.Identity()  # remove final classification layer
        texture_feature_dim = self.texture_branch.fc.in_features if hasattr(self.texture_branch.fc, 'in_features') else 512

        # Final Classifier (after concatenation)
        self.classifier = nn.Sequential(
            nn.Linear(64 + texture_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, shape_img, masked_img):
        # Shape: (B, 1, 224, 224)
        shape_feat = self.shape_branch(shape_img)   # (B, 64, 1, 1)
        shape_feat = shape_feat.view(shape_feat.size(0), -1)  # (B, 64)

        # Masked: (B, 3, 224, 224)
        texture_feat = self.texture_branch(masked_img)  # (B, 512)

        # Combine
        combined = torch.cat((shape_feat, texture_feat), dim=1)  # (B, 576)
        output = self.classifier(combined)  # (B, num_classes)
        return output

# -------------------- Evaluation --------------------
def evaluate_model(model, loader, device):
    print("[INFO] Starting model evaluation.")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (masked_images, masks), labels in loader:
            masked_images = masked_images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(masks, masked_images)
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

    transform_masked = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = ShapeMaskDataset(
        mask_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masks",
        masked_img_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masked_images",
        csv_file="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv",
        transform_mask=transform_mask,
        transform_masked=transform_masked
    )

    test_set = ShapeMaskDataset(
        mask_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masks",
        masked_img_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masked_images",
        csv_file="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv",
        transform_mask=transform_mask,
        transform_masked=transform_masked
    )

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchCNN(num_classes=2).to(device)

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
        for (masked_images, masks), labels in train_loader:
            masked_images = masked_images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(masks, masked_images)
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
            for (masked_images, masks), labels in val_loader:
                masked_images = masked_images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(masks, masked_images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "dual_branch_best_model.pth")
            print("[INFO] Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("DualBranchCNN Loss")
    plt.savefig("dual_branch_loss_curve.png")
    print("[INFO] Loss curve saved.")

    print("[INFO] Evaluating on test set...")
    evaluate_model(model, test_loader, device)

    features, labels = [], []

    model.eval()
    with torch.no_grad():
        for (masked_images, masks), lbls in tqdm(val_loader):
            masked_images = masked_images.to(device)
            masks = masks.to(device)
            feats = model(masks, masked_images)  # Pass shape first, texture second
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
    plt.title("PCA of DualBranch Features")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.colorbar(label="Pathology (0=Benign, 1=Malignant)")
    plt.tight_layout()
    plt.savefig("dual_branch_pca.png")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(7, 5))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("t-SNE of DualBranch Features")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.colorbar(label="Pathology (0=Benign, 1=Malignant)")
    plt.tight_layout()
    plt.savefig("dual_branch_tsne.png")
    plt.show()

    print("Silhouette Score:", silhouette_score(features_scaled, labels))
    print("Davies-Bouldin Index:", davies_bouldin_score(features_scaled, labels))


if __name__ == "__main__":
    main()
