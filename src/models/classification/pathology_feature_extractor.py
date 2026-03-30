import os, torch, timm
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
 
print("[INFO] Libraries imported successfully.")
 
# -------------------- Dataset --------------------
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
        
        # Compute base names from image files (remove extension and anything after '#')
        # and then filter image files to only those that have a matching CSV row.
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
        # Extract base name: remove extension and any augmentation suffix after '#'
        base_name = os.path.splitext(image_name)[0].split('#')[0]
        
        # Open image in RGB mode since images are colored now
        image = Image.open(image_path).convert("RGB")
        
        # Retrieve matching row in CSV using the base name
        matching_rows = self.data[self.data["image_name"] == base_name]
        if matching_rows.empty:
            # If not found, try the next image cyclically
            return self.__getitem__((idx + 1) % len(self.image_files))
        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(str(row["pathology"]).lower(), 1)
        if self.transform:
            image = self.transform(image)
        return image, pathology
 
    def get_all_labels(self):
        return [self.pathology_classes.get(str(row["pathology"]).lower(), 1)
                for _, row in self.data.iterrows()]
 
 
# -------------------- Model --------------------
# class BIRADSClassifier(nn.Module):
#     def __init__(self, backbone_name="resnet50", use_radimagenet=False, radimagenet_weights_path=None, num_classes=3):
#         super().__init__()
#         print(f"[INFO] Initializing model: {backbone_name}, RadImageNet: {use_radimagenet}")
#         if use_radimagenet:
#             self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
#             if radimagenet_weights_path is None:
#                 raise ValueError("Please provide radimagenet_weights_path when use_radimagenet is True.")
#             print(f"[INFO] Loading RadImageNet weights from: {radimagenet_weights_path}")
#             state_dict = torch.load(radimagenet_weights_path, map_location="cpu")
#             self.backbone.load_state_dict(state_dict, strict=False)
#         else:
#             self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
 
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, 224, 224)
#             feature_dim = self.backbone(dummy_input).shape[1]
#             print(f"[INFO] Feature dimension from backbone: {feature_dim}")
#         self.fc = nn.Linear(feature_dim, num_classes)
 
#     def forward(self, x):
#         features = self.backbone(x)
#         return self.fc(features)
 
 
class BIRADSClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", use_radimagenet=False, radimagenet_weights_path=None, num_classes=2):
        super().__init__()
        print(f"[INFO] Initializing model: {backbone_name}, RadImageNet: {use_radimagenet}")
 
        self.backbone = timm.create_model(backbone_name, pretrained=not use_radimagenet, num_classes=0, features_only=False)
 
        if use_radimagenet:
            if radimagenet_weights_path is None:
                raise ValueError("Please provide radimagenet_weights_path when use_radimagenet is True.")
            print(f"[INFO] Loading RadImageNet weights from: {radimagenet_weights_path}")
            state_dict = torch.load(radimagenet_weights_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)
 
        # Freeze first 4 residual blocks (except BatchNorms)
        frozen_blocks = 0
        for name, module in self.backbone.named_children():
            if "layer" in name and frozen_blocks < 4:
                for param in module.parameters():
                    param.requires_grad = False
                # Keep BatchNorm layers trainable
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        for p in m.parameters():
                            p.requires_grad = True
                frozen_blocks += 1
 
        # Determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]
            print(f"[INFO] Feature dimension from backbone: {feature_dim}")
 
        # Add classifier head: FC(1024) → Dropout(0.3) → ReLU → FC(num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
 
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x)
 
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
            true = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_preds.extend(preds)
            all_labels.extend(true)
    print("Pathology Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
 
# -------------------- Main --------------------
def main():
    print("[INFO] Starting main...")
 
    train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images"
    test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masked_images"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
 
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    print("[INFO] Transforms defined.")
 
    print("[INFO] Loading datasets...")
    train_set = PathologyDataset(train_path, csv_file, transform)
    test_set = PathologyDataset(test_path, csv_file, transform)
 
    # Split the training set into training and validation subsets (80/20 split)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    print(f"[INFO] Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")
 
    print("[INFO] Creating dataloaders...")
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
 
    radimagenet_weights_path = "/ediss_data/ediss2/xai-texture/src/models/classification/ResNet50.pt"
    print("[INFO] Initializing model...")
    model = BIRADSClassifier(backbone_name="resnet50", use_radimagenet=False,
                             radimagenet_weights_path=radimagenet_weights_path, num_classes=2)
 
    state_dict = torch.load("best_model.pth", map_location="cpu")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # strip 'module.'
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to("cpu")
    print("[INFO] Starting feature extraction on validation set...")
    
    features = []
    labels = []
    for idx, (images, lbls) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            output = model.extract_features(images)  # shape: [batch_size, feature_dim]
            features.append(output.cpu().numpy())   # move to CPU if using GPU
            labels.extend(lbls.numpy())

        if idx % 10 == 0:
            print(f"[INFO] Processed batch {idx+1}/{len(val_loader)}")

    print(f"[INFO] Feature extraction completed. Total samples: {len(labels)}")
    features = np.concatenate(features)
    labels = np.array(labels)

    # ---------------- PCA ----------------
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    plt.figure(figsize=(8,6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("PCA: ResNet Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Pathology (0: Benign, 1: Malignant)")
    plt.savefig("pca_resnet_features.png")
    plt.show()

    # ---------------- t-SNE ----------------
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("t-SNE: ResNet Features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label="Pathology (0: Benign, 1: Malignant)")
    plt.savefig("tsne_resnet_features.png")
    plt.show()
 
if __name__ == "__main__":
    print("Starting The Classifier")
    main()