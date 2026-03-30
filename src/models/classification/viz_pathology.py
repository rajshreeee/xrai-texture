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
from torchviz import make_dot


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

class AttentionFusion(nn.Module):
    def __init__(self, shape_dim=64, texture_dim=512, fused_dim=128):
        super().__init__()
        self.shape_proj = nn.Linear(shape_dim, fused_dim)
        self.texture_proj = nn.Linear(texture_dim, fused_dim)

        self.attn_fc = nn.Sequential(
            nn.Linear(fused_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, shape_feat, texture_feat):
        shape_proj = self.shape_proj(shape_feat)     # [B, fused_dim]
        texture_proj = self.texture_proj(texture_feat)

        combined = torch.cat((shape_proj, texture_proj), dim=1)
        weights = self.attn_fc(combined)             # [B, 2]

        shape_w = weights[:, 0].unsqueeze(1)
        texture_w = weights[:, 1].unsqueeze(1)

        fused = shape_w * shape_proj + texture_w * texture_proj  # [B, fused_dim]
        return fused
    
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
        self.attn_fusion =  AttentionFusion(shape_dim=64, texture_dim=512, fused_dim=128)
        self.texture_branch = models.resnet18(pretrained=True)

        # unfreezing resnet to learn. 
        for param in self.texture_branch.parameters():
            param.requires_grad = True

        self.texture_branch.fc = nn.Identity()  # remove final classification layer
        texture_feature_dim = self.texture_branch.fc.in_features if hasattr(self.texture_branch.fc, 'in_features') else 512

        # Final Classifier (after concatenation)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
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
        fused_feat = self.attn_fusion(shape_feat, texture_feat)
        output = self.classifier(fused_feat)  # (B, num_classes)
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


def load_model_and_evaluate(model_path, model_class, device, loader, name=""):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model for {name} set.")
    evaluate_model(model, loader, device)

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
        mask_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masks",
        masked_img_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images",
        csv_file="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv",
        transform_mask=transform_mask,
        transform_masked=transform_masked
    )

    test_set = ShapeMaskDataset(
        mask_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masks",
        masked_img_dir="/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masked_images",
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

    model = DualBranchCNN(num_classes=2)
    model.eval()

    # Create dummy inputs: 
    # Shape image (grayscale): [B, 1, 224, 224]
    # Masked image (RGB): [B, 3, 224, 224]
    dummy_shape = torch.randn(1, 1, 224, 224)
    dummy_masked = torch.randn(1, 3, 224, 224)

    # Pass the dummy inputs through the model
    output = model(dummy_shape, dummy_masked)

    # Generate the graph
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Save to file
    dot.format = "png"
    # dot.directory = "model_graph"
    dot.render("dual_branch_attention")

    print("✅ Model graph saved to: model_graph/dual_branch_attention.png")
   


if __name__ == "__main__":
    main()
