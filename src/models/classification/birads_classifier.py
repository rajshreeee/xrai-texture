import os, torch, timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# -------------------- Dataset --------------------
class BIRADSDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        # Get all image files from the directory (assuming file names match CSV's "image_name" field)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_name_no_ext = os.path.splitext(image_name)[0]

        image = Image.open(image_path).convert("L")
        # Convert grayscale to 3-channel by stacking the single channel
        image = np.stack([np.array(image)] * 3, axis=-1)
        image = Image.fromarray(image)

        # Find the matching row in the CSV file
        row = self.data[self.data["image_name"] == image_name_no_ext].iloc[0]
        # Return label as an integer
        label = int(row["assessment"])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_all_labels(self):
        return [int(row["assessment"]) for _, row in self.data.iterrows()]


# -------------------- Model --------------------
class BIRADSClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", use_radimagenet=False, radimagenet_weights_path=None):
        super().__init__()
        if use_radimagenet:
            # Create backbone without ImageNet weights
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            if radimagenet_weights_path is None:
                raise ValueError("Please provide radimagenet_weights_path when use_radimagenet is True.")
            state_dict = torch.load(radimagenet_weights_path, map_location="cpu")
            # Load RadImageNet weights; using strict=False to account for any naming differences
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to ImageNet pretrained weights if not using RadImageNet
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]
        self.fc = nn.Linear(feature_dim, 6)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# -------------------- Evaluation --------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # labels are integers already
            true = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_preds.extend(preds)
            all_labels.extend(true)

    print("BI-RADS Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


# -------------------- Main --------------------
def main():
    # Paths (update these as needed)
    train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/train/images"
    test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/test/images"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT_KAGGLE.csv"

    # Data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    # Create datasets and dataloaders
    train_set = BIRADSDataset(train_path, csv_file, transform)
    test_set = BIRADSDataset(test_path, csv_file, transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Set use_radimagenet=True and provide the path to the RadImageNet weights file.
    radimagenet_weights_path = "/ediss_data/ediss2/xai-texture/src/models/classification/ResNet50.pt"  # <-- Update with actual file path
    model = BIRADSClassifier(backbone_name="resnet50", use_radimagenet=False,
                             radimagenet_weights_path=radimagenet_weights_path).to(device)

    # Fine-tuning: In this case, we fine-tune the entire model.
    for param in model.parameters():
        param.requires_grad = True

    # Compute class weights for CrossEntropyLoss
    train_labels = train_set.get_all_labels()
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "birads_model_radimagenet.pth")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()