import os, torch
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
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_name_no_ext = os.path.splitext(image_name)[0]
        
        image = Image.open(image_path).convert("L")
        # Convert grayscale to 3-channel
        image = np.stack([np.array(image)] * 3, axis=-1)
        image = Image.fromarray(image)

        # Get label from CSV
        row = self.data[self.data["image_name"] == image_name_no_ext].iloc[0]
        label = int(row["assessment"])
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_all_labels(self):
        return [int(row["assessment"]) for _, row in self.data.iterrows()]

# -------------------- Simple CNN Model --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112x112 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 56x56 -> 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 28x28 -> 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                            # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------------------- Evaluation --------------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print("BI-RADS Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# -------------------- Main --------------------
def main():
    # Paths (update these as needed)
    train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masked_images"
    test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masked_images"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

    # Data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Datasets and DataLoaders
    train_set = BIRADSDataset(train_path, csv_file, transform)
    test_set = BIRADSDataset(test_path, csv_file, transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=6).to(device)

    # Compute class weights for CrossEntropyLoss
    train_labels = train_set.get_all_labels()
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training Loop
    for epoch in range(30):
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

    torch.save(model.state_dict(), "simple_cnn_birads.pth")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()