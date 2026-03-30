import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix
import torch.nn.functional as F
import time

class CBISDataLoader(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image_name_no_ext = os.path.splitext(image_name)[0]

        image = Image.open(image_path).convert("L")
        image = np.array(image)
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel, was required for ResNet
        image = Image.fromarray(image)

        matching_rows = self.data[self.data["image_name"] == image_name_no_ext]
        if matching_rows.empty:
            print(f"Warning: Image '{image_name_no_ext}' not found in CSV. Skipping.")
            return self.__getitem__((idx + 1) % len(self.image_files))  # Pick next valid index

        row = matching_rows.iloc[0]

        pathology = 1 if str(row["pathology"]).lower() == "malignant" else 0

        # TODO: Cater More Shapes here.
        shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        mass_shape = shape_classes.get(str(row["mass shape"]).lower(), 3)  # Default to "irregular"

        # BI-RADS with One-hot encoding
        birads = int(row["assessment"])
        birads_one_hot = torch.zeros(6)  # One-hot vector for 6 classes (0-5)
        birads_one_hot[birads] = 1.0

        labels = torch.cat([torch.tensor([pathology, mass_shape]), birads_one_hot])

        if self.transform:
            image = self.transform(image)

        return image, labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_resnet_model(model_name):
    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers except FC
    return model

resnet50 = load_resnet_model("resnet50")
resnet101 = load_resnet_model("resnet101")
resnet152 = load_resnet_model("resnet152")

class StackedEnsemble(nn.Module):
    def __init__(self):
        super(StackedEnsemble, self).__init__()
        self.resnet50 = resnet50
        self.resnet101 = resnet101
        self.resnet152 = resnet152

        # Fully Connected Layer: (Pathology: 2, BI-RADS: 6, Shape: 4)
        self.fc = nn.Linear(3 * 2048, 12)  # 12 output neurons (2+6+4)

    def forward(self, x):
        x1 = self.resnet50(x).flatten(start_dim=1)
        x2 = self.resnet101(x).flatten(start_dim=1)
        x3 = self.resnet152(x).flatten(start_dim=1)

        combined_features = torch.cat((x1, x2, x3), dim=1)
        output = self.fc(combined_features)

        return output[:, :2], output[:, 2:8], output[:, 8:]  # Pathology (2), BI-RADS (6), Shape (4)

def train(model, train_loader, optimizer, criterion1, criterion2, criterion3, device):
    model.train()
    total_loss = 0

    for images, labels in train_loader:    
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        pathology_pred, birads_pred, shape_pred = model(images)

        # Split labels
        pathology_label = labels[:, 0].long()
        shape_label = labels[:, 1].long()
        birads_label = labels[:, 2:].float()  # One-hot encoded BI-RADS

        pathology_loss = criterion1(pathology_pred, pathology_label)
        birads_loss = criterion2(birads_pred, birads_label)
        shape_loss = criterion3(shape_pred, shape_label)

        loss = pathology_loss + birads_loss + shape_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    pathology_preds, birads_preds, shape_preds = [], [], []
    pathology_labels, birads_labels, shape_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            pathology_pred, birads_pred, shape_pred = model(images)

            pathology_preds.extend(torch.argmax(pathology_pred, dim=1).cpu().numpy())
            birads_preds.extend(torch.argmax(birads_pred, dim=1).cpu().numpy())
            shape_preds.extend(torch.argmax(shape_pred, dim=1).cpu().numpy())

            pathology_labels.extend(labels[:, 0].cpu().numpy())
            birads_labels.extend(torch.argmax(labels[:, 2:], dim=1).cpu().numpy())  # Convert one-hot to class index
            shape_labels.extend(labels[:, 1].cpu().numpy())

    pathology_acc = accuracy_score(pathology_labels, pathology_preds)
    birads_acc = accuracy_score(birads_labels, birads_preds)
    shape_acc = accuracy_score(shape_labels, shape_preds)

    print(f"Pathology Accuracy: {pathology_acc:.4f}")
    print(f"BI-RADS Accuracy: {birads_acc:.4f}")
    print(f"Shape Accuracy: {shape_acc:.4f}")

    print("\nBI-RADS Classification Report:")
    print(classification_report(birads_labels, birads_preds, digits=4))

    print("\nShape Classification Report:")
    print(classification_report(shape_labels, shape_preds, digits=4))

    print("\nBI-RADS Confusion Matrix:")
    print(confusion_matrix(birads_labels, birads_preds))

    print("\nShape Confusion Matrix:")
    print(confusion_matrix(shape_labels, shape_preds))

    return pathology_acc, birads_acc, shape_acc

def main():
    train_images = "/ediss_data/ediss2/xai-texture/data/CBIS-DDSM-Patches-Mass-Context/train/masks"
    test_images = "/ediss_data/ediss2/xai-texture/data/CBIS-DDSM-Patches-Mass-Context/test/masks"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS-DDSM-Patches-Mass-Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

    train_dataset = CBISDataLoader(train_images, csv_file, transform=transform)
    test_dataset = CBISDataLoader(test_images, csv_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedEnsemble().to(device)

    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion2 = nn.BCEWithLogitsLoss()  # Used for one-hot encoded labels
    criterion3 = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_time = time.time() 
    for epoch in range(80):
        loss = train(model, train_loader, optimizer, criterion1, criterion2, criterion3, device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/80], Loss: {loss:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    minutes = training_time // 60
    seconds = training_time % 60
    print(f"Total Training Time: {int(minutes)} minutes {seconds:.2f} seconds")
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()