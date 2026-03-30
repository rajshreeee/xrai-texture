import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import timm
import time


# **Dataset Paths**
train_images = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masks"
test_images = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masks"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

# **EfficientNet & InceptionResNet Loading**
def load_custom_model(model_name):
    """Loads a pre-trained model and removes the classification layer."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove FC layer
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor layers
    return model


# **Dataset Class**
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
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel
        image = Image.fromarray(image)

        matching_rows = self.data[self.data["image_name"] == image_name_no_ext]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]

        pathology_classes = {"malignant": 2, "benign": 1, "benign_without_callback": 0}
        pathology = pathology_classes.get(str(row["pathology"]).lower(), 1)

        shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "spiculated": 3, "ill-defined": 4, "irregular": 5}
        mass_shape = shape_classes.get(str(row["mass shape"]).lower(), 5)

        birads = int(row["assessment"])
        birads_one_hot = torch.zeros(6)
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


class StackedEnsemble(nn.Module):
    def __init__(self):
        super(StackedEnsemble, self).__init__()

        self.efficientnet = load_custom_model("tf_efficientnet_b7")
        self.inception_resnet = load_custom_model("inception_resnet_v2")

        efficientnet_features = self.efficientnet(torch.randn(1, 3, 224, 224)).shape[1]
        inception_features = self.inception_resnet(torch.randn(1, 3, 224, 224)).shape[1]

        total_features = efficientnet_features + inception_features
        print(f"EfficientNet Features: {efficientnet_features}, InceptionResNet Features: {inception_features}")
        print(f"Total Combined Features: {total_features}")

        self.fc = nn.Linear(total_features, 15)

    def forward(self, x):
        x1 = self.efficientnet(x).flatten(start_dim=1)
        x2 = self.inception_resnet(x).flatten(start_dim=1)

        combined_features = torch.cat((x1, x2), dim=1)
        output = self.fc(combined_features)

        return output[:, :3], output[:, 3:9], output[:, 9:]
    

def train(model, train_loader, optimizer, criterion1, criterion2, criterion3, device):
    model.train()
    total_loss = 0

    for images, labels in train_loader:    
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        pathology_pred, birads_pred, shape_pred = model(images)

        pathology_label = labels[:, 0].long()  # Pathology: 3 classes
        shape_label = labels[:, 1].long()  # Shape: 6 classes
        birads_label = labels[:, 2:].float()  # BI-RADS: 6 one-hot encoded classes

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
    # train_images = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masks"
    # test_images = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masks"
    # csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
    train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/train/images"
    test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/test/images"
    csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context_Kaggle/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT_KAGGLE.csv"

    train_dataset = CBISDataLoader(train_images, csv_file, transform=transform)
    test_dataset = CBISDataLoader(test_images, csv_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedEnsemble().to(device)

    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)  # Pathology
    criterion2 = nn.BCEWithLogitsLoss()  # BI-RADS (one-hot)
    criterion3 = nn.CrossEntropyLoss(label_smoothing=0.1)  # Shape

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(20):
        loss = train(model, train_loader, optimizer, criterion1, criterion2, criterion3, device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/20], Loss: {loss:.4f}")

    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()