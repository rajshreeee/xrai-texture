import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import torch.nn.functional as F
import pandas as pd


# ==== Paths and config ====
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_aug_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Dataset ====
class CBISDDSMDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')
        valid_pathologies = {"malignant", "benign"}
        valid_shapes = {"round", "oval", "lobulated", "irregular"}
        valid_birads = {2, 3, 4, 5, 6}

        self.data = self.data[
            (self.data['pathology'].isin(valid_pathologies)) &
            (self.data['mass shape'].str.lower().isin(valid_shapes)) &
            (self.data['assessment'].isin(valid_birads))
        ].reset_index(drop=True)

        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.pathology_classes = {"benign": 0, "malignant": 1}
        self.shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        base_name = os.path.splitext(image_name)[0].split('#')[0]

        image = Image.open(image_path).convert("RGB")
        matching_rows = self.data[self.data['image_name'] == base_name]

        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))

        row = matching_rows.iloc[0]

        pathology = self.pathology_classes.get(row['pathology'], 1)
        shape = self.shape_classes.get(str(row['mass shape']).lower(), 3)
        birads = self.birads_mapping.get(int(row['assessment']), 4)

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([pathology, shape, birads], dtype=torch.long)
        return image, labels


# ==== Model ====
def load_resnet_model(model_name):
    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    modules = list(model.children())[:-2]
    return nn.Sequential(*modules)

class StackedEnsemble(nn.Module):
    def __init__(self):
        super(StackedEnsemble, self).__init__()
        self.resnet50 = load_resnet_model("resnet50")
        self.resnet101 = load_resnet_model("resnet101")
        self.resnet152 = load_resnet_model("resnet152")

        # Adaptive pooling to reduce 3D feature maps to 1D vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer: (Pathology: 1, BI-RADS: 4, Shape: 4)
        self.fc = nn.Linear(3 * 2048, 1 + 4 + 4) 

    def forward(self, x):
        x1 = self.pool(self.resnet50(x)).flatten(start_dim=1)
        x2 = self.pool(self.resnet101(x)).flatten(start_dim=1)
        x3 = self.pool(self.resnet152(x)).flatten(start_dim=1)

        combined_features = torch.cat((x1, x2, x3), dim=1)
        output = self.fc(combined_features)

        out_pathology = output[:, :1]     # 1 neuron
        out_birads = output[:, 1:5]        # 4 neurons
        out_shape = output[:, 5:]          # 4 neurons

        return out_pathology, out_birads, out_shape


# ==== Transform and Load Data ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_dataset = CBISDDSMDataset(test_path, csv_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==== Load Model ====
model = StackedEnsemble().to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("saads_stacked_ensemble_model.pth"))
model.eval()

# ==== Evaluation ====
preds_path, targets_path = [], []
preds_shape, targets_shape = [], []
preds_birads, targets_birads = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out_pathology, out_birads, out_shape = model(images)

        probs_pathology = torch.sigmoid(out_pathology)
        pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
        preds_path.extend(pred_labels_path.flatten())
        targets_path.extend(labels[:, 0].cpu().numpy())

        pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
        preds_shape.extend(pred_labels_shape.flatten())
        targets_shape.extend(labels[:, 1].cpu().numpy())

        pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
        preds_birads.extend(pred_labels_birads.flatten())
        targets_birads.extend(labels[:, 2].cpu().numpy())

# ==== Print Metrics ====
print("\n========= [INFO] Test Results =========")

print("\n--- Pathology ---")
print("Accuracy:", accuracy_score(targets_path, preds_path))
print(classification_report(targets_path, preds_path))
print("Confusion Matrix:\n", confusion_matrix(targets_path, preds_path))

print("\n--- Shape ---")
print("Accuracy:", accuracy_score(targets_shape, preds_shape))
print(classification_report(targets_shape, preds_shape))
print("Confusion Matrix:\n", confusion_matrix(targets_shape, preds_shape))

print("\n--- BIRADS ---")
print("Accuracy:", accuracy_score(targets_birads, preds_birads))
print(classification_report(targets_birads, preds_birads))
print("Confusion Matrix:\n", confusion_matrix(targets_birads, preds_birads))
