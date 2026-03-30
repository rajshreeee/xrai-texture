import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification
from torchvision import models
from collections import Counter
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
import optuna
import neptune
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_param_importances

run = neptune.init_run(
    project="XRAI-Pipeline/XAI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNzM1ZDY3Ny04ODhjLTQwZDktODQyNC0zMGRhNjZjODgwOTQifQ==",#wrongkeyfromhere",
    name="Swin-Ensemble-Tuning"
)
run["sys/tags"].add(["optuna", "swin-ensemble", "xai"])

train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32
num_epochs = 30
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # pass class weights if needed

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        return bce_loss + dice_loss


class CBISDDSMDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")

        # Treat 'benign_without_callback' as 'benign'
        self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')

        # Filter valid entries
        valid_pathologies = {"malignant", "benign"}
        valid_shapes = {"round", "oval", "lobulated", "irregular"}
        valid_birads = {2, 3, 4, 5}

        self.data = self.data[
            (self.data['pathology'].isin(valid_pathologies)) &
            (self.data['mass shape'].str.lower().isin(valid_shapes)) &
            (self.data['assessment'].isin(valid_birads))
        ].reset_index(drop=True)

        print(f"[INFO] Filtered rows: {len(self.data)}")

        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[INFO] Found {len(self.image_files)} images.")

        # Define mappings
        self.pathology_classes = {"benign": 0, "malignant": 1}
        self.shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_mapping = {2: 0, 3: 1, 4: 2, 5: 3}

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

        birads_raw = int(row['assessment'])
        birads = self.birads_mapping.get(birads_raw, None)
        if birads is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([pathology, shape, birads], dtype=torch.long)

        return image, labels

    def get_all_labels(self):
        labels = []
        for _, row in self.data.iterrows():
            pathology = self.pathology_classes.get(row["pathology"], 1)
            mass_shape = self.shape_classes.get(str(row["mass shape"]).lower(), 3)
            birads_raw = int(row["assessment"])
            birads = self.birads_mapping.get(birads_raw, 4)  # Map BIRADS correctly
            labels.append(torch.tensor([pathology, mass_shape, birads]))
        return labels

def load_swin_model(freeze_until="layers.2"):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=False, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if freeze_until == "none" or freeze_until in name:
            param.requires_grad = True
    return model


class StackedEnsemble(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.swin = load_swin_model()
        self.shared_dim = 768
        self.fc_pathology = nn.Sequential(
            nn.Linear(self.shared_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 1)
        )
        self.fc_birads = nn.Sequential(
            nn.Linear(self.shared_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 4)
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(self.shared_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 4)
        )

    def forward(self, x):
        f = self.swin(x)
        return self.fc_pathology(f), self.fc_birads(f), self.fc_shape(f)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ========= Prepare Data =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CBISDDSMDataset(train_path, csv_file, transform=train_transform)
test_dataset = CBISDDSMDataset(test_path, csv_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

print(f"[INFO] Training samples: {len(train_dataset)}")
print(f"[INFO] Testing samples:  {len(test_dataset)}")

train_labels = train_dataset.get_all_labels()  # list of tensors
train_labels_np = torch.stack(train_labels).numpy()  # shape: (N, 3)


# === Shape Class Weights ===
shape_labels = train_labels_np[:, 1]
shape_classes = np.unique(shape_labels)
shape_weights = compute_class_weight(class_weight='balanced', classes=shape_classes, y=shape_labels)
shape_weights = torch.tensor(shape_weights, dtype=torch.float).to(device)

# === BIRADS Class Weights ===
birads_labels = train_labels_np[:, 2]
birads_classes = np.unique(birads_labels)
birads_weights = compute_class_weight(class_weight='balanced', classes=birads_classes, y=birads_labels)
birads_weights = torch.tensor(birads_weights, dtype=torch.float).to(device)

def objective(trial):
    # Hyperparams
    trial_prefix = f"trial_{trial.number}"
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    freeze_layers_until = trial.suggest_categorical("freeze_layers_until", ["layers.1", "layers.2", "layers.3", "none"])

    model = StackedEnsemble(dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_pathology = BCEDiceLoss()
    loss_shape = FocalLoss(gamma=gamma, weight=shape_weights)
    loss_birads = FocalLoss(gamma=gamma, weight=birads_weights)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    run["params/lr"] = lr
    run["params/weight_decay"] = weight_decay
    run["params/dropout"] = dropout
    run["params/focal_gamma"] = gamma
    run["params/freeze"] = freeze_layers_until

    start = time.time()
    model.train()
    for epoch in range(30):
        preds_p, targets_p = [], []
        preds_s, targets_s = [], []
        preds_b, targets_b = [], []
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            pt = labels[:, 0].float().unsqueeze(1)
            st = labels[:, 1].long()
            bt = labels[:, 2].long()

            optimizer.zero_grad()
            out_p, out_b, out_s = model(images)
            loss = loss_pathology(out_p, pt) + loss_shape(out_s, st) + loss_birads(out_b, bt)
            loss.backward(); optimizer.step()
            running_loss += loss.item()

            preds_p.extend((torch.sigmoid(out_p) > 0.5).long().cpu().numpy().flatten())
            targets_p.extend(pt.cpu().numpy().flatten())
            preds_s.extend(out_s.argmax(1).cpu().numpy())
            targets_s.extend(st.cpu().numpy())
            preds_b.extend(out_b.argmax(1).cpu().numpy())
            targets_b.extend(bt.cpu().numpy())

        acc_p = accuracy_score(targets_p, preds_p)
        acc_s = accuracy_score(targets_s, preds_s)
        acc_b = accuracy_score(targets_b, preds_b)

        run[f"{trial_prefix}/train/acc_pathology"].append(acc_p)
        run[f"{trial_prefix}/train/acc_shape"].append(acc_s)
        run[f"{trial_prefix}/train/acc_birads"].append(acc_b)
        run[f"{trial_prefix}/train/loss"].append(running_loss / len(train_loader))

    # Evaluation
    model.eval()
    preds_p, targets_p = [], []
    preds_s, targets_s = [], []
    preds_b, targets_b = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out_p, out_b, out_s = model(images)
            preds_p.extend((torch.sigmoid(out_p) > 0.5).long().cpu().numpy().flatten())
            targets_p.extend(labels[:, 0].cpu().numpy())
            preds_s.extend(out_s.argmax(1).cpu().numpy())
            targets_s.extend(labels[:, 1].cpu().numpy())
            preds_b.extend(out_b.argmax(1).cpu().numpy())
            targets_b.extend(labels[:, 2].cpu().numpy())

    acc_p = accuracy_score(targets_p, preds_p)
    acc_s = accuracy_score(targets_s, preds_s)
    acc_b = accuracy_score(targets_b, preds_b)
    avg_score = (acc_p + acc_s + acc_b) / 3


    run[f"{trial_prefix}/test/acc_pathology"] = acc_p
    run[f"{trial_prefix}/test/acc_shape"] = acc_s
    run[f"{trial_prefix}/test/acc_birads"] = acc_b
    run[f"{trial_prefix}/test/avg_score"] = avg_score
    run[f"{trial_prefix}/timing/train_time_sec"] = time.time() - start

    return avg_score


study = optuna.create_study(
    direction="maximize",
    study_name="stacked-swin-ensemble-tuning",
    sampler=optuna.samplers.TPESampler(seed=42),
)

# Launch optimization
study.optimize(objective, n_trials=70, show_progress_bar=True)

# Save best trial info
print("\nBest Trial:")
best = study.best_trial
print(f"  Value: {best.value:.4f}")
print("  Params:")
for key, value in best.params.items():
    print(f"    {key}: {value}")

fig = plot_param_importances(study)
fig.savefig("param_importances_swin.png")
run.stop()
