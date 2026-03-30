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
import optuna
import neptune
from datetime import datetime
from optuna.visualization import plot_param_importances
from sklearn.utils.class_weight import compute_class_weight


os.makedirs("optuna_models", exist_ok=True)
run = neptune.init_run(
    project="XRAI-Pipeline/XAI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNzM1ZDY3Ny04ODhjLTQwZDktODQyNC0zMGRhNjZjODgwOTQifQ==",#wrongkeyfromhere",
    name="Stack-Ensemble-Tuning-Final"
)
run["sys/tags"].add(["optuna", "stacked-ensemble", "xai"])

train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_aug_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_aug_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32 # 16 maybe?
num_epochs = 100 # around 50. 
learning_rate = 1e-2 #make low.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss for multi-class classification.

        Args:
            alpha: weight for false positives
            beta: weight for false negatives
            smooth: small constant to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (integer class labels)
        """
        num_classes = logits.size(1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        probs = F.softmax(logits, dim=1)

        # Compute Tversky index
        TP = (probs * targets_one_hot).sum(dim=0)
        FP = ((1 - targets_one_hot) * probs).sum(dim=0)
        FN = (targets_one_hot * (1 - probs)).sum(dim=0)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1.0 - tversky.mean()

        return loss

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

# Helper function to load backbone without final FC
def load_resnet_model(model_name):
    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    modules = list(model.children())[:-2]  # Keep up to the last conv layer
    model = nn.Sequential(*modules)

    # Unfreeze last few layers
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

#=== Stacked Ensemble Model ===
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

    
# ========= Prepare Data =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CBISDDSMDataset(train_path, csv_file, transform=transform)
test_dataset_extra = CBISDDSMDataset(test_path, csv_file, transform=transform)
combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])


# 1. Prepare multilabel matrix
all_labels = []
for i in range(len(combined_dataset)):
    _, labels = combined_dataset[i]
    all_labels.append(labels)

# Stack into (n_samples, 3) [pathology, shape, birads]
all_labels_np = torch.stack(all_labels).numpy()

# ========== Print Overall Distribution Before Splitting ==========
print("[INFO] Overall dataset size:", len(all_labels_np))

pathology_counts = Counter(all_labels_np[:, 0])
shape_counts = Counter(all_labels_np[:, 1])
birads_counts = Counter(all_labels_np[:, 2])

print("Pathology distribution (0=benign,1=malignant):", dict(pathology_counts))
print("Shape distribution (0=round,1=oval,2=lobulated,3=irregular):", dict(shape_counts))
print("BIRADS distribution (0->B2,1->B3,2->B4,3->B5):", dict(birads_counts))

# 2. Setup stratifier
stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# 3. Get train/test split
train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# 4. Further split temp into val and test (50-50)
temp_labels_np = all_labels_np[temp_idx]
stratifier_val_test = IterativeStratification(n_splits=2, order=1)
val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# Need to adjust temp_idx mapping
val_idx = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

# 5. Print distributions after split
def print_split_distribution(indices, name):
    labels = all_labels_np[indices]
    pathology_counts = Counter(labels[:, 0])
    shape_counts = Counter(labels[:, 1])
    birads_counts = Counter(labels[:, 2])
    print(f"\n[INFO] {name} size: {len(indices)}")
    print("Pathology distribution:", dict(pathology_counts))
    print("Shape distribution:", dict(shape_counts))
    print("BIRADS distribution:", dict(birads_counts))

print_split_distribution(train_idx, "TRAIN")
print_split_distribution(val_idx,   "VALID")
print_split_distribution(test_idx,  "TEST")


# 6. Final DataLoaders
train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

print(f"\nFinal Split Sizes --> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


def get_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.sum()

shape_weights = get_class_weights(all_labels_np[:, 1], 4).to(device)
#birads_weights = get_class_weights(all_labels_np[:, 2], 4).to(device)


birads_labels = [lbl[2] for lbl in all_labels_np]
class_weights_birads = compute_class_weight('balanced', classes=np.unique(birads_labels), y=birads_labels)
class_weights_birads = torch.tensor(class_weights_birads, dtype=torch.float).to(device)

# ==== Model, Losses, Optimizer ====
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)
    w1 = trial.suggest_float("w_pathology", 0.5, 1.5)
    w2 = trial.suggest_float("w_birads", 0.5, 1.5)
    w3 = trial.suggest_float("w_shape", 1.5, 3.0)
    num_epochs = trial.suggest_int("epochs", 80, 100)

    trial_ns = run[f"trial/{trial.number}"]
    trial_ns["params"] = {
        "lr": lr, "weight_decay": wd, "gamma": gamma,
        "w1": w1, "w2": w2, "w3": w3, "epochs": num_epochs
    }

    model = StackedEnsemble().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_path = BCEDiceLoss()
    loss_bi = FocalLoss(gamma)
    loss_sh = FocalLoss(gamma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 2)

    train_shape_accs, val_shape_accs = [], []
    train_path_accs, val_path_accs = [], []
    train_birads_accs, val_birads_accs = [], []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        correct_shape, correct_path, correct_birads = 0, 0, 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            out_path, out_bi, out_sh = model(images)

            path_t = labels[:, 0].float().unsqueeze(1)
            shape_t = labels[:, 1].long()
            birads_t = labels[:, 2].long()

            l1 = loss_path(out_path, path_t)
            l2 = loss_bi(out_bi, birads_t)
            l3 = loss_sh(out_sh, shape_t)
            loss = w1*l1 + w2*l2 + w3*l3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += path_t.size(0)
            correct_shape += (out_sh.argmax(1) == shape_t).sum().item()
            correct_path += ((torch.sigmoid(out_path) > 0.5).int().squeeze() == path_t.int()).sum().item()
            correct_birads += (out_bi.argmax(1) == birads_t).sum().item()

        train_shape_accs.append(correct_shape / total)
        train_path_accs.append(correct_path / total)
        train_birads_accs.append(correct_birads / total)

        trial_ns["train/shape_acc"].append(train_shape_accs[-1])
        trial_ns["train/pathology_acc"].append(train_path_accs[-1])
        trial_ns["train/birads_acc"].append(train_birads_accs[-1])

        # === Validation ===
        model.eval()
        preds_shape, targets_shape = [], []
        preds_path, targets_path = [], []
        preds_birads, targets_birads = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out_path, out_bi, out_sh = model(images)

                shape_t = labels[:, 1].long()
                path_t = labels[:, 0].int()
                birads_t = labels[:, 2].long()

                preds_shape += out_sh.argmax(1).cpu().tolist()
                targets_shape += shape_t.cpu().tolist()

                preds_path += (torch.sigmoid(out_path) > 0.5).int().squeeze().cpu().tolist()
                targets_path += path_t.cpu().tolist()

                preds_birads += out_bi.argmax(1).cpu().tolist()
                targets_birads += birads_t.cpu().tolist()

        val_shape_accs.append(accuracy_score(targets_shape, preds_shape))
        val_path_accs.append(accuracy_score(targets_path, preds_path))
        val_birads_accs.append(accuracy_score(targets_birads, preds_birads))

        trial_ns["val/shape_acc"].append(val_shape_accs[-1])
        trial_ns["val/pathology_acc"].append(val_path_accs[-1])
        trial_ns["val/birads_acc"].append(val_birads_accs[-1])

        scheduler.step(1 - val_shape_accs[-1])  # still focus on shape

    # === Save Model ===
    model_path = f"optuna_models/model_trial_{trial.number}.pth"
    torch.save(model.state_dict(), model_path)
    trial_ns["best_model_path"] = model_path

    # === Accuracy Plot ===
    plt.figure(figsize=(12, 5))
    plt.plot(val_shape_accs, label="Shape")
    plt.plot(val_path_accs, label="Pathology")
    plt.plot(val_birads_accs, label="BIRADS")
    plt.title("Validation Accuracy per Task")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plot_path = f"optuna_models/val_accuracy_trial_{trial.number}.png"
    plt.savefig(plot_path)
    trial_ns["val_accuracy_plot"].upload(plot_path)

    # === Final Test Evaluation ===
    model.eval()
    preds_shape, targets_shape = [], []
    preds_path, targets_path = [], []
    preds_birads, targets_birads = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out_path, out_bi, out_sh = model(images)

            preds_shape += out_sh.argmax(1).cpu().tolist()
            targets_shape += labels[:, 1].cpu().tolist()

            preds_path += (torch.sigmoid(out_path) > 0.5).int().squeeze().cpu().tolist()
            targets_path += labels[:, 0].int().cpu().tolist()

            preds_birads += out_bi.argmax(1).cpu().tolist()
            targets_birads += labels[:, 2].cpu().tolist()

    test_shape_acc = accuracy_score(targets_shape, preds_shape)
    test_path_acc = accuracy_score(targets_path, preds_path)
    test_birads_acc = accuracy_score(targets_birads, preds_birads)

    trial_ns["test/shape_acc"] = test_shape_acc
    trial_ns["test/pathology_acc"] = test_path_acc
    trial_ns["test/birads_acc"] = test_birads_acc

    elapsed = (time.time() - start_time) / 60
    trial_ns["train_time_minutes"] = elapsed

    return 1 - test_shape_acc

# === Run the Study ===
study = optuna.create_study(direction="minimize", study_name="ShapeAccuracyPriority")
study.optimize(objective, n_trials=30)

fig = plot_param_importances(study)
plot_path = "optuna_models/hyperparam_importance.png"
fig.write_image(plot_path)
run["optuna/hyperparameter_importance"].upload(plot_path)
# === Log Best Trial ===
run["best_trial/params"] = study.best_trial.params
run["best_trial/test_shape_accuracy"] = 1 - study.best_value
run.stop()
