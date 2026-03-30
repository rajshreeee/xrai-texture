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
import optuna
import neptune.new as neptune
import cv2


run = neptune.init_run(
    project="XRAI-Pipeline/XAI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNzM1ZDY3Ny04ODhjLTQwZDktODQyNC0zMGRhNjZjODgwOTQifQ==",#wrongkeyfromhere",
    name="PathologyClassifier-resnetv2_152x2_bit-Paper-Params-L5E5"
)

print("[INFO] Libraries imported successfully.")
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
        
        # Compute base names (remove extension and anything after '#')
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
        base_name = os.path.splitext(image_name)[0].split('#')[0]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        matching_rows = self.data[self.data["image_name"] == base_name]
        if matching_rows.empty:
            return self.__getitem__((idx + 1) % len(self.image_files))
        row = matching_rows.iloc[0]
        pathology = self.pathology_classes.get(str(row["pathology"]).lower(), 1)
        return image, pathology

    def get_all_labels(self):
        return [self.pathology_classes.get(str(row["pathology"]).lower(), 1)
                for _, row in self.data.iterrows()]



class BIRADSClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", use_radimagenet=False, radimagenet_weights_path=None, num_classes=1):
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
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        for p in m.parameters():
                            p.requires_grad = True
                frozen_blocks += 1

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]
            print(f"[INFO] Feature dimension from backbone: {feature_dim}")

        # Classifier head: FC(1024) → ReLU → Dropout → FC(1)
        # downsample here. upsampling is wrong.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # This will be updated with hyperparameter value
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ---------- Optuna Objective Function ----------
def objective(trial):
    trial_namespace = f"trials/trial_{trial.number}" 
    trial_train_accuracies = []
    trial_val_accuracies = []
    epoch_times = []
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_categorical("epochs", [20, 30, 50])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.3])
    #lr = trial.suggest_loguniform("lr", 1e-3, 1e-4)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    smoothing = trial.suggest_categorical("smoothing", [0.0, 0.2, 0.25])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    
    # Prepare dataset splits (using the original train_set loaded earlier)
    dataset = PathologyDataset(train_path, csv_file, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = BIRADSClassifier(backbone_name="resnet34", use_radimagenet=False,
                             radimagenet_weights_path=radimagenet_weights_path, num_classes=1)
    # Update dropout layer with suggested value:
    model.classifier[2] = nn.Dropout(dropout)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Setup optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Compute positive weight from training labels (from original train_set)
    train_labels = dataset.get_all_labels()
    num_neg = sum(1 for label in train_labels if label == 0)
    num_pos = sum(1 for label in train_labels if label == 1)
    pos_weight_value = num_neg / (num_pos + 1e-6)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    trial_train_losses = []
    trial_val_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):

        start_epoch = time.time()
        model.train()
        total_train_loss = 0
        train_preds, train_labels = [], []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            original_labels = labels.clone()  # Save original labels (before smoothing!)

            if smoothing > 0:
                labels = torch.where(labels == 1, 1 - smoothing, torch.tensor(smoothing, device=device))

            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # For accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().cpu().numpy().flatten()

            true = original_labels.cpu().numpy().flatten().astype(int)  # <--- this line ensures 0/1 integers
            train_preds.extend(preds)
            train_labels.extend(true)
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        trial_train_losses.append(avg_train_loss)
        trial_train_accuracies.append(train_acc)
        
        # Validation step
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                original_labels = labels.clone()

                if smoothing > 0:
                    labels = torch.where(labels == 1, 1 - smoothing, torch.tensor(smoothing, device=device))

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().cpu().numpy().flatten()
                
                true = original_labels.cpu().numpy().flatten().astype(int)
                val_preds.extend(preds)
                val_labels.extend(true)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        trial_val_losses.append(avg_val_loss)
        trial_val_accuracies.append(val_acc)

        epoch_times.append(time.time() - start_epoch)
        
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                break  # early stopping
    
    elapsed_time = time.time() - start_time
    
    # Evaluate final accuracy on the validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().cpu().numpy().flatten()
            true = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy is {accuracy}")

    run[f"{trial_namespace}/epoch_metrics/train_loss"].append(avg_train_loss)
    run[f"{trial_namespace}/epoch_metrics/val_loss"].append(avg_val_loss)
    run[f"{trial_namespace}/epoch_metrics/train_acc"].append(train_acc)
    run[f"{trial_namespace}/epoch_metrics/val_acc"].append(val_acc)
    run[f"{trial_namespace}/epoch_metrics/epoch_time_minutes"].append((time.time() - start_epoch)/60)
    
    # Log results to Neptune
    trial_namespace = f"trials/trial_{trial.number}"
    run[f"{trial_namespace}/hyperparameters"] = {
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": dropout,
        "lr": lr,
        "smoothing": smoothing
    }
    run[f"{trial_namespace}/metrics"] = {
        "avg_train_loss": trial_train_losses[-1],
        "avg_val_loss": trial_val_losses[-1],
        "training_time_minutes": elapsed_time / 60,
        "accuracy": accuracy
    }

    run[f"{trial_namespace}/metrics/final"] = {
        "final_avg_train_loss": trial_train_losses[-1],
        "final_avg_val_loss": trial_val_losses[-1],
        "final_train_accuracy": trial_train_accuracies[-1],
        "final_val_accuracy": trial_val_accuracies[-1],
        "training_time_minutes": elapsed_time / 60,
        "best_val_loss": best_val_loss,
    }
    
    # Optionally, save the loss curves plot for this trial
    plt.figure()
    epochs_range = range(1, len(trial_train_losses)+1)
    plt.plot(epochs_range, trial_train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs_range, trial_val_losses, 'ro-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Trial Loss Curve')
    trial_loss_plot = f"loss_curve_trial_{trial.number}.png"
    plt.savefig(trial_loss_plot)
    run[f"{trial_namespace}/artifacts/loss_curve"].upload(trial_loss_plot)
    
    return best_val_loss


train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
radimagenet_weights_path = "/ediss_data/ediss2/xai-texture/src/models/classification/ResNet50.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

study = optuna.create_study(study_name="PathologyClassifier-resnetv2_152x2_bit-Paper-Params", direction="minimize")
study.optimize(objective, n_trials=20)

# Log study results to Neptune
run["optuna_study/trials"] = study.trials_dataframe()
run.stop()

print("Study completed!")
