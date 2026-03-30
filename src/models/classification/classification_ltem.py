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
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjVmZGJkZi1mODFiLTRlZTgtYWY0MC1hYTY1YjE0ODI4ZjYifQ==",#wrongkeyfromhere",
    name="PathologyClassifier-ResNet50-Paper-Params-L5E5"
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
    
    def apply_L5E5(self, image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Define the convolution kernels
        L5 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)  # L5 mask Level
        E5 = np.array([-1, -2, 0, 2, 1]).reshape(1, 5)  # E5 mask Edge
        L5E5 = cv2.filter2D(image, -1,  np.outer(L5, E5))
        E5L5 = cv2.filter2D(image, -1,  np.outer(E5, L5))
        ltem_feature = (L5E5 + E5L5) / 2
        # Normalize feature_1 to the range [0, 255] for image conversion
        normalized_ltem_feature = cv2.normalize(ltem_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Convert the normalized grayscale image to RGB
        ltem_feature_rgb = cv2.cvtColor(normalized_ltem_feature, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(ltem_feature_rgb, 'RGB')
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        base_name = os.path.splitext(image_name)[0].split('#')[0]
        
        image = self.apply_L5E5(image_path)
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
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_categorical("epochs", [20, 30, 50])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.3])
    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
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
    model = BIRADSClassifier(backbone_name="resnetv2_152x2_bit", use_radimagenet=False,
                             radimagenet_weights_path=radimagenet_weights_path, num_classes=1)
    # Update dropout layer with suggested value:
    model.classifier[2] = nn.Dropout(dropout)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Setup optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            # Apply label smoothing if needed
            if smoothing > 0:
                labels = torch.where(labels == 1, 1 - smoothing, torch.tensor(smoothing, device=device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        trial_train_losses.append(avg_train_loss)
        
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                if smoothing > 0:
                    labels = torch.where(labels == 1, 1 - smoothing, torch.tensor(smoothing, device=device))
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        trial_val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
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


train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
radimagenet_weights_path = "/ediss_data/ediss2/xai-texture/src/models/classification/ResNet50.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

study = optuna.create_study(study_name="PathologyClassifier-ResNet50-Paper-Params", direction="minimize")
study.optimize(objective, n_trials=20)

# Log study results to Neptune
run["optuna_study/trials"] = study.trials_dataframe()
run.stop()

print("Study completed!")
