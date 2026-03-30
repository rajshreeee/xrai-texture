# import os
# import torch
# import timm
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from PIL import Image
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import time
# import matplotlib.pyplot as plt
# from torch.utils.data import ConcatDataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from skmultilearn.model_selection import IterativeStratification
# from torchvision import models
# from collections import Counter


# # ========= Config =========
# train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masked_images"
# test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masked_images"
# csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
# batch_size = 32
# num_epochs = 30
# learning_rate = 1e-3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class CBISDDSMDataset(Dataset):
#     def __init__(self, image_dir, csv_file, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform

#         self.data = pd.read_csv(csv_file)
#         print(f"[INFO] CSV loaded: {csv_file}, Total rows: {len(self.data)}")

#         # Treat 'benign_without_callback' as 'benign'
#         self.data['pathology'] = self.data['pathology'].str.lower().replace('benign_without_callback', 'benign')

#         # Filter valid entries
#         valid_pathologies = {"malignant", "benign"}
#         valid_shapes = {"round", "oval", "lobulated", "irregular"}
#         valid_birads = {2, 3, 4, 5, 6}

#         self.data = self.data[
#             (self.data['pathology'].isin(valid_pathologies)) &
#             (self.data['mass shape'].str.lower().isin(valid_shapes)) &
#             (self.data['assessment'].isin(valid_birads))
#         ].reset_index(drop=True)

#         print(f"[INFO] Filtered rows: {len(self.data)}")

#         self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         print(f"[INFO] Found {len(self.image_files)} images.")

#         # Define mappings
#         self.pathology_classes = {"benign": 0, "malignant": 1}
#         self.shape_classes = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
#         self.birads_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_name = self.image_files[idx]
#         image_path = os.path.join(self.image_dir, image_name)

#         base_name = os.path.splitext(image_name)[0].split('#')[0]

#         image = Image.open(image_path).convert("RGB")
#         matching_rows = self.data[self.data['image_name'] == base_name]

#         if matching_rows.empty:
#             return self.__getitem__((idx + 1) % len(self.image_files))

#         row = matching_rows.iloc[0]

#         pathology = self.pathology_classes.get(row['pathology'], 1)
#         shape = self.shape_classes.get(str(row['mass shape']).lower(), 3)

#         birads_raw = int(row['assessment'])
#         birads = self.birads_mapping.get(birads_raw, None)
#         if birads is None:
#             return self.__getitem__((idx + 1) % len(self.image_files))

#         if self.transform:
#             image = self.transform(image)
#             # augmented = self.transform(image=np.array(image))
#             # image = augmented["image"]

#         labels = torch.tensor([pathology, shape, birads], dtype=torch.long)

#         return image, labels

#     def get_all_labels(self):
#         labels = []
#         for _, row in self.data.iterrows():
#             pathology = self.pathology_classes.get(row["pathology"], 1)
#             mass_shape = self.shape_classes.get(str(row["mass shape"]).lower(), 3)
#             birads_raw = int(row["assessment"])
#             birads = self.birads_mapping.get(birads_raw, 4)  # Map BIRADS correctly
#             labels.append(torch.tensor([pathology, mass_shape, birads]))
#         return labels

# # Helper function to load backbone without final FC
# # === Load Individual Frozen ResNet Backbones ===
# def load_resnet_model(model_name):
#     model = getattr(models, model_name)(weights="IMAGENET1K_V1")
#     modules = list(model.children())[:-2]  # Keep up to the last conv layer
#     model = nn.Sequential(*modules)

#     # Unfreeze last few layers
#     for name, param in model.named_parameters():
#         if "layer4" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#     return model

# # Define the Stacked Ensemble
# # === Stacked Ensemble Model ===
# # class StackedEnsemble(nn.Module):
# #     def __init__(self):
# #         super(StackedEnsemble, self).__init__()
# #         self.resnet50 = load_resnet_model("resnet50")
# #         self.resnet101 = load_resnet_model("resnet101")
# #         self.resnet152 = load_resnet_model("resnet152")

# #         # Adaptive pooling to reduce 3D feature maps to 1D vectors
# #         self.pool = nn.AdaptiveAvgPool2d((1, 1))

# #         # Fully Connected Layer: (Pathology: 1, BI-RADS: 4, Shape: 4)
# #         #self.fc = nn.Linear(3 * 2048, 1 + 4 + 4) 

# #         self.fc = nn.Sequential(
# #             nn.Linear(3 * 2048, 2048),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(2048, 512),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(512, 1 + 4 + 4)
# #         )

# #     def forward(self, x):
# #         x1 = self.pool(self.resnet50(x)).flatten(start_dim=1)
# #         x2 = self.pool(self.resnet101(x)).flatten(start_dim=1)
# #         x3 = self.pool(self.resnet152(x)).flatten(start_dim=1)

# #         combined_features = torch.cat((x1, x2, x3), dim=1)
# #         output = self.fc(combined_features)

# #         out_pathology = output[:, :1]     # 1 neuron
# #         out_birads = output[:, 1:5]        # 4 neurons
# #         out_shape = output[:, 5:]          # 4 neurons

# #         return out_pathology, out_birads, out_shape


# class ResidualAdapter(nn.Module):
#     def __init__(self, input_dim, hidden_dim=1024):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         return x + self.linear2(self.relu(self.linear1(x)))

# class StackedEnsemble(nn.Module):
#     def __init__(self):
#         super(StackedEnsemble, self).__init__()
#         self.resnet50 = load_resnet_model("resnet50")
#         self.resnet101 = load_resnet_model("resnet101")
#         self.resnet152 = load_resnet_model("resnet152")

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.shared_dim = 3 * 2048

#         self.adapter_output_dim = 1024

#         self.fc_pathology = nn.Sequential(
#             nn.Linear(self.shared_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1)
#         )

#         self.fc_shape = nn.Sequential(
#             nn.Linear(self.shared_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 4)
#         )

#         self.fc_birads = nn.Sequential(
#             nn.Linear(self.shared_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 4)
#         )

#         # Adapter layers for each task
#         # self.adapter_path = nn.Sequential(
#         #     nn.Linear(self.shared_dim, 1024),
#         #     nn.ReLU()
#         # )
#         # self.adapter_shape = nn.Sequential(
#         #     nn.Linear(self.shared_dim, 1024),
#         #     nn.ReLU()
#         # )
#         # self.adapter_birads = nn.Sequential(
#         #     nn.Linear(self.shared_dim, 1024),
#         #     nn.ReLU()
#         # )

#         # self.adapter_path = ResidualAdapter(self.shared_dim, hidden_dim = self.adapter_output_dim)
#         # self.adapter_shape = ResidualAdapter(self.shared_dim, hidden_dim = self.adapter_output_dim)
#         # self.adapter_birads = ResidualAdapter(self.shared_dim, hidden_dim = self.adapter_output_dim)

#     def forward(self, x):
#         x1 = self.pool(self.resnet50(x)).flatten(start_dim=1)
#         x2 = self.pool(self.resnet101(x)).flatten(start_dim=1)
#         x3 = self.pool(self.resnet152(x)).flatten(start_dim=1)

#         combined_features = torch.cat((x1, x2, x3), dim=1)

#         #OLD
#         # out_pathology = self.fc_pathology(combined_features)
#         # out_shape = self.fc_shape(combined_features)
#         # out_birads = self.fc_birads(combined_features)

#         # adapted_pathology = self.adapter_path(combined_features)
#         # adapted_shape = self.adapter_shape(combined_features)
#         # adapted_birads = self.adapter_birads(combined_features)

#         out_pathology = self.fc_pathology(combined_features)
#         out_shape     = self.fc_shape(combined_features)
#         out_birads    = self.fc_birads(combined_features)

#         return out_pathology, out_birads, out_shape

# # ========= Prepare Data =========
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])

# train_transform = A.Compose([
#     A.RandomRotate90(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#     A.Resize(224, 224),
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])

# # For validation/test, only resize + normalize
# val_test_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])


# train_dataset = CBISDDSMDataset(train_path, csv_file, transform=transform)
# test_dataset_extra = CBISDDSMDataset(test_path, csv_file, transform=transform)
# combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])

# # all_labels = []
# # for i in range(len(combined_dataset)):
# #     _, labels = combined_dataset[i]
# #     all_labels.append(labels)

# # # Convert to NumPy for slicing
# # all_labels_np = torch.stack(all_labels).numpy()

# # # Extract individual components
# # pathology = all_labels_np[:, 0].astype(int)
# # shape = all_labels_np[:, 1].astype(int)
# # birads = all_labels_np[:, 2:].argmax(axis=1)  # One-hot → class index

# # # Create composite label (tuples work for stratification)
# # composite_labels = list(zip(pathology, shape, birads))

# # # ==== 2. Stratified Split (80/10/10) ====
# # indices = np.arange(len(combined_dataset))

# # # 80% train, 20% temp
# # train_idx, temp_idx, _, temp_labels = train_test_split(
# #     indices, composite_labels, stratify=composite_labels, test_size=0.2, random_state=42
# # )

# # # 50% val, 50% test (from temp, i.e., 10% each)
# # val_idx, test_idx = train_test_split(
# #     temp_idx, stratify=[composite_labels[i] for i in temp_idx], test_size=0.5, random_state=42
# # )

# # # ==== 3. Build DataLoaders ====
# # train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
# # val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
# # test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

# # print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")

# # # 1. Prepare multilabel matrix
# # all_labels = []
# # for i in range(len(combined_dataset)):
# #     _, labels = combined_dataset[i]
# #     all_labels.append(labels)

# # # Stack into (n_samples, 3) [pathology, shape, birads]
# # all_labels_np = torch.stack(all_labels).numpy()

# # # 2. Setup stratifier
# # stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# # # 3. Get train/test split
# # train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# # # 4. Further split temp into val and test (50-50)
# # temp_labels_np = all_labels_np[temp_idx]
# # stratifier_val_test = IterativeStratification(n_splits=2, order=1)
# # val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# # # Need to adjust temp_idx mapping
# # val_idx = temp_idx[val_idx]
# # test_idx = temp_idx[test_idx]

# # # 5. Final DataLoaders
# # train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
# # val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
# # test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

# # print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")




# # 1. Prepare multilabel matrix
# all_labels = []
# for i in range(len(combined_dataset)):
#     _, labels = combined_dataset[i]
#     all_labels.append(labels)

# # Stack into (n_samples, 3) [pathology, shape, birads]
# all_labels_np = torch.stack(all_labels).numpy()

# # ========== Print Overall Distribution Before Splitting ==========
# print("[INFO] Overall dataset size:", len(all_labels_np))

# pathology_counts = Counter(all_labels_np[:, 0])
# shape_counts = Counter(all_labels_np[:, 1])
# birads_counts = Counter(all_labels_np[:, 2])

# print("Pathology distribution (0=benign,1=malignant):", dict(pathology_counts))
# print("Shape distribution (0=round,1=oval,2=lobulated,3=irregular):", dict(shape_counts))
# print("BIRADS distribution (0->B2,1->B3,2->B4,3->B5):", dict(birads_counts))

# # 2. Setup stratifier
# stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# # 3. Get train/test split
# train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# # 4. Further split temp into val and test (50-50)
# temp_labels_np = all_labels_np[temp_idx]
# stratifier_val_test = IterativeStratification(n_splits=2, order=1)
# val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# # Need to adjust temp_idx mapping
# val_idx = temp_idx[val_idx]
# test_idx = temp_idx[test_idx]

# # 5. Print distributions after split
# def print_split_distribution(indices, name):
#     labels = all_labels_np[indices]
#     pathology_counts = Counter(labels[:, 0])
#     shape_counts = Counter(labels[:, 1])
#     birads_counts = Counter(labels[:, 2])
#     print(f"\n[INFO] {name} size: {len(indices)}")
#     print("Pathology distribution:", dict(pathology_counts))
#     print("Shape distribution:", dict(shape_counts))
#     print("BIRADS distribution:", dict(birads_counts))

# print_split_distribution(train_idx, "TRAIN")
# print_split_distribution(val_idx,   "VALID")
# print_split_distribution(test_idx,  "TEST")

# # 6. Final DataLoaders
# train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

# print(f"\nFinal Split Sizes --> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")




# def get_class_weights(labels, num_classes):
#     counts = Counter(labels)
#     total = sum(counts.values())
#     weights = [total / counts.get(i, 1) for i in range(num_classes)]
#     weights = torch.tensor(weights, dtype=torch.float)
#     return weights / weights.sum()

# shape_weights = get_class_weights(all_labels_np[:, 1], 4).to(device)
# #birads_weights = get_class_weights(all_labels_np[:, 2], 4).to(device)

# from sklearn.utils.class_weight import compute_class_weight

# birads_labels = [lbl[2] for lbl in all_labels_np]
# class_weights_birads = compute_class_weight('balanced', classes=np.unique(birads_labels), y=birads_labels)
# class_weights_birads = torch.tensor(class_weights_birads, dtype=torch.float).to(device)

# loss_birads = nn.CrossEntropyLoss(weight=class_weights_birads)


# # ==== Model, Losses, Optimizer ====
# model = StackedEnsemble().to(device)

# if torch.cuda.device_count() > 1:
#     print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# loss_pathology = nn.BCEWithLogitsLoss()
# #loss_shape = nn.CrossEntropyLoss(weight=shape_weights)
# loss_shape = nn.CrossEntropyLoss(weight=shape_weights)
# #loss_birads = nn.CrossEntropyLoss(weight=birads_weights)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# early_stopping_patience = 5
# best_val_loss = float('inf')
# epochs_no_improve = 0

# train_losses, val_losses = [], []
# train_accuracies, val_accuracies = [], []
# val_accuracies_pathology, val_accuracies_shape, val_accuracies_birads = [], [], []

# # ==== Training Loop ====
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         pathology_labels = labels[:, 0]
#         shape_labels = labels[:, 1]
#         #birads_one_hot = labels[:, 2:]
#         birads_labels = labels[:, 2:].long()

#         # print("Pathology labels unique:", pathology_labels.unique())
#         # print("Shape labels unique:", shape_labels.unique())
#         # print("BIRADS labels unique:", birads_labels.unique())

#         pathology_targets = labels[:, 0].float().unsqueeze(1)  # float + unsqueeze for BCE
#         shape_targets = labels[:, 1].long()
#         birads_targets = labels[:, 2].long()

#         optimizer.zero_grad()
#         out_pathology, out_birads, out_shape = model(images)

#         loss1 = loss_pathology(out_pathology, pathology_targets)
#         loss2 = loss_birads(out_birads, birads_targets)
#         loss3 = loss_shape(out_shape, shape_targets)
#         total_loss = loss1 + loss2 + loss3

#         total_loss.backward()
#         optimizer.step()
#         running_loss += total_loss.item()

#     avg_train_loss = running_loss / len(train_loader)

#     # ==== Validation ====
#     model.eval()
#     val_loss = 0

#     preds_path, targets_path = [], []
#     preds_shape, targets_shape = [], []
#     preds_birads, targets_birads = [], []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)

#             pathology_targets = labels[:, 0].unsqueeze(1).float()
#             shape_targets = labels[:, 1].long()
#             birads_targets = labels[:, 2].long()

#             out_pathology, out_birads, out_shape = model(images)

#             loss1 = loss_pathology(out_pathology, pathology_targets)
#             loss2 = loss_birads(out_birads, birads_targets)
#             loss3 = loss_shape(out_shape, shape_targets)
#             total_val = loss1 + loss2 + loss3
#             val_loss += total_val.item()

#             # --- Pathology predictions ---
#             probs_pathology = torch.sigmoid(out_pathology)
#             pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
#             preds_path.extend(pred_labels_path.flatten())
#             targets_path.extend(pathology_targets.cpu().numpy().flatten())

#             # --- Shape predictions ---
#             pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
#             preds_shape.extend(pred_labels_shape.flatten())
#             targets_shape.extend(shape_targets.cpu().numpy().flatten())

#             # --- BIRADS predictions ---
#             pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
#             preds_birads.extend(pred_labels_birads.flatten())
#             targets_birads.extend(birads_targets.cpu().numpy().flatten())

#     avg_val_loss = val_loss / len(val_loader)

#     # === Compute all accuracies ===
#     val_acc_path = accuracy_score(targets_path, preds_path)
#     val_acc_shape = accuracy_score(targets_shape, preds_shape)
#     val_acc_birads = accuracy_score(targets_birads, preds_birads)

#     scheduler.step(avg_val_loss)

#     train_losses.append(avg_train_loss)
#     val_losses.append(avg_val_loss)

#     # Save all accuracies separately
#     val_accuracies_pathology.append(val_acc_path)
#     val_accuracies_shape.append(val_acc_shape)
#     val_accuracies_birads.append(val_acc_birads)

#     # --- Print all ---
#     print(f"Epoch {epoch+1}/{num_epochs} | "
#         f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
#         f"Pathology Acc={val_acc_path:.4f} | Shape Acc={val_acc_shape:.4f} | BIRADS Acc={val_acc_birads:.4f}")

#     # === Early Stopping ===
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= early_stopping_patience:
#             print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
#             break

# # ==== Test Evaluation (Pathology, Shape, BIRADS) ====
# model.eval()

# # Separate lists for each task
# preds_path, targets_path = [], []
# preds_shape, targets_shape = [], []
# preds_birads, targets_birads = [], []

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         out_pathology, out_birads, out_shape = model(images)

#         # --- Pathology ---
#         probs_pathology = torch.sigmoid(out_pathology)
#         pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
#         preds_path.extend(pred_labels_path.flatten())
#         targets_path.extend(labels[:, 0].cpu().numpy().flatten())

#         # --- Shape ---
#         pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
#         preds_shape.extend(pred_labels_shape.flatten())
#         targets_shape.extend(labels[:, 1].cpu().numpy().flatten())

#         # --- BIRADS ---
#         pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
#         preds_birads.extend(pred_labels_birads.flatten())
#         targets_birads.extend(labels[:, 2].cpu().numpy())

# # === Print Final Metrics ===

# print("\n========= [INFO] Final Test Results =========")

# print("\n--- Pathology Classification ---")
# print("Accuracy:", accuracy_score(targets_path, preds_path))
# print("Classification Report:\n", classification_report(targets_path, preds_path))
# print("Confusion Matrix:\n", confusion_matrix(targets_path, preds_path))

# print("\n--- Shape Classification ---")
# print("Accuracy:", accuracy_score(targets_shape, preds_shape))
# print("Classification Report:\n", classification_report(targets_shape, preds_shape))
# print("Confusion Matrix:\n", confusion_matrix(targets_shape, preds_shape))

# print("\n--- BIRADS Classification ---")
# print("Accuracy:", accuracy_score(targets_birads, preds_birads))
# print("Classification Report:\n", classification_report(targets_birads, preds_birads))
# print("Confusion Matrix:\n", confusion_matrix(targets_birads, preds_birads))

# # ==== Save Training Curves ====
# os.makedirs("stacked-ensemble-plots", exist_ok=True)
# epochs_range = range(1, len(train_losses) + 1)

# plt.figure(figsize=(18, 8))

# # Loss curve
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, train_losses, label='Train Loss')
# plt.plot(epochs_range, val_losses, label='Val Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()

# # Accuracy curves
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, val_accuracies_pathology, label='Pathology Acc')
# plt.plot(epochs_range, val_accuracies_shape, label='Shape Acc')
# plt.plot(epochs_range, val_accuracies_birads, label='BIRADS Acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Validation Accuracies Curve')
# plt.legend()

# plt.tight_layout()
# plt.savefig("stacked-ensemble-plots/training_curves.png")
# plt.close()

# print("[INFO] Training plots saved to stacked-ensemble-plots")


























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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification
from torchvision import models
from collections import Counter
import torch.nn.functional as F

# ========= Config =========
train_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_aug_masked_images"
test_path = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/L5E5_aug_masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"
batch_size = 32
num_epochs = 200
learning_rate = 1e-2
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
        valid_birads = {2, 3, 4, 5, 6}

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

        birads_raw = int(row['assessment'])
        birads = self.birads_mapping.get(birads_raw, None)
        if birads is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            image = self.transform(image)
            # augmented = self.transform(image=np.array(image))
            # image = augmented["image"]

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
# === Load Individual Frozen ResNet Backbones ===
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

# Define the Stacked Ensemble
# === Stacked Ensemble Model ===
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

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# For validation/test, only resize + normalize
val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

train_dataset = CBISDDSMDataset(train_path, csv_file, transform=transform)
test_dataset_extra = CBISDDSMDataset(test_path, csv_file, transform=transform)
combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])

# all_labels = []
# for i in range(len(combined_dataset)):
#     _, labels = combined_dataset[i]
#     all_labels.append(labels)

# # Convert to NumPy for slicing
# all_labels_np = torch.stack(all_labels).numpy()

# # Extract individual components
# pathology = all_labels_np[:, 0].astype(int)
# shape = all_labels_np[:, 1].astype(int)
# birads = all_labels_np[:, 2:].argmax(axis=1)  # One-hot → class index

# # Create composite label (tuples work for stratification)
# composite_labels = list(zip(pathology, shape, birads))

# # ==== 2. Stratified Split (80/10/10) ====
# indices = np.arange(len(combined_dataset))

# # 80% train, 20% temp
# train_idx, temp_idx, _, temp_labels = train_test_split(
#     indices, composite_labels, stratify=composite_labels, test_size=0.2, random_state=42
# )

# # 50% val, 50% test (from temp, i.e., 10% each)
# val_idx, test_idx = train_test_split(
#     temp_idx, stratify=[composite_labels[i] for i in temp_idx], test_size=0.5, random_state=42
# )

# # ==== 3. Build DataLoaders ====
# train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

# print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")

# # 1. Prepare multilabel matrix
# all_labels = []
# for i in range(len(combined_dataset)):
#     _, labels = combined_dataset[i]
#     all_labels.append(labels)

# # Stack into (n_samples, 3) [pathology, shape, birads]
# all_labels_np = torch.stack(all_labels).numpy()

# # 2. Setup stratifier
# stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# # 3. Get train/test split
# train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# # 4. Further split temp into val and test (50-50)
# temp_labels_np = all_labels_np[temp_idx]
# stratifier_val_test = IterativeStratification(n_splits=2, order=1)
# val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# # Need to adjust temp_idx mapping
# val_idx = temp_idx[val_idx]
# test_idx = temp_idx[test_idx]

# # 5. Final DataLoaders
# train_loader = DataLoader(Subset(combined_dataset, train_idx), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(Subset(combined_dataset, val_idx), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(Subset(combined_dataset, test_idx), batch_size=batch_size, shuffle=False)

# print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")



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

from sklearn.utils.class_weight import compute_class_weight

birads_labels = [lbl[2] for lbl in all_labels_np]
class_weights_birads = compute_class_weight('balanced', classes=np.unique(birads_labels), y=birads_labels)
class_weights_birads = torch.tensor(class_weights_birads, dtype=torch.float).to(device)

# ==== Model, Losses, Optimizer ====
model = StackedEnsemble().to(device)

# if torch.cuda.device_count() > 1:
#     print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# loss_pathology = nn.BCEWithLogitsLoss()
# loss_birads = nn.CrossEntropyLoss()
# loss_shape = nn.CrossEntropyLoss()
loss_pathology = BCEDiceLoss()
loss_birads = FocalLoss(gamma=2.0)
loss_shape = FocalLoss(gamma=2.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
val_accuracies_pathology, val_accuracies_shape, val_accuracies_birads = [], [], []

start_time = time.time()
# ==== Training Loop ====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        pathology_labels = labels[:, 0]
        shape_labels = labels[:, 1]
        #birads_one_hot = labels[:, 2:]
        birads_labels = labels[:, 2:].long()

        # print("Pathology labels unique:", pathology_labels.unique())
        # print("Shape labels unique:", shape_labels.unique())
        # print("BIRADS labels unique:", birads_labels.unique())

        pathology_targets = labels[:, 0].float().unsqueeze(1)  # float + unsqueeze for BCE
        shape_targets = labels[:, 1].long()
        birads_targets = labels[:, 2].long()

        optimizer.zero_grad()
        out_pathology, out_birads, out_shape = model(images)

        loss1 = loss_pathology(out_pathology, pathology_targets)
        loss2 = loss_birads(out_birads, birads_targets)
        loss3 = loss_shape(out_shape, shape_targets)
        total_loss = loss1 + loss2 + loss3

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ==== Validation ====
    model.eval()
    val_loss = 0

    preds_path, targets_path = [], []
    preds_shape, targets_shape = [], []
    preds_birads, targets_birads = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            pathology_targets = labels[:, 0].unsqueeze(1).float()
            shape_targets = labels[:, 1].long()
            birads_targets = labels[:, 2].long()

            out_pathology, out_birads, out_shape = model(images)

            loss1 = loss_pathology(out_pathology, pathology_targets)
            loss2 = loss_birads(out_birads, birads_targets)
            loss3 = loss_shape(out_shape, shape_targets)
            total_val = loss1 + loss2 + loss3
            val_loss += total_val.item()

            # --- Pathology predictions ---
            probs_pathology = torch.sigmoid(out_pathology)
            pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
            preds_path.extend(pred_labels_path.flatten())
            targets_path.extend(pathology_targets.cpu().numpy().flatten())

            # --- Shape predictions ---
            pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
            preds_shape.extend(pred_labels_shape.flatten())
            targets_shape.extend(shape_targets.cpu().numpy().flatten())

            # --- BIRADS predictions ---
            pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
            preds_birads.extend(pred_labels_birads.flatten())
            targets_birads.extend(birads_targets.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)

    # === Compute all accuracies ===
    val_acc_path = accuracy_score(targets_path, preds_path)
    val_acc_shape = accuracy_score(targets_shape, preds_shape)
    val_acc_birads = accuracy_score(targets_birads, preds_birads)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save all accuracies separately
    val_accuracies_pathology.append(val_acc_path)
    val_accuracies_shape.append(val_acc_shape)
    val_accuracies_birads.append(val_acc_birads)

    # --- Print all ---
    print(f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
        f"Pathology Acc={val_acc_path:.4f} | Shape Acc={val_acc_shape:.4f} | BIRADS Acc={val_acc_birads:.4f}")

    # === Early Stopping ===
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     epochs_no_improve = 0
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve >= early_stopping_patience:
    #         print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
    #         break

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60

print(f"Training time: {elapsed_time_minutes:.2f} minutes")

# ==== Test Evaluation (Pathology, Shape, BIRADS) ====
model.eval()

# Separate lists for each task
preds_path, targets_path = [], []
preds_shape, targets_shape = [], []
preds_birads, targets_birads = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out_pathology, out_birads, out_shape = model(images)

        # --- Pathology ---
        probs_pathology = torch.sigmoid(out_pathology)
        pred_labels_path = (probs_pathology > 0.5).long().cpu().numpy()
        preds_path.extend(pred_labels_path.flatten())
        targets_path.extend(labels[:, 0].cpu().numpy().flatten())

        # --- Shape ---
        pred_labels_shape = out_shape.argmax(dim=1).cpu().numpy()
        preds_shape.extend(pred_labels_shape.flatten())
        targets_shape.extend(labels[:, 1].cpu().numpy().flatten())

        # --- BIRADS ---
        pred_labels_birads = out_birads.argmax(dim=1).cpu().numpy()
        preds_birads.extend(pred_labels_birads.flatten())
        targets_birads.extend(labels[:, 2].cpu().numpy())

# === Print Final Metrics ===

print("\n========= [INFO] Final Test Results =========")

print("\n--- Pathology Classification ---")
print("Accuracy:", accuracy_score(targets_path, preds_path))
print("Classification Report:\n", classification_report(targets_path, preds_path))
print("Confusion Matrix:\n", confusion_matrix(targets_path, preds_path))

print("\n--- Shape Classification ---")
print("Accuracy:", accuracy_score(targets_shape, preds_shape))
print("Classification Report:\n", classification_report(targets_shape, preds_shape))
print("Confusion Matrix:\n", confusion_matrix(targets_shape, preds_shape))

print("\n--- BIRADS Classification ---")
print("Accuracy:", accuracy_score(targets_birads, preds_birads))
print("Classification Report:\n", classification_report(targets_birads, preds_birads))
print("Confusion Matrix:\n", confusion_matrix(targets_birads, preds_birads))

# ==== Save Training Curves ====
os.makedirs("stacked-ensemble-plots", exist_ok=True)
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(18, 8))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies_pathology, label='Pathology Acc')
plt.plot(epochs_range, val_accuracies_shape, label='Shape Acc')
plt.plot(epochs_range, val_accuracies_birads, label='BIRADS Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracies Curve')
plt.legend()

plt.tight_layout()
plt.savefig("stacked-ensemble-plots/training_curves.png")
plt.close()

print("[INFO] Training plots saved to stacked-ensemble-plots")
