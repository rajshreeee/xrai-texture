import os
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from utils import calculate_iou, add_to_test_results
import config
from tqdm import tqdm

# for fcbFormer
from timm.models.vision_transformer import _cfg
from functools import partial
from . import pvt_v2
import neptune
import time


# visualization

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import optuna
import neptune.integrations.optuna as npt_utils
from neptune.integrations.optuna import NeptuneCallback
from monai.losses import DiceLoss

import logging
import json


#ViT
from torchvision.models import vit_b_16
import torch.nn.functional as F

#Swin
from torchvision.models.swin_transformer import swin_t

# Enable logging for Optuna
optuna_log_file = "optuna_logs.json"
logging.basicConfig(
    filename="optuna_progress.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Save the study in a JSON file
def save_study_to_json(study, file_path):
    trials_data = {
        "trials": [
            {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
        "best_trial": {
            "params": study.best_params,
            "value": study.best_value,
            "number": study.best_trial.number,
        } if study.best_trial else None,
    }
    with open(file_path, "w") as f:
        json.dump(trials_data, f, indent=4)
# Initialize Neptune
# run = None
run = neptune.init_run(
    project="XRAI-Pipeline/XAI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNzM1ZDY3Ny04ODhjLTQwZDktODQyNC0zMGRhNjZjODgwOTQifQ==",#wrongkeyfromhere",
    name="FcbFormer-Tuning-BCEWithLogitsLoss-Adam-Early-Stopping"
)

# # Hyperparameters
hyperparameters = {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "input_size": 512,
    "num_epochs": 3,
    "optimizer": "Adam",
    "loss_function": "BCEWithLogitsLoss"
}
#run["parameters"] = hyperparameters

def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=512,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        # checkpoint = torch.load("/home/mikolaj/oke_laptop/Programs/breast_cancer/segmentation/imports/FCBFormer/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        # backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l
    

# VIT TRAINSFORMER
# class TB(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Load pre-trained Vision Transformer
#         self.transformer = vit_b_16(weights='DEFAULT')
        
#         # Optional: Add a linear layer to adjust transformer output to match FCBFormer's needs
#         self.feature_extractor = nn.Sequential(
#             self.transformer,
#             nn.Linear(1000, 512)  # Example adjustment
#         )

#     def forward(self, x):
#         # Ensure the input is 4D
#         if x.dim() == 2:  # If input is 2D [height, width]
#             x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
#         elif x.dim() == 3:  # If input is 3D [channels, height, width]
#             x = x.unsqueeze(0)  # Add batch dimension
        
#         # Resize input to 224x224 for Vision Transformer
#         x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # Pass resized input through Vision Transformer
#         features = self.feature_extractor(x_resized)
#         return features


#SWIN TRANSFORMER
# class TB(nn.Module):
#     def __init__(self):
#         super(TB, self).__init__()
#         # Load the pre-trained Swin Transformer model from torchvision
#         self.swin_transformer = swin_t(weights="IMAGENET1K_V1")

#         # Remove the classification head (fully connected layer)
#         self.swin_transformer.head = nn.Identity()

#         # Optional: Adjust feature dimensions if needed
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(768, 512),  # Swin-T has 768 output features
#             nn.ReLU()
# #         )

#     def forward(self, x):
#         # Ensure the input is 4D: [B, C, H, W]
#         if x.dim() == 2:
#             x = x.unsqueeze(0).unsqueeze(0)  # Convert [H, W] -> [1, 1, H, W]
#         elif x.dim() == 3:
#             x = x.unsqueeze(0)  # Convert [C, H, W] -> [1, C, H, W]
        
#         # Ensure input has 3 channels for Swin Transformer
#         if x.shape[1] == 1:
#             x = x.expand(-1, 3, -1, -1)  # Convert grayscale to RGB

#         # Resize to 256x256, Swin Transformer accepts various input sizes
#         x_resized = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

#         # Extract features using Swin Transformer
#         features = self.swin_transformer(x_resized)

#         # Process features for compatibility with FCBFormer
#         features = self.feature_extractor(features)

#         return features


class FCBFormer(nn.Module):
    def __init__(self, size=512):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out# Import your FCBFormer model here

# def train_fcbformer(save_path, data_loader, input_size=512):
#     """
#     Train the FCBFormer model with Optuna for hyperparameter tuning and Neptune for tracking results.
#     """
#     def objective(trial):
#         # Suggest hyperparameters using Optuna

#         #for XAI-15
#         # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
#         # batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
#         # num_epochs = trial.suggest_int("num_epochs", 10, 50, step=10)
#         # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
#         # loss_function_name = trial.suggest_categorical(
#         #     "loss_function", ["BCEWithLogitsLoss", "DiceLoss"]
#         # )

#         # XAI-17
#         # learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True)
#         # batch_size = trial.suggest_categorical("batch_size", [8, 16])
#         # num_epochs = trial.suggest_int("num_epochs", 20, 40, step=10)
#         # loss_function_name = "DiceLoss"
#         # optimizer_name = "Adam"

#         # XAI-18
#         learning_rate = trial.suggest_float("learning_rate", 3e-5, 1e-4, log=True)
#         batch_size = trial.suggest_categorical("batch_size", [4, 8])
#         num_epochs = trial.suggest_int("num_epochs", 30, 50, step=10)
#         loss_function_name = "BCEWithLogitsLoss"
#         optimizer_name = "Adam"

#         # XAI-19 & 20
#         # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
#         # batch_size = trial.suggest_categorical("batch_size", [6, 8, 12])
#         # num_epochs = trial.suggest_int("num_epochs", 20, 40, step=10)
#         # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
#         # loss_function_name = trial.suggest_categorical(
#         #     "loss_function", ["HybridLoss"]
#         # )




#         # Log trial hyperparameters to Neptune
#         run[f"trial/{trial.number}/hyperparameters"] = {
#             "learning_rate": learning_rate,
#             "batch_size": batch_size,
#             "input_size": input_size,
#             "num_epochs": num_epochs,
#             "optimizer": optimizer_name,
#             "loss_function": loss_function_name,
#         }

#         # Initialize the FCBFormer model
#         model = FCBFormer(size=input_size)
#         if torch.cuda.is_available():
#             device_id = [0, 1]
#             num_gpus = torch.cuda.device_count()
#             if num_gpus > 1:
#                 model = nn.DataParallel(model, device_ids=device_id)
#             model = model.cuda()
#         else:
#             model = model.cpu()

#         # Define optimizer and loss function
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == "Adam" else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#         if loss_function_name == "BCEWithLogitsLoss":
#             criterion = nn.BCEWithLogitsLoss()
#         elif loss_function_name == "DiceLoss":
#             criterion = DiceLoss(to_onehot_y=False, softmax=False)
#         elif loss_function_name == "HybridLoss":
#             bce_loss = nn.BCEWithLogitsLoss()
#             dice_loss = DiceLoss(to_onehot_y=False, softmax=False)

#             def hybrid_loss(outputs, masks):
#                 return 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)

#             criterion = hybrid_loss

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
#         # Start training
#         start_time = time.time()
#         best_iou = 0.0
#         best_model_path = None

#         for epoch in range(num_epochs):
#             model.train()
#             running_loss = 0.0
#             iou_scores = []

#             for batch_idx, (images, masks) in enumerate(data_loader):
#                  # Add channel dimension for DiceLoss
#                 if images.shape[0] < 2 or images.shape[2] < 2 or images.shape[3] < 2 or \
#                         masks.shape[0] < 2 or masks.shape[2] < 2 or masks.shape[3] < 2:
#                     continue

#                 if torch.cuda.is_available():
#                     images, masks = images.cuda(), masks.cuda()
                
#                 if loss_function_name in ["DiceLoss", "HybridLoss"]:
#                     masks = masks.unsqueeze(1) 

#                 optimizer.zero_grad()
#                 outputs = model(images)
                

#                 if loss_function_name in ["DiceLoss", "HybridLoss"]:
#                     outputs = outputs.unsqueeze(1)
#                 else: 
#                     outputs = outputs.squeeze(1)
#                 if masks.dim() == 4:
#                     masks = masks.squeeze(1)

#                 loss = criterion(outputs, masks)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 iou_score = calculate_iou(torch.sigmoid(outputs) > 0.5, masks)
#                 iou_scores.append(iou_score)

#                 # Log batch metrics
#                 run[f"trial/{trial.number}/batch_loss"].append(loss.item())
#                 run[f"trial/{trial.number}/batch_iou"].append(iou_score)

#             epoch_loss = running_loss / len(data_loader)
#             epoch_iou = np.mean(iou_scores)

#             # Log epoch metrics
#             run[f"trial/{trial.number}/epoch_loss"].append(epoch_loss)
#             run[f"trial/{trial.number}/epoch_iou"].append(epoch_iou)

#             # Save the best model
#             if epoch_iou > best_iou:
#                 best_iou = epoch_iou
#                 best_model_path = os.path.join(save_path, f"fcbformer_best_trial_{trial.number}.pth")
#                 torch.save(model.state_dict(), best_model_path)
            
#             # Step the scheduler
#             scheduler.step()

#         # Log training time
#         total_training_time = time.time() - start_time
#         run[f"trial/{trial.number}/total_training_time"] = total_training_time / 60.0

#         # Upload the best model
#         if best_model_path:
#             run[f"trial/{trial.number}/best_model"].upload(best_model_path)

#         return best_iou
#     # Set up Optuna study
#     study = optuna.create_study(
#         study_name="FCBFormer_Optuna_Study_With_Hybrid",  # Name of the study
#         direction="maximize",
#         load_if_exists=True  # Load existing study
#     )
#     neptune_callback = NeptuneCallback(run)

#     # Optimize hyperparameters
#     # Optimize hyperparameters
#     try:
#         study.optimize(objective, n_trials=7, callbacks=[neptune_callback])
#     except Exception as e:
#         logger.error(f"Study interrupted: {e}")
#     # Save study results to a JSON file
#     save_study_to_json(study, optuna_log_file)

#     # Log best parameters to Neptune
#     run["optuna/best_params"] = study.best_params
#     run["optuna/best_value"] = study.best_value
#     run["optuna/study_summary"] = str(study)

#     print(f"Best hyperparameters: {study.best_params}")
#     print(f"Best IoU: {study.best_value}")

#     # Stop the Neptune run
#     run.stop()

def train_fcbformer(save_path, data_loader, val_loader, input_size=512, patience=5, 
                   learning_rate=4.7751948374780065e-05, batch_size=8, num_epochs=50):
    """
    Train the FCBFormer model with early stopping and detailed logging.
    Uses best hyperparameters from Optuna trial 6.
    """
    
    # ✅ Print training configuration
    print("\n" + "="*80)
    print("🚀 TRAINING FCBFormer - BASELINE")
    print("="*80)
    print(f"📂 Save path: {save_path}")
    print(f"🔧 Configuration:")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Num epochs: {num_epochs}")
    print(f"   • Input size: {input_size}×{input_size}")
    print(f"   • Early stopping patience: {patience}")
    print(f"   • Optimizer: Adam")
    print(f"   • Loss function: BCEWithLogitsLoss")
    print(f"📊 Data:")
    print(f"   • Train batches: {len(data_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    print(f"   • Train samples: {len(data_loader.dataset)}")
    print(f"   • Val samples: {len(val_loader.dataset)}")
    print("="*80 + "\n")
    
    # Force flush output
    import sys
    sys.stdout.flush()
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize the FCBFormer model
    print("🔨 Initializing model...")
    model = FCBFormer(size=input_size)
    
    if torch.cuda.is_available():
        device_id = [0, 1]
        num_gpus = torch.cuda.device_count()
        print(f"✅ Found {num_gpus} GPU(s)")
        if num_gpus > 1:
            model = nn.DataParallel(model, device_ids=device_id)
            print(f"   Using DataParallel on GPUs: {device_id}")
        model = model.cuda()
    else:
        print("⚠️  No GPU found, using CPU")
        model = model.cpu()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters:")
    print(f"   • Total: {total_params:,}")
    print(f"   • Trainable: {trainable_params:,}")
    print()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Start training
    start_time = time.time()
    best_iou = 0.0
    best_dice = 0.0
    best_model_path = os.path.join(save_path, "fcbformer_best.pth")
    early_stopping_counter = 0
    
    # Log file
    log_file = os.path.join(save_path, "training_log.txt")
    
    print("🏋️  Starting training...\n")
    sys.stdout.flush()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"📅 Epoch [{epoch+1}/{num_epochs}]")
        print('='*80)
        
        # Training phase
        model.train()
        running_loss = 0.0
        iou_scores = []
        dice_scores = []
        
        # Progress bar for training
        train_pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                         desc=f"🔄 Training", 
                         bar_format='{l_bar}{bar:30}{r_bar}')
        
        for batch_idx, (images, masks) in train_pbar:
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            batch_iou = calculate_iou(preds, masks)
            batch_dice = dice_score(preds, masks).item()
            
            iou_scores.append(batch_iou)
            dice_scores.append(batch_dice)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })

        epoch_loss = running_loss / len(data_loader)
        epoch_iou = np.mean(iou_scores)
        epoch_dice = np.mean(dice_scores)

        # Validation phase
        print(f"\n🔍 Validating...")
        model.eval()
        val_loss = 0.0
        val_iou_scores = []
        val_dice_scores = []
        
        val_pbar = tqdm(val_loader, total=len(val_loader),
                       desc=f"   Validation",
                       bar_format='{l_bar}{bar:30}{r_bar}')
        
        with torch.no_grad():
            for val_images, val_masks in val_pbar:
                if torch.cuda.is_available():
                    val_images, val_masks = val_images.cuda(), val_masks.cuda()

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()
                val_preds = torch.sigmoid(val_outputs) > 0.5
                val_iou_scores.append(calculate_iou(val_preds, val_masks))
                val_dice_scores.append(dice_score(val_preds, val_masks).item())

        val_loss /= len(val_loader)
        val_iou = np.mean(val_iou_scores)
        val_dice = np.mean(val_dice_scores)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Training:")
        print(f"      • Loss: {epoch_loss:.4f}")
        print(f"      • IoU:  {epoch_iou:.4f} ({epoch_iou*100:.2f}%)")
        print(f"      • Dice: {epoch_dice:.4f}")
        print(f"   Validation:")
        print(f"      • Loss: {val_loss:.4f}")
        print(f"      • IoU:  {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"      • Dice: {val_dice:.4f}")
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"  Train - Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, Dice: {epoch_dice:.4f}\n")
            f.write(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}\n")
            f.write(f"  Time: {epoch_time:.1f}s\n\n")
        
        # Save best model
        if val_iou > best_iou:
            improvement = val_iou - best_iou
            best_iou = val_iou
            best_dice = val_dice
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou,
                'dice': best_dice,
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'input_size': input_size
                }
            }, best_model_path)
            
            early_stopping_counter = 0
            print(f"\n   ✅ New best model saved!")
            print(f"      • IoU improvement: +{improvement:.4f}")
            print(f"      • Best IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
            print(f"      • Best Dice: {best_dice:.4f}")
        else:
            early_stopping_counter += 1
            print(f"\n   ⏸️  No improvement (patience: {early_stopping_counter}/{patience})")

        # Check early stopping
        if early_stopping_counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"   No improvement for {patience} consecutive epochs")
            break

        # Step scheduler
        scheduler.step()
        print('='*80)
        sys.stdout.flush()

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print('='*80)
    print(f"⏱️  Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"🏆 Best Results:")
    print(f"   • IoU:  {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"   • Dice: {best_dice:.4f}")
    print(f"💾 Model saved to: {best_model_path}")
    print('='*80 + "\n")
    
    # Save final summary
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training Complete\n")
        f.write(f"Total time: {total_time/60:.2f} minutes\n")
        f.write(f"Best IoU: {best_iou:.4f}\n")
        f.write(f"Best Dice: {best_dice:.4f}\n")
        f.write(f"{'='*80}\n")

    return best_model_path

# Function to denormalize images for visualization
def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image * std[:, None, None] + mean[:, None, None]  # Reverse normalization
    return np.clip(image, 0, 1)

def visualize_segmentation(images, masks, predictions, output_folder, index):
    # Define normalization parameters used in the data loader
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(images.shape[0]):  # Iterate through the batch
        img = images[i].cpu().numpy()  # Get the image as a NumPy array
        img = denormalize(img, mean, std).transpose(1, 2, 0)  # Denormalize and convert to HWC format

        mask = masks[i].cpu().numpy().squeeze()  # Remove channel dimension if present
        pred = predictions[i].cpu().numpy().squeeze()  # Remove channel dimension if present

        # Plot the original image, mask, and prediction
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Ground truth mask
        ax[1].imshow(mask, cmap='gray')  # Ground truth
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')

        # Predicted segmentation
        ax[2].imshow(img)
        ax[2].imshow(pred, cmap='jet', alpha=0.5)  # Overlay prediction on the original image
        ax[2].set_title('Predicted Segmentation')
        ax[2].axis('off')

        #plt.tight_layout()
        frame_path = os.path.join(output_folder, f"frame_{index:04d}.png")
        plt.savefig(frame_path)
        plt.close()


def test_fcbformer_old(result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    print("Testing FCBFormer on Feature: " + str(feature_dataset_choice) + " of " + dataset)

    model_path = os.path.join(config.saved_models_path, 'FCBFormer', dataset, f'Feature_{feature_dataset_choice}', 'fcbformer_segmentation.pth')

    print("===================")
    print(model_path)
    print("===================")


    if not os.path.exists(model_path):
        print("The given model does not exist, Train the model before testing.")
        return

    # this code is for the models saved through this codebase and for neptune use the below code.
    #model = torch.load(model_path, map_location=torch.device('cpu'))

    model = FCBFormer(size=input_size)  # Ensure this matches your model initialization logic

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    iou_scores = []

    output_folder = os.path.join(config.results_path, "visualizations", dataset, f"Feature_{feature_dataset_choice}")
    os.makedirs(output_folder, exist_ok=True)

    frame_index = 0

    for images, masks in data_loader:
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)

        predictions = torch.sigmoid(outputs) > 0.5  # Threshold to binary mask
        iou_score = calculate_iou(predictions, masks)
        iou_scores.append(iou_score)

        # Save visualization frames
        visualize_segmentation(images, masks, predictions, output_folder, frame_index)
        frame_index += images.size(0)

    average_iou = np.mean(iou_scores)
    print(f"Average IoU: {average_iou:.4f}")

    # Generate GIF
    gif_path = os.path.join(output_folder, "segmentation_visualization.gif")
    create_gif(output_folder, gif_path)

    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)
    print(f"Testing Successful. GIF saved at {gif_path}")

def create_gif(frame_folder, gif_path):
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")])
    images = [Image.open(frame) for frame in frames]
    imageio.mimsave(gif_path, images, fps=2) 



def test_fcbformer(result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    print("Testing FCBFormer on Feature: " + str(feature_dataset_choice) + " of " + dataset)

    # model_path = os.path.join(config.saved_models_path, 'FCBFormer', dataset, f'Feature_{feature_dataset_choice}', 'fcbformer_segmentation.pth')
   
    # For feature 1, produces an IoU of 71%
    # model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM/Feature_1/fcbformer_best_trial_6.pth"
    
    
    # model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM/Feature_10/fcbformer_best.pth"
    model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM_PATCHES/TestFeature_10/fcbformer_best.pth"

    print("===================")
    print(model_path)
    print("===================")

    checkpoint = torch.load(model_path, map_location="cpu")

    if not os.path.exists(model_path):
        print("The given model does not exist. Train the model before testing.")
        return

    # Initialize the FCBFormer model
    model = FCBFormer(size=input_size)  # Ensure this matches your training model's initialization logic

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Remove "module." prefix if it exists
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Strip the "module." prefix
        else:
            new_state_dict[key] = value

    # Load the model state_dict
    try:
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    iou_scores = []

    # Prepare output folder for visualizations
    output_folder = os.path.join(config.results_path, "visualizations", dataset, f"Feature_{feature_dataset_choice}")
    os.makedirs(output_folder, exist_ok=True)

    frame_index = 0

    for images, masks, file_names in tqdm(data_loader, total=len(data_loader), desc="Testing"):
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)

        predictions = torch.sigmoid(outputs) > 0.5  # Threshold to binary mask
        iou_score = calculate_iou(predictions, masks)
        iou_scores.append(iou_score)

        # Save visualization frames
        visualize_segmentation(images, masks, predictions, output_folder, frame_index)
        frame_index += images.size(0)

    average_iou = np.mean(iou_scores)
    print(f"Average IoU: {average_iou:.4f}")

    # Generate GIF
    gif_path = os.path.join(output_folder, "segmentation_visualization.gif")
    create_gif(output_folder, gif_path)

    # Log results
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)
    print(f"Testing Successful. GIF saved at {gif_path}")