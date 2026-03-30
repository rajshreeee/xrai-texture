import os
import cv2
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils import calculate_iou, add_to_test_results
import config
import pandas as pd
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
import sys

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

# ============================================================================
# Reusable Logger Class - Can be used for training, testing, etc.
# ============================================================================
class DualLogger:
    """
    Logger that writes to both console and file simultaneously.
    Usage:
        logger = DualLogger(log_path)
        logger.log("Your message")
        logger.close()
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Open log file with line buffering
        self.log_file = open(log_path, "w", buffering=1)
        
        # Redirect stdout
        sys.stdout = self
        
        self.log(f"📝 Log file: {log_path}")
        self.log(f"🕐 Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        """Flush both streams"""
        self.terminal.flush()
        self.log_file.flush()
    
    def log(self, message):
        """Convenience method to print a message"""
        print(message)
    
    def close(self):
        """Restore stdout and close log file"""
        self.log(f"\n🕐 Ended: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"📝 Log saved to: {self.log_path}\n")
        self.flush()
        sys.stdout = self.terminal
        self.log_file.close()
        print(f"✅ Log saved: {self.log_path}")


def get_log_path(base_path, prefix="training"):
    """
    Generate timestamped log file path.
    
    Args:
        base_path: Base directory for logs
        prefix: Prefix for log filename (e.g., 'training', 'testing')
    
    Returns:
        Full path to log file
    """
    from datetime import datetime
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{timestamp}.log")


# ============================================================================
# CHANGE 1: Reduced dropout from 0.3 to 0.2 for moderate regularization
# CHANGE 2: Removed BatchNorm (conflicts with GroupNorm, causes over-regularization)
# ============================================================================
class RB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):  # CHANGED: 0.3 → 0.2
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout_prob)  # CHANGED: Dropout2d instead of Dropout, removed BatchNorm
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout_prob)  # CHANGED: Dropout2d instead of Dropout, removed BatchNorm
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


# ============================================================================
# CHANGE 3: Restored FCB to match paper specifications
# - Changed n_levels_down from 5 → 6
# - Changed n_levels_up from 5 → 6  
# - Changed min_channel_mults from [1,1,2,2,3] → [1,1,2,2,4,4]
# - Changed n_RBs from 1 → 2
# - Removed excessive BatchNorm layers
# - Reduced dropout to 0.2
# ============================================================================
class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],  # CHANGED: Restored to paper config
        n_levels_down=6,  # CHANGED: 5 → 6
        n_levels_up=6,    # CHANGED: 5 → 6
        n_RBs=2,          # CHANGED: 1 → 2
        in_resolution=512,
        dropout_prob=0.2  # CHANGED: 0.3 → 0.2
    ):
        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]  # CHANGED: Removed BatchNorm/Dropout
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels, dropout_prob))  # CHANGED: Removed extra BatchNorm/Dropout
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))  # CHANGED: Removed BatchNorm/Dropout
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch, dropout_prob), RB(ch, ch, dropout_prob))  # CHANGED: Removed extra BatchNorm/Dropout

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                        dropout_prob
                    )  # CHANGED: Removed extra BatchNorm/Dropout
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )  # CHANGED: Removed BatchNorm/Dropout
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


# ============================================================================
# CHANGE 4: Modified TB to output [64, 88, 88] as per paper
# - Changed Upsample from scale_factor=2 → size=88 (direct upsampling)
# - Removed excessive BatchNorm layers
# - Reduced dropout to 0.2
# ============================================================================
class TB(nn.Module):
    def __init__(self, dropout_prob=0.2):  # CHANGED: 0.3 → 0.2
        super().__init__()
        # Load Vision Transformer (ViT-B/16) as the encoder
        self.backbone = vit_b_16(weights="DEFAULT")
        
        # Remove the classification head (VIT)
        self.backbone.heads.head = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Progressive Locality Decoder (PLD+)
        # CHANGED: Upsample to size=88 instead of scale_factor=2, removed BatchNorm
        self.LE = nn.ModuleList([
            nn.Sequential(
                RB(768, 64, dropout_prob),  # Residual Block (RB)
                RB(64, 64, dropout_prob),
                nn.Upsample(size=88, mode='bilinear', align_corners=False)  # CHANGED: Direct upsample to 88
            ) for _ in range(4)
        ])

        # CHANGED: Removed BatchNorm from SFA
        self.SFA = nn.ModuleList([
            nn.Sequential(
                RB(128, 64, dropout_prob),  # Residual Block (RB)
                RB(64, 64, dropout_prob)
            ) for _ in range(3)
        ])

    def get_pyramid(self, x):
        """
        Extract feature pyramid from Vision Transformer (ViT) encoder.
        ViT-B/16 expects 224×224 input, so we resize first.
        """
        # FIXED: Resize to 224×224 for ViT (it's pretrained on this size)
        B, C, H, W = x.shape
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Forward pass through ViT
        x = self.backbone(x_resized)  # Output shape: [B, 768]

        # Reshape to [B, 768, 1, 1] (since ViT outputs a 1D feature vector)
        B, C = x.shape
        x = x.reshape(B, C, 1, 1)

        # Upsample to [B, 768, 14, 14] to match patch grid dimensions
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)

        # Repeat the same feature map for all levels (since ViT is non-hierarchical)
        pyramid = [x] * 4
        return pyramid

    def forward(self, x):
        """
        Forward pass for the Transformer Branch (TB) with Vision Transformer (ViT).
        Output: [B, 64, 88, 88] as per paper specifications
        """
        # Ensure the input is 4D: [batch_size, channels, height, width]
        if x.dim() == 2:  # If input is 2D [height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif x.dim() == 3:  # If input is 3D [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension

        # Extract feature pyramid from ViT encoder
        pyramid = self.get_pyramid(x)

        # Apply Local Emphasis (LE) modules to each level of the pyramid
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        # Stepwise Feature Aggregation (SFA)
        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l_i  # Output: [B, 64, 88, 88]


# ============================================================================
# CHANGE 5: Updated FCBFormer with reduced dropout
# Dimensions now match paper:
# - TB: [64, 88, 88] → upsampled to [64, 512, 512]
# - FCB: [32, 512, 512]
# - Concat: [96, 512, 512]
# - Output: [1, 512, 512]
# ============================================================================
class FCBFormer(nn.Module):
    def __init__(self, size=512, dropout_prob=0.2):  # CHANGED: 0.3 → 0.2
        super().__init__()
        self.TB = TB(dropout_prob=dropout_prob)  # Transformer Branch (ViT)
        self.FCB = FCB(in_resolution=size, dropout_prob=dropout_prob)  # Fully Convolutional Branch
        self.PH = nn.Sequential(
            RB(64 + 32, 64, dropout_prob),  # Residual Block: [96, 512, 512] → [64, 512, 512]
            RB(64, 64, dropout_prob),
            nn.Conv2d(64, 1, kernel_size=1)  # Final prediction layer: [64, 512, 512] → [1, 512, 512]
        )
        self.up_tosize = nn.Upsample(size=size)  # Upsample to full resolution

    def forward(self, x):
        # try:
            # CHANGED: Added dimension verification for debugging
        x1 = self.TB(x)  # Transformer Branch output: [B, 64, 88, 88]
        x2 = self.FCB(x)  # Fully Convolutional Branch output: [B, 32, 512, 512]
            
        x1 = self.up_tosize(x1)  # Upsample TB output: [B, 64, 88, 88] → [B, 64, 512, 512]
            
        if x2.shape[2:] != (512, 512):
            x2 = F.interpolate(x2, size=(512, 512), mode='bilinear', align_corners=False)
            
        x = torch.cat((x1, x2), dim=1)  # Concatenate: [B, 96, 512, 512]
        out = self.PH(x)  # Prediction head: [B, 1, 512, 512]
        return out
        # except Exception as e:
        #     raise Exception('An exception occurred during forward pass') from e


import time
import os

def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def load_fcbformer(model_path: str) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Instantiate the model architecture
    model = FCBFormer(size=512)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove 'module.' prefix if model was trained with DataParallel
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict)
        print("✅ Model loaded successfully.")
    except RuntimeError as e:
        print("❌ Error loading state_dict:", e)
        raise e

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model.eval()


# ============================================================================
# CHANGE 6: Added weight_decay to optimizer for L2 regularization
# ============================================================================
def train_fcbformer(save_path, data_loader, val_loader, input_size=512, patience=5,
                   learning_rate=1e-4, batch_size=8, num_epochs=50):
    """
    Train the FCBFormer model with early stopping and file logging.
    Logs are saved to: save_path/logs/training_YYYYMMDD_HHMMSS.log
    """
    # Setup logging
    log_path = get_log_path(save_path, prefix="training")
    logger = DualLogger(log_path)
    
    try:
        logger.log(f"{'='*80}")
        logger.log(f"🚀 Training FCBFormer - Paper Configuration")
        logger.log(f"{'='*80}")
        logger.log(f"💾 Model save path: {save_path}")
        
        # Initialize the FCBFormer model
        model = FCBFormer(size=input_size)
        if torch.cuda.is_available():
            device_id = [0, 1]
            num_gpus = torch.cuda.device_count()
            logger.log(f"🖥️  Using {num_gpus} GPU(s): {device_id}")
            if num_gpus > 1:
                model = nn.DataParallel(model, device_ids=device_id)
            model = model.cuda()
        else:
            logger.log("⚠️  Using CPU (no GPU available)")
            model = model.cpu()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        start_time = time.time()
        best_iou = 0.0
        best_model_path = os.path.join(save_path, "fcbformer_best.pth")
        early_stopping_counter = 0
        best_dice = 0.0

        logger.log(f"\nArchitecture:")
        logger.log(f"  TB:  ViT-B/16 → [64, 88, 88] → [64, 512, 512]")
        logger.log(f"  FCB: UNet-6 → [32, 512, 512]")
        logger.log(f"  Final: [96, 512, 512] → [1, 512, 512]")
        logger.log(f"\nTraining config:")
        logger.log(f"  LR: {learning_rate}, Batch: {batch_size}, Epochs: {num_epochs}")
        logger.log(f"  Dropout: 0.2, Weight decay: 1e-4, Patience: {patience}")
        logger.log(f"  Train batches: {len(data_loader)}, Val batches: {len(val_loader)}")
        logger.log(f"{'='*80}\n")

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            iou_scores = []
            dice_scores = []

            for batch_idx, (images, masks) in tqdm(enumerate(data_loader), 
                                                    desc=f"Training: Epoch {epoch+1}/{num_epochs}", 
                                                    total=len(data_loader)):
                if torch.cuda.is_available():
                    images, masks = images.cuda(), masks.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                iou_scores.append(calculate_iou(preds, masks))
                dice_scores.append(dice_score(preds, masks).item())

            epoch_loss = running_loss / len(data_loader)
            epoch_iou = np.mean(iou_scores)
            epoch_dice = np.mean(dice_scores)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_iou_scores = []
            val_dice_scores = []
            with torch.no_grad():
                for val_images, val_masks in tqdm(val_loader, 
                                                  desc=f"Validation: Epoch {epoch+1}/{num_epochs}", 
                                                  total=len(val_loader)):
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

            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            logger.log(f"\n[{current_time}] Epoch {epoch+1}/{num_epochs}:")
            logger.log(f"  Train - Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f} ({epoch_iou*100:.2f}%), Dice: {epoch_dice:.4f}")
            logger.log(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f} ({val_iou*100:.2f}%), Dice: {val_dice:.4f}")
                    
            # Save best model
            if val_iou > best_iou:
                improvement = val_iou - best_iou
                best_iou = val_iou
                best_dice = val_dice
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
                        'dropout': 0.2,
                        'weight_decay': 1e-4
                    }
                }, best_model_path)
                early_stopping_counter = 0
                logger.log(f"  ✅ New best model saved! IoU: {best_iou:.4f} (+{improvement:.4f})")
            else:
                early_stopping_counter += 1
                logger.log(f"  ⏸️  No improvement (patience: {early_stopping_counter}/{patience})")

            if early_stopping_counter >= patience:
                logger.log(f"\n🛑 Early stopping at epoch {epoch + 1}")
                break

            scheduler.step()

        total_training_time = time.time() - start_time
        logger.log(f"\n{'='*80}")
        logger.log(f"✅ Training Completed!")
        logger.log(f"{'='*80}")
        logger.log(f"Duration: {total_training_time / 60:.2f} minutes ({total_training_time/3600:.2f} hours)")
        logger.log(f"🏆 Best IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
        logger.log(f"🏆 Best Dice: {best_dice:.4f}")
        logger.log(f"💾 Model: {best_model_path}")
        logger.log(f"{'='*80}")

        return best_model_path
        
    except Exception as e:
        logger.log(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        logger.close()


# Function to denormalize images for visualization
def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image * std[:, None, None] + mean[:, None, None]
    return np.clip(image, 0, 1)


def visualize_segmentation(images, masks, predictions, output_folder, index):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(images.shape[0]):
        img = images[i].cpu().numpy()
        img = denormalize(img, mean, std).transpose(1, 2, 0)

        mask = masks[i].cpu().numpy().squeeze()
        pred = predictions[i].cpu().numpy().squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')

        ax[2].imshow(img)
        ax[2].imshow(pred, cmap='jet', alpha=0.5)
        ax[2].set_title('Predicted Segmentation')
        ax[2].axis('off')

        frame_path = os.path.join(output_folder, f"frame_{index:04d}.png")
        plt.savefig(frame_path)
        plt.close()


def create_gif(frame_folder, gif_path):
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")])
    images = [Image.open(frame) for frame in frames]
    imageio.mimsave(gif_path, images, fps=2)


def save_fcbformer_layers(model, file_path):
    """Recursively saves all layers and sub-layers of the FCBFormer model to a text file."""
    def write_layer(f, module, indent=0):
        for name, layer in module.named_children():
            f.write("  " * indent + f"├── {name}: {layer.__class__.__name__}\n")
            write_layer(f, layer, indent + 1)

    with open(file_path, "w") as f:
        f.write("Layers in FCBFormer during testing (including sub-layers):\n\n")
        for name, layer in model.named_children():
            f.write(f"Layer Name: {name}, Type: {layer.__class__.__name__}\n")
            write_layer(f, layer, indent=1)


from torchvision.transforms import ToPILImage

def test_fcbformer(result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CUSTOM/kernel0_kr_11/fcbformer_best.pth"
    test_fcbformer(model_path, result_path, dataset, feature_dataset_choice, data_loader, input_size)

def test_fcbformer(model_path, result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    print("=" * 20)
    print(model_path)
    print("=" * 20)

    if not os.path.exists(model_path):
        print("The given model does not exist. Train the model before testing.")
        return

    model = load_fcbformer(model_path)
    model.eval()
    iou_scores = []

    output_folder = os.path.join(config.results_path, "visualizations", dataset, f"Feature_{feature_dataset_choice}")
    predictions_folder = os.path.join(output_folder, "predictions")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)

    frame_index = 0
    to_pil = ToPILImage()

    for images, masks, file_names in tqdm(data_loader, total=len(data_loader), desc="Testing"):
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)

        predictions = torch.sigmoid(outputs) > 0.5
        iou_score = calculate_iou(predictions, masks)
        iou_scores.append(iou_score)

        visualize_segmentation(images, masks, predictions, output_folder, frame_index)

        for i in range(predictions.shape[0]):
            original_name = os.path.splitext(file_names[i])[0]
            mask_pil = to_pil(predictions[i].cpu().float())
            mask_pil.save(os.path.join(predictions_folder, f"{original_name}_prediction.png"))

        frame_index += images.size(0)

    average_iou = np.mean(iou_scores)
    print(f"Average IoU: {average_iou:.4f}")

    gif_path = os.path.join(output_folder, "segmentation_visualization.gif")
    create_gif(output_folder, gif_path)

    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)
    print(f"Testing Successful. GIF saved at {gif_path}")
    print(f"All predicted masks saved in: {predictions_folder}")
