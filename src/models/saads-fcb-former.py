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

from torchsummary import summary

import os
import torch
import numpy as np
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage


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


class RB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
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
        min_channel_mults=[1, 1, 2, 2, 3],
        n_levels_down=5,
        n_levels_up=5,
        n_RBs=1,
        in_resolution=512,
        dropout_prob=0.3
    ):
        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(min_level_channels),  # Batch Normalization
                nn.Dropout(dropout_prob)  # Dropout
            )]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(
                        RB(ch, min_channel_mult * min_level_channels, dropout_prob),
                        nn.BatchNorm2d(min_channel_mult * min_level_channels),  # Batch Normalization
                        nn.Dropout(dropout_prob)  # Dropout
                    )
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(ch),  # Batch Normalization
                        nn.Dropout(dropout_prob)  # Dropout
                    )
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(
            RB(ch, ch, dropout_prob),
            nn.BatchNorm2d(ch),  # Batch Normalization
            nn.Dropout(dropout_prob),  # Dropout
            RB(ch, ch, dropout_prob),
            nn.BatchNorm2d(ch),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
        )

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                        dropout_prob
                    ),
                    nn.BatchNorm2d(min_channel_mult * min_level_channels),  # Batch Normalization
                    nn.Dropout(dropout_prob)  # Dropout
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                            nn.BatchNorm2d(ch),  # Batch Normalization
                            nn.Dropout(dropout_prob)  # Dropout
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
    def __init__(self, dropout_prob=0.3):
        super().__init__()
        # Load Vision Transformer (ViT-B/16) as the encoder
        self.backbone = vit_b_16(weights="DEFAULT")
        
        # Remove the classification head (VIT)
        self.backbone.heads.head = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Progressive Locality Decoder (PLD+)
        self.LE = nn.ModuleList([
            nn.Sequential(
                RB(768, 64, dropout_prob),  # Residual Block (RB)
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                RB(64, 64, dropout_prob),
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(4)
        ])

        self.SFA = nn.ModuleList([
            nn.Sequential(
                RB(128, 64, dropout_prob),  # Residual Block (RB)
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                RB(64, 64, dropout_prob),
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob)  # Dropout
            ) for _ in range(3)
        ])

    def get_pyramid(self, x):
        """
        Extract feature pyramid from Vision Transformer (ViT) encoder.
        """
        # Forward pass through ViT
        x = self.backbone(x)  # Output shape: [B, 768]

        # Reshape to [B, 768, 1, 1] (since ViT outputs a 1D feature vector)
        B, C = x.shape
        x = x.reshape(B, C, 1, 1)

        # Upsample to [B, 768, 14, 14] to match the expected spatial dimensions
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)

        # Repeat the same feature map for all levels (since ViT is non-hierarchical)
        pyramid = [x] * 4
        return pyramid

    def forward(self, x):
        """
        Forward pass for the Transformer Branch (TB) with Vision Transformer (ViT).
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

        return l_i


class FCBFormer(nn.Module):
    def __init__(self, size=512, dropout_prob=0.3):
        super().__init__()
        self.TB = TB(dropout_prob=dropout_prob)  # Transformer Branch (ViT)
        self.FCB = FCB(in_resolution=size, dropout_prob=dropout_prob)  # Fully Convolutional Branch
        self.PH = nn.Sequential(
            RB(64 + 32, 64, dropout_prob),  # Residual Block
            RB(64, 64, dropout_prob),
            nn.Conv2d(64, 1, kernel_size=1)  # Final prediction layer
        )
        self.up_tosize = nn.Upsample(size=size)  # Upsample to full resolution

    def forward(self, x):
        try:
            x1 = self.TB(x)  # Transformer Branch output
            x2 = self.FCB(x)  # Fully Convolutional Branch output
            x1 = self.up_tosize(x1)
            if x2.shape[2:] != (512, 512):
                x2 = F.interpolate(x2, size=(512, 512), mode='bilinear', align_corners=False)
            x = torch.cat((x1, x2), dim=1)
            out = self.PH(x)
            return out
        except Exception as e:
            raise Exception('An exception occurred during forward pass') from e        


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
        print(" Model loaded successfully.")
    except RuntimeError as e:
        print(" Error loading state_dict:", e)
        raise e

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model.eval()


def train_fcbformer(save_path, data_loader, val_loader, input_size=512, patience=5, learning_rate=4.7751948374780065e-05, batch_size=8, num_epochs=20):
    """
    Train the FCBFormer model with early stopping.
    Saves only the best model (highest IoU) instead of saving every improvement.
    """

    # Initialize the FCBFormer model
    model = FCBFormer(size=input_size)
    if torch.cuda.is_available():
        device_id = [0, 1]
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = nn.DataParallel(model, device_ids=device_id)
        model = model.cuda()
    else:
        model = model.cpu()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Start training
    start_time = time.time()
    best_iou = 0.0
    best_model_path = os.path.join(save_path, "fcbformer_best.pth")  # Fixed path to overwrite
    early_stopping_counter = 0  # For tracking epochs without improvement
    best_dice=0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        iou_scores = []
        dice_scores = []

        for batch_idx, (images, masks) in tqdm(enumerate(data_loader),desc=f"Training: Epoch {epoch}/{num_epochs}", total=len(data_loader)):
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
            for val_images, val_masks in tqdm(val_loader,desc=f"Validation: Epoch {epoch}/{num_epochs}", total=len(val_loader)):
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

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - IoU: {epoch_iou:.4f} - Dice: {epoch_dice:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f} - Val Dice: {val_dice:.4f}")
                
        # Save only the best model (overwrite previous one)
        if val_iou > best_iou:
            best_iou = val_iou
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            early_stopping_counter = 0  # Reset counter
            print(f"New best model saved with IoU: {best_iou:.4f}")
        else:
            early_stopping_counter += 1

        # Check early stopping condition
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Step the scheduler
        scheduler.step()

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time / 60:.2f} minutes. Best IoU: {best_iou:.4f}. Best Dice Score: {best_dice:.4f}")

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



def create_gif(frame_folder, gif_path):
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")])
    images = [Image.open(frame) for frame in frames]
    imageio.mimsave(gif_path, images, fps=2) 

def save_fcbformer_layers(model, file_path):
    """Recursively saves all layers and sub-layers of the FCBFormer model to a text file."""
    def write_layer(f, module, indent=0):
        """Helper function to write layers recursively with indentation."""
        for name, layer in module.named_children():
            f.write("  " * indent + f"├── {name}: {layer.__class__.__name__}\n")
            write_layer(f, layer, indent + 1)  # Recursively process sub-layers

    with open(file_path, "w") as f:
        f.write("Layers in FCBFormer during testing (including sub-layers):\n\n")
        for name, layer in model.named_children():
            f.write(f"Layer Name: {name}, Type: {layer.__class__.__name__}\n")
            write_layer(f, layer, indent=1)  # Start recursion



def test_fcbformer(result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    print("Testing FCBFormer on Feature:", feature_dataset_choice, "of", dataset)

    model_path = os.path.join(
        config.saved_models_path, 'FCBFormer', dataset, f'Feature_{feature_dataset_choice}', 'fcbformer_best.pth'
    )

    print("=" * 20)
    print(model_path)
    print("=" * 20)

    if not os.path.exists(model_path):
        print("The given model does not exist. Train the model before testing.")
        return

    # Use the new loader
    model = load_fcbformer(model_path)

    model.eval()
    iou_scores = []

    # Prepare output folders
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
