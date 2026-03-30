import os
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
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

# Initialize Neptune
run = None
# run = neptune.init_run(
#     project="XRAI-Pipeline/XAI",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNzM1ZDY3Ny04ODhjLTQwZDktODQyNC0zMGRhNjZjODgwOTQifQ==",#wrongkeyfromhere",
#     name="FcbFormer-Tuning-BCEWithLogitsLoss-Adam-Early-Stopping-And-Swin"
# )

# # # Hyperparameters
# hyperparameters = {
#     "learning_rate": 1e-4,
#     "batch_size": 4,
#     "input_size": 512,
#     "num_epochs": 50,
#     "optimizer": "Adam",
#     "loss_function": "BCEWithLogitsLoss"
# }
#run["parameters"] = hyperparameters


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

# ORIGINAL
# class TB(nn.Module):
#     def __init__(self):

#         super().__init__()

#         backbone = pvt_v2.PyramidVisionTransformerV2(
#             patch_size=4,
#             embed_dims=[64, 128, 320, 512],
#             num_heads=[1, 2, 5, 8],
#             mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             depths=[3, 4, 18, 3],
#             sr_ratios=[8, 4, 2, 1],
#         )

#         # checkpoint = torch.load("/home/mikolaj/oke_laptop/Programs/breast_cancer/segmentation/imports/FCBFormer/pvt_v2_b3.pth")
#         backbone.default_cfg = _cfg()
#         # backbone.load_state_dict(checkpoint)
#         self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

#         for i in [1, 4, 7, 10]:
#             self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

#         self.LE = nn.ModuleList([])
#         for i in range(4):
#             self.LE.append(
#                 nn.Sequential(
#                     RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
#                 )
#             )

#         self.SFA = nn.ModuleList([])
#         for i in range(3):
#             self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

#     def get_pyramid(self, x):
#         pyramid = []
#         B = x.shape[0]
#         for i, module in enumerate(self.backbone):
#             if i in [0, 3, 6, 9]:
#                 x, H, W = module(x)
#             elif i in [1, 4, 7, 10]:
#                 for sub_module in module:
#                     x = sub_module(x, H, W)
#             else:
#                 x = module(x)
#                 x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#                 pyramid.append(x)

#         return pyramid

#     def forward(self, x):
#         pyramid = self.get_pyramid(x)
#         pyramid_emph = []
#         for i, level in enumerate(pyramid):
#             pyramid_emph.append(self.LE[i](pyramid[i]))

#         l_i = pyramid_emph[-1]
#         for i in range(2, -1, -1):
#             l = torch.cat((pyramid_emph[i], l_i), dim=1)
#             l = self.SFA[i](l)
#             l_i = l

#         return l
    

# NEW TRANSFORMERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

class TB(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Vision Transformer (ViT-B/16) as the encoder
        # self.backbone = vit_b_16(weights="DEFAULT")
        
        # Load Vision Transformer (Swin-B/16) as the encoder
        self.backbone = swin_t(weights="DEFAULT")
        
        
        # Remove the classification head (VIT)
        # self.backbone.heads.head = nn.Identity()
        
        # Remove the classification head (Swin)
        self.backbone.head = nn.Identity()


        # Progressive Locality Decoder (PLD+)
        self.LE = nn.ModuleList([
            nn.Sequential(
                RB(768, 64),  # Residual Block (RB)
                RB(64, 64),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(4)
        ])

        self.SFA = nn.ModuleList([
            nn.Sequential(
                RB(128, 64),  # Residual Block (RB)
                RB(64, 64)
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

    
# #Original
# class FCBFormer(nn.Module):
#     def __init__(self, size=512):

#         super().__init__()

#         self.TB = TB()

#         self.FCB = FCB(in_resolution=size)
#         self.PH = nn.Sequential(
#             RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
#         )
#         self.up_tosize = nn.Upsample(size=size)

#     def forward(self, x):
#         print(f"X before forward pass {x.shape}")
#         x1 = self.TB(x)
#         x2 = self.FCB(x)
#         print(f"x1 shape: {x1.shape}")
#         print(f"x2 shape: {x2.shape}")

#         x1 = self.up_tosize(x1)
#         x = torch.cat((x1, x2), dim=1)
#         print("x after torch cat")
#         print(x1.shape)
#         print(x.shape)
#         out = self.PH(x)

        # return out

class FCBFormer(nn.Module):
    def __init__(self, size=512):
        super().__init__()
        self.TB = TB()  # Transformer Branch (ViT)
        self.FCB = FCB(in_resolution=size)  # Fully Convolutional Branch
        self.PH = nn.Sequential(
            RB(64 + 32, 64),  # Residual Block
            RB(64, 64),
            nn.Conv2d(64, 1, kernel_size=1)  # Final prediction layer
        )
        self.up_tosize = nn.Upsample(size=size)  # Upsample to full resolution

    def forward(self, x):
        try:
            # print("Transformer branch output")
            x1 = self.TB(x)  # Transformer Branch output
            # print("TB didn't fail")
            
            # print("Before FCB Branch ")
            x2 = self.FCB(x)  # Fully Convolutional Branch output
            # print("After pass into fcb")
            # Upsample x1 to match the spatial dimensions of x2
            # print(" beforeup_to_size ")
            x1 = self.up_tosize(x1)
            # print("up to size didn't fail")
            # Concatenate x1 and x2 along the channel dimension
            # x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
            # print("shapes of x1 and x2")
            # print(x1.shape,x2.shape)
            # print("concatting x1 and x2")
            if x2.shape[2:] != (512, 512):
                # print("Adjusting x2 to 512x512")
                x2 = F.interpolate(x2, size=(512, 512), mode='bilinear', align_corners=False)
            # print("x2 shape after resizing (if needed):", x2.shape)
            x = torch.cat((x1, x2), dim=1)
            # print("concatted x1 and x2")
            # Pass through the prediction head (PH)
            
            # print("prediction head")
            out = self.PH(x)
            return out
        except Exception as e:
          raise('An exception occurred',e)
        

def train_fcbformer(save_path, data_loader, val_loader, input_size=512, patience=5):
    """
    Train the FCBFormer model with Optuna for hyperparameter tuning and Neptune for tracking results.
    Early stopping is added to prevent overfitting.
    """
    def objective(trial):
        # Suggest hyperparameters using Optuna
        learning_rate = trial.suggest_float("learning_rate", 3e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        num_epochs = trial.suggest_int("num_epochs", 30, 50, step=10)
        loss_function_name = "BCEWithLogitsLoss"
        optimizer_name = "Adam"

        # Log trial hyperparameters to Neptune
        run[f"trial/{trial.number}/hyperparameters"] = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_size": input_size,
            "num_epochs": num_epochs,
            "optimizer": optimizer_name,
            "loss_function": loss_function_name,
        }

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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == "Adam" else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss() if loss_function_name == "BCEWithLogitsLoss" else DiceLoss(to_onehot_y=False, softmax=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Start training
        start_time = time.time()
        best_iou = 0.0
        best_model_path = None
        early_stopping_counter = 0  # For tracking epochs without improvement

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            iou_scores = []

            for batch_idx, (images, masks) in enumerate(data_loader):
                if torch.cuda.is_available():
                    images, masks = images.cuda(), masks.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                iou_scores.append(calculate_iou(torch.sigmoid(outputs) > 0.5, masks))

            epoch_loss = running_loss / len(data_loader)
            epoch_iou = np.mean(iou_scores)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_iou_scores = []
            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    if torch.cuda.is_available():
                        val_images, val_masks = val_images.cuda(), val_masks.cuda()

                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_masks).item()
                    val_iou_scores.append(calculate_iou(torch.sigmoid(val_outputs) > 0.5, val_masks))

            val_loss /= len(val_loader)
            val_iou = np.mean(val_iou_scores)

            # Log metrics
            run[f"trial/{trial.number}/epoch_loss"].append(epoch_loss)
            run[f"trial/{trial.number}/epoch_iou"].append(epoch_iou)
            run[f"trial/{trial.number}/val_loss"].append(val_loss)
            run[f"trial/{trial.number}/val_iou"].append(val_iou)

            # Early stopping
            if val_iou > best_iou:
                best_iou = val_iou
                best_model_path = os.path.join(save_path, f"fcbformer_best_trial_{trial.number}_with_Swin.pth")
                torch.save(model.state_dict(), best_model_path)
                early_stopping_counter = 0  # Reset counter
            else:
                early_stopping_counter += 1

            # Check early stopping condition
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Step the scheduler
            scheduler.step()

        # Log training time
        total_training_time = time.time() - start_time
        run[f"trial/{trial.number}/total_training_time"] = total_training_time / 60.0

        # Upload the best model
        if best_model_path:
            run[f"trial/{trial.number}/best_model"].upload(best_model_path)

        return best_iou

    # Set up Optuna study
    study = optuna.create_study(
        study_name="FCBFormer_Optuna_Study_With_Early_Stopping_Swin",  # Name of the study
        direction="maximize",
        load_if_exists=True  # Load existing study
    )
    neptune_callback = NeptuneCallback(run)

    # Optimize hyperparameters
    try:
        study.optimize(objective, n_trials=7, callbacks=[neptune_callback])
    except Exception as e:
        logger.error(f"Study interrupted: {e}")

    # Save study results to a JSON file
    save_study_to_json(study, optuna_log_file)

    # Log best parameters to Neptune
    run["optuna/best_params"] = study.best_params
    run["optuna/best_value"] = study.best_value
    run["optuna/study_summary"] = str(study)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best IoU: {study.best_value}")

    # Stop the Neptune run
    run.stop()

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

    model_path = os.path.join(config.saved_models_path, 'FCBFormer', dataset, f'Feature_{feature_dataset_choice}', 'fcbformer_segmentation.pth')

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

    # Log results
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)
    print(f"Testing Successful. GIF saved at {gif_path}")