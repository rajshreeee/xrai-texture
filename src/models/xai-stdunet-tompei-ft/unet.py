import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import wandb
import json
from pathlib import Path
from dataset import make_loaders
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from logger import setup_logger
import config
import random
from vanilla_unet import VanillaUNet

SEEDS = [42]

EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)
CHECKPOINT_DIR.mkdir(exist_ok=True)
KERNEL_JITTER_SEED = 42  # top-level constant — document this in your paper

# Raw 5x5 texture kernels from feature-ranking analysis.
# Keys are ResNet-style labels — all kernels are injected into the target UNet block.
# Kaiming scaling is NOT applied here; it is applied dynamically in inject_kernels()
# based on the actual layer's fan-in, so the same kernels work correctly for enc1, enc2, etc.
all_layer_kernels = {
    "layer3": [
        np.array([[-10, -6, -3, -2, -3],
                  [ -5,  0,  2,  3,  2],
                  [ -2,  2,  5,  6,  4],
                  [ -2,  2,  4,  5,  3],
                  [ -5,  0,  0,  0,  0]], dtype=np.float32),
        np.array([[ -8, -4,  0, -2, -5],
                  [ -4,  0,  4,  4,  0],
                  [ -3,  3,  6,  6,  4],
                  [ -5,  0,  5,  6,  4],
                  [-10, -4,  0,  2,  0]], dtype=np.float32),
    ],
    "layer4": [
        np.array([[-10, -6, -3, -2, -2],
                  [ -6, -2,  0,  2,  0],
                  [ -3,  0,  3,  4,  3],
                  [ -2,  2,  4,  5,  4],
                  [ -2,  0,  3,  4,  2]], dtype=np.float32),
    ],
    "classifier": [
        np.array([[ -9, -4, -3, -5, -10],
                  [ -4,  0,  4,  3,  -2],
                  [ -3,  4,  7,  7,   2],
                  [ -4,  3,  7,  7,   4],
                  [ -9, -2,  3,  4,   0]], dtype=np.float32),
        np.array([[-10, -5, -2, -2, -3],
                  [ -5,  0,  2,  2,  0],
                  [ -2,  2,  4,  4,  2],
                  [  0,  3,  4,  4,  2],
                  [ -3,  0,  2,  2,  0]], dtype=np.float32),
    ],
}


def resize_kernels_bilinear(kernel_dict, target_size=(3, 3)):
    """
    Resize 5x5 kernels to target_size via bilinear interpolation, then zero-mean
    and unit-std normalise. Kaiming scaling is intentionally omitted so that
    inject_kernels() can apply the correct std for whichever layer is targeted.
    """
    resized = {}
    for layer_name, kernels in kernel_dict.items():
        out = []
        for k in kernels:
            k_t = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)
            k_r = nnF.interpolate(k_t, size=target_size, mode='bilinear', align_corners=False)
            k_f = k_r.squeeze().numpy()
            k_f = k_f - np.mean(k_f)
            s = np.std(k_f)
            if s > 1e-8:
                k_f = k_f / s          # unit std; Kaiming scale applied at injection time
            out.append(k_f.astype(np.float32))
        resized[layer_name] = out
    return resized


_RESIZED = resize_kernels_bilinear(all_layer_kernels)
KERNELS = [k for group in _RESIZED.values() for k in group]  # flat list, 5 kernels


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(seed, inject_blocks=None):
    set_all_seeds(seed)
    model = VanillaUNet(in_channels=3, out_channels=1)
    if inject_blocks is not None:
        for block_name in inject_blocks:
            inject_kernels(model, block_name, KERNELS, KERNEL_JITTER_SEED)
    return model.to(DEVICE)


def inject_kernels(model, block_name, kernels_list, seed, scale_to_kaiming=True):
    """
    Inject texture kernels into the first Conv2d of a DoubleConv encoder block.
    Each (out_idx, in_idx) slot gets an independently jittered variant:
      - random 90° rotation (0/90/180/270)
      - random sign flip (±1)
    This prevents identical weights across channels that occurred in run3-ft.
    scale_to_kaiming rescales kernels to match the He-normal std for the target layer,
    so enc1 (fan_in=27, std≈0.272) and enc2 (fan_in=576, std≈0.059) are both correct.
    """
    block = getattr(model, block_name)       # e.g. model.enc1
    conv_weight = block.conv[0].weight       # shape: (C_out, C_in, 3, 3)

    n_out = conv_weight.shape[0]
    n_in  = conv_weight.shape[1]
    n_kernels = len(kernels_list)

    print(f"Injecting into {block_name}.conv[0] | shape: {tuple(conv_weight.shape)}")
    print(f"Cycling {n_kernels} kernels with sign+orientation jitter across {n_out}×{n_in} slots")

    if scale_to_kaiming:
        fan_in = n_in * 3 * 3
        kaiming_std = np.sqrt(2.0 / fan_in)
        all_vals = np.concatenate([k.flatten() for k in kernels_list])
        current_std = float(np.std(all_vals))
        kaiming_scale = kaiming_std / current_std if current_std > 1e-8 else 1.0
    else:
        kaiming_scale = 1.0
        fan_in = n_in * 3 * 3
        kaiming_std = np.sqrt(2.0 / fan_in)
        current_std = float(np.std(np.concatenate([k.flatten() for k in kernels_list])))
    print(f"  kaiming_scale={kaiming_scale:.4f} (fan_in={fan_in}, kaiming_std={kaiming_std:.4f}, kernel_std={current_std:.4f})")

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for out_idx in range(n_out):
            base_k = kernels_list[out_idx % n_kernels]
            for in_idx in range(n_in):
                k = np.rot90(base_k, k=int(rng.integers(0, 4)))
                k = k * rng.choice([-1.0, 1.0])
                k = k * kaiming_scale
                conv_weight[out_idx, in_idx, :, :] = torch.tensor(
                    k.copy(), dtype=torch.float32
                )


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2 * intersection + eps) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps)).mean()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def tversky_loss(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    """
    alpha: weight on False Negatives
    beta:  weight on False Positives (set high to punish blob predictions)
    """
    pred_sig = torch.sigmoid(pred)
    tp = (pred_sig * target).sum(dim=(2, 3))
    fp = (pred_sig * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - pred_sig) * target).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return (1 - tversky).mean()

def combined_loss(pred, target, bce_weight=0.3):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    tv  = tversky_loss(pred, target, alpha=0.3, beta=0.7)
    return bce_weight * bce + (1 - bce_weight) * tv

# def combined_loss(pred, target, bce_weight=0.5):
#     bce = nn.BCEWithLogitsLoss()(pred, target)
#     pred_sig = torch.sigmoid(pred)
#     intersection = (pred_sig * target).sum(dim=(2,3))
#     dice = 1 - ((2*intersection + 1) / (pred_sig.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1)).mean()
#     return bce_weight * bce + (1 - bce_weight) * dice


def train_one_run(group_name, seed, inject_blocks, train_loader, val_loader, test_loader, freeze_epochs=0):
    set_all_seeds(seed)
    model = build_model(seed, inject_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    if inject_blocks is not None and freeze_epochs > 0:
        for block_name in inject_blocks:
            for p in getattr(model, block_name).conv[0].parameters():
                p.requires_grad = False

    logger = setup_logger(group_name, seed)
    logger.info(f"Starting | group={group_name} | seed={seed} | inject_blocks={inject_blocks} | freeze_epochs={freeze_epochs} | device={DEVICE}")
    logger.info(f"Epochs={EPOCHS} | LR={LR} | BatchSize={BATCH_SIZE}")
    logger.info("-" * 70)

    run = wandb.init(
        entity="rajshreerai931-abo-akademi",
        project=config.PROJECT_NAME,
        name=f"{group_name}_seed{seed}",
        config={
            "group":         group_name,
            "seed":          seed,
            "inject_blocks":  inject_blocks if inject_blocks is not None else [],
            "freeze_epochs": freeze_epochs,
            "epochs":        EPOCHS,
            "lr":            LR,
            "batch_size":    BATCH_SIZE,
            "model":         "VanillaUNet",
            "dataset":       "CBIS_DDSM_augmented",
        },
        reinit=True
    )

    best_val_dice = 0.0
    best_epoch    = 0
    early_dice    = {}

    for epoch in range(1, EPOCHS + 1):
        if inject_blocks is not None and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for block_name in inject_blocks:
                for p in getattr(model, block_name).conv[0].parameters():
                    p.requires_grad = True
            logger.info(f"Unfreezing {inject_blocks} conv[0] at epoch {epoch}")

        # --- Train ---
        model.train()
        train_loss, train_dice, train_iou = 0.0, 0.0, 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss  = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(preds, masks).item()
            train_iou  += iou_score(preds, masks).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou  /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds     = model(images)
                val_loss += combined_loss(preds, masks).item()
                val_dice += dice_score(preds, masks).item()
                val_iou  += iou_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou  /= len(val_loader)

        scheduler.step()

        # --- Test (observation only — no model selection from these numbers) ---
        model.eval()
        test_loss, test_dice, test_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds      = model(images)
                test_loss += combined_loss(preds, masks).item()
                test_dice += dice_score(preds, masks).item()
                test_iou  += iou_score(preds, masks).item()

        test_loss /= len(test_loader)
        test_dice /= len(test_loader)
        test_iou  /= len(test_loader)

        if epoch % 5 == 0:
            early_dice[f"dice_val_epoch{epoch}"] = val_dice
            logger.info(f">>> CHECKPOINT ep{epoch} | Val Dice={val_dice:.4f}")

        improved = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch    = epoch
            torch.save(model.state_dict(),
                       CHECKPOINT_DIR / f"{group_name}_seed{seed}_best.pt")
            improved = "  ★ best"

        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_dice={train_dice:.4f} | train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f}   | val_dice={val_dice:.4f}   | val_iou={val_iou:.4f} | "
            f"test_loss={test_loss:.4f} | test_dice={test_dice:.4f} | test_iou={test_iou:.4f}"
            f"{improved}"
        )

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss":   val_loss,   "val_dice":   val_dice,   "val_iou":   val_iou,
            "test_loss":  test_loss,  "test_dice":  test_dice,  "test_iou":  test_iou,
            "lr": scheduler.get_last_lr()[0],
        })

    logger.info("-" * 70)
    logger.info(f"Done | best_val_dice={best_val_dice:.4f} at epoch {best_epoch}")
    logger.info(f"Early dice: {early_dice}")

    wandb.log(early_dice)
    wandb.log({"best_val_dice": best_val_dice, "best_epoch": best_epoch})
    wandb.finish()

    return {
        "group": group_name, "seed": seed,
        "best_val_dice": best_val_dice, "best_epoch": best_epoch,
        **early_dice
    }


def main():
    conditions = [
        ("A_high_lr_baseline", None, 0),
        # ("B2_enc2_init",       ["enc2"],         0),
        # ("C2_enc2_freeze5",    ["enc2"],         5),
        # ("D_enc2_freeze10",   ["enc2"],        10),
        # ("E_enc1_init",       ["enc1"],         0),
        # ("F_identical_enc1_freeze5",    ["enc1"],         5),
        # ("F_identical_enc1",    ["enc1"],         0),
        ("B_high_lr",    ["enc1"],         5),

        # ("G_enc1_freeze10",   ["enc1"],        10),
        # ("H_enc1enc2_init",   ["enc1", "enc2"], 0),
        # ("I_enc1enc2_freeze5",["enc1", "enc2"], 5),
        # ("J_enc1enc2_freeze5",["enc1", "enc2"], 10),
    ]

    all_results = []

    for group_name, inject_block, freeze_epochs in conditions:
        print(f"\n{'='*50}")
        print(f"Running group: {group_name}")
        for seed in SEEDS:
            print(f"  Seed: {seed}")
            train_loader, val_loader, test_loader = make_loaders(seed, BATCH_SIZE)
            result = train_one_run(group_name, seed, inject_block,
                                   train_loader, val_loader, test_loader,
                                   freeze_epochs=freeze_epochs)
            all_results.append(result)

    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df.groupby("group").agg(["mean", "std"]).round(4)
    print("\n=== Run1 Summary ===")
    print(summary)
    summary.to_csv("phase1_results.csv")


if __name__ == "__main__":
    main()
