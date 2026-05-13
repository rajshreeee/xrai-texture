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
from vgg16_unet import VGG16UNet

SEEDS = [42]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]

MAX_EPOCHS          = 100
EARLY_STOP_PATIENCE = 10
MIN_DELTA           = 0.001  # minimum val_dice improvement to count as progress
T_MAX               = 100    # CosineAnnealingLR period
BATCH_SIZE = 8
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)
CHECKPOINT_DIR.mkdir(exist_ok=True)
KERNEL_JITTER_SEED = 42

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
                k_f = k_f / s
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
    model = VGG16UNet(in_channels=3, out_channels=1)
    if inject_blocks is not None:
        for block_name in inject_blocks:
            inject_kernels(model, block_name, KERNELS, KERNEL_JITTER_SEED)
    return model.to(DEVICE)


def inject_kernels(model, block_name, kernels_list, seed, scale_to_kaiming=True):
    block = getattr(model, block_name)
    conv_weight = block.conv[0].weight  # shape: (C_out, C_in, 3, 3)

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


def train_one_run(group_name, seed, inject_blocks, train_loader, val_loader, test_loader,
                  freeze_epochs=0, train_fraction=1.0,
                  max_epochs=MAX_EPOCHS, t_max=T_MAX,
                  patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA):
    set_all_seeds(seed)
    model = build_model(seed, inject_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-7)

    if inject_blocks is not None and freeze_epochs > 0:
        for block_name in inject_blocks:
            for p in getattr(model, block_name).conv[0].parameters():
                p.requires_grad = False

    frac_tag = f"frac{int(train_fraction * 100)}"
    run_name = f"{group_name}_{frac_tag}_seed{seed}"
    n_train  = len(train_loader.dataset)

    logger = setup_logger(run_name, seed)
    logger.info(f"Starting | run={run_name} | inject_blocks={inject_blocks} | "
                f"train_fraction={train_fraction} | n_train={n_train} | device={DEVICE}")
    logger.info(f"Epochs={max_epochs} | LR={LR} | BatchSize={BATCH_SIZE} | EarlyStopPatience={patience} | MinDelta={min_delta} | T_max={t_max}")
    logger.info("-" * 70)

    run = wandb.init(
        entity="rajshreerai931-abo-akademi",
        project=config.PROJECT_NAME,
        name=run_name,
        config={
            "group":          group_name,
            "seed":           seed,
            "inject_blocks":  inject_blocks if inject_blocks is not None else [],
            "freeze_epochs":  freeze_epochs,
            "epochs":                max_epochs,
            "early_stop_patience":   patience,
            "min_delta":             min_delta,
            "t_max":                 t_max,
            "lr":                    LR,
            "batch_size":     BATCH_SIZE,
            "train_fraction": train_fraction,
            "n_train_images": n_train,
            "model":          "VGG16UNet",
            "dataset":        "TOMPEI",
        },
        reinit=True
    )

    best_val_dice    = 0.0
    best_epoch       = 0
    early_dice       = {}
    patience_counter = 0
    stopped_early    = False

    for epoch in range(1, max_epochs + 1):
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
        if val_dice >= best_val_dice + min_delta:
            best_val_dice    = val_dice
            best_epoch       = epoch
            torch.save(model.state_dict(),
                       CHECKPOINT_DIR / f"{run_name}_best.pt")
            improved         = "  ★ best"
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stop at epoch {epoch} (no val_dice improvement ≥{min_delta} for {patience} epochs)")
                stopped_early = True

        logger.info(
            f"Epoch {epoch:03d}/{max_epochs} | "
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

        if stopped_early:
            break

    logger.info("-" * 70)
    logger.info(f"Done | best_val_dice={best_val_dice:.4f} at epoch {best_epoch} | stopped_early={stopped_early}")
    logger.info(f"Early dice: {early_dice}")

    wandb.log(early_dice)
    wandb.log({"best_val_dice": best_val_dice, "best_epoch": best_epoch,
               "stopped_early": stopped_early, "early_stop_patience": patience})
    wandb.finish()

    return {
        "group":          group_name,
        "seed":           seed,
        "train_fraction": train_fraction,
        "n_train":        n_train,
        "best_val_dice":  best_val_dice,
        "best_epoch":     best_epoch,
        "stopped_early":  stopped_early,
        **early_dice
    }


def main():
    conditions = [
        ("A_baseline", None,     0),
        ("B_enc1",     ["enc1"], 0),
    ]

    all_results = []

    for fraction in FRACTIONS:
        print(f"\n{'='*60}")
        print(f"Train fraction: {fraction} ({int(fraction*100)}%)")
        for group_name, inject_block, freeze_epochs in conditions:
            print(f"\n  Condition: {group_name}")
            for seed in SEEDS:
                print(f"    Seed: {seed}")
                train_loader, val_loader, test_loader = make_loaders(
                    seed, BATCH_SIZE, train_fraction=fraction
                )
                n_train = len(train_loader.dataset)
                print(f"    n_train={n_train} | n_val={len(val_loader.dataset)}")
                result = train_one_run(
                    group_name, seed, inject_block,
                    train_loader, val_loader, test_loader,
                    freeze_epochs=freeze_epochs,
                    train_fraction=fraction,
                )
                all_results.append(result)

    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv("datasize_results.csv", index=False)

    summary = (
        df.groupby(["group", "train_fraction"])["best_val_dice"]
          .agg(["mean", "std"])
          .round(4)
    )
    print("\n=== Data Size Ablation Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
