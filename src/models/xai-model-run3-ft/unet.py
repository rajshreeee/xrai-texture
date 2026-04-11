import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.ndimage import zoom
import wandb
import json
from pathlib import Path
from dataset import make_loaders
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from logger import setup_logger
import config
import random

# SEEDS = [42, 123, 256]
SEEDS = [123, 256]

EPOCHS = 80
BATCH_SIZE = 8
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)
CHECKPOINT_DIR.mkdir(exist_ok=True)

kernels_by_layer = {
    "layer3": [
        np.array([
            [-0.11768152, -0.02260669, -0.01690219],
            [-0.01119771,  0.08577861,  0.08007413],
            [-0.04162164,  0.02302924,  0.02112775]
        ], dtype=np.float32),
        np.array([
            [-0.08016192,  0.01866784, -0.03404136],
            [-0.01592258,  0.08784867,  0.06808272],
            [-0.09663353,  0.02360934,  0.02855083]
        ], dtype=np.float32)
    ],
    "layer4": [
        np.array([
        [-0.12475829, -0.03309914, -0.01782261],
        [-0.03309914,  0.05283131,  0.05856001],
        [-0.01782261,  0.05856001,  0.05665045]
        ], dtype=np.float32)
    ],
    "classifier": [
        np.array([
       [-0.07598736, -0.00847928, -0.07451978],
       [-0.00847928,  0.09278284,  0.04875584],
       [-0.06571438,  0.05756124,  0.03408018]
        ], dtype=np.float32),
        np.array([
       [-0.13138972, -0.01335755, -0.03084379],
       [-0.01335755,  0.07844524,  0.05221588],
       [-0.01991489,  0.05221588,  0.02598652]
        ], dtype=np.float32)
    ]
}

kernels = [k for kernel_values in kernels_by_layer.values() for k in kernel_values]


def set_all_seeds(seed):
    # ✏️ CHANGE: extracted into a single function so every run
    # (build_model, DataLoader, augmentation) uses the exact same
    # seed → guarantees cross-condition comparability per seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(seed, inject_layer=None):
    set_all_seeds(seed)  # ✏️ CHANGE: uses shared helper

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if inject_layer is not None:
        if inject_layer == "all":
            for layer_name in ["layer1", "layer2"]:
                inject_kernels(model, layer_name, kernels)
        else:
            inject_kernels(model, inject_layer, kernels)

    return model.to(DEVICE)


def inject_kernels(model, layer_name, kernels_list):
    target = getattr(model.encoder, layer_name)
    conv_weight = target[0].conv1.weight  # shape: (C_out, C_in, 3, 3)

    n_out = conv_weight.shape[0]
    n_in  = conv_weight.shape[1]
    n_kernels = len(kernels_list)

    print(f"Injecting into encoder.{layer_name}[0].conv1 | shape: {conv_weight.shape}")
    print(f"Cycling {n_kernels} kernels across {n_out} output channels × {n_in} input channels")

    with torch.no_grad():
        for out_idx in range(n_out):
            k = kernels_list[out_idx % n_kernels]
            k_tensor = torch.tensor(k, dtype=torch.float32)
            for in_idx in range(n_in):
                conv_weight[out_idx, in_idx, :, :] = k_tensor


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2 * intersection + eps) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps)).mean()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def combined_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2,3))
    dice = 1 - ((2*intersection + 1) / (pred_sig.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1)).mean()
    return bce_weight * bce + (1 - bce_weight) * dice


def train_one_run(group_name, seed, inject_layer, train_loader, val_loader, test_loader):
    # ✏️ CHANGE: test_loader added as parameter

    model = build_model(seed, inject_layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # ✏️ CHANGE: cosine LR decay added
    

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    logger = setup_logger(group_name, seed)
    logger.info(f"Starting | group={group_name} | seed={seed} | inject_layer={inject_layer} | device={DEVICE}")
    logger.info(f"Epochs={EPOCHS} | LR={LR} | BatchSize={BATCH_SIZE}")
    logger.info("-" * 70)

    run = wandb.init(
        entity="rajshreerai931-abo-akademi",
        project=config.PROJECT_NAME,
        name=f"{group_name}_seed{seed}",
        config={
            "group": group_name,
            "seed": seed,
            "inject_layer": inject_layer,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        reinit=True
    )

    best_val_dice = 0.0
    best_epoch    = 0        # ✏️ CHANGE: track which epoch was best
    early_dice    = {}

    for epoch in range(1, EPOCHS + 1):
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

        # ✏️ CHANGE: cosine scheduler steps here, after val, once per epoch
        scheduler.step()

        # ✏️ CHANGE: test evaluation per epoch — post-hoc observation only,
        # NO model selection or decisions made from these numbers
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

        # Early convergence checkpoints
        if epoch % 5 == 0:
            early_dice[f"dice_val_epoch{epoch}"] = val_dice
            logger.info(f">>> CHECKPOINT ep{epoch} | Val Dice={val_dice:.4f}")

        # ✏️ CHANGE: best checkpoint still chosen by val_dice only — not test
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
            "val_loss":   val_loss,   "val_dice":   val_dice,   "val_iou":  val_iou,
            # ✏️ CHANGE: test metrics logged to wandb as observation curves
            "test_loss":  test_loss,  "test_dice":  test_dice,  "test_iou": test_iou,
            # ✏️ CHANGE: log current LR so you can verify cosine decay in wandb
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


def compute_kernel_drift(group_name, seed, inject_layer, kernels_3x3):
    model = build_model(seed=seed, inject_layer=None)
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"{group_name}_seed{seed}_best.pt",
                   map_location=DEVICE)
    )

    target = getattr(model.encoder, inject_layer)
    conv_weight = target[0].conv1.weight.detach().cpu().numpy()

    drifts = []
    for i, k_init in enumerate(kernels_3x3):
        if i >= conv_weight.shape[0]:
            break
        w_final = conv_weight[i, 0]
        drift = np.linalg.norm(w_final - k_init) / (np.linalg.norm(k_init) + 1e-8)
        drifts.append({"kernel_id": i, "drift": drift})
        print(f"  Kernel {i} drift: {drift:.4f}")

    return drifts


def main():
    conditions = [
        ("A_baseline",       None),
        # ("B_layer2_init",    "layer2"),
        # ("C_layer3_init",    "layer3"),
        # ("D_layer1_init",    "layer1"),
        # ("E_alllayers_init", "all"),
    ]

    all_results = []

    for group_name, inject_layer in conditions:
        print(f"\n{'='*50}")
        print(f"Running group: {group_name}")
        for seed in SEEDS:
            print(f"  Seed: {seed}")
            # ✏️ CHANGE: make_loaders now returns test_loader too
            train_loader, val_loader, test_loader = make_loaders(seed, BATCH_SIZE)
            result = train_one_run(group_name, seed, inject_layer,
                                   train_loader, val_loader, test_loader)
            all_results.append(result)

    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df.groupby("group").agg(["mean", "std"]).round(4)
    print("\n=== Phase 1 Summary ===")
    print(summary)
    summary.to_csv("phase1_results.csv")


if __name__ == "__main__":
    main()