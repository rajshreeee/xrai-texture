# ============================================================
#  INFERENCE VISUALIZER
#  Original | Ground Truth | Prediction | Prob Map | Overlay
# ============================================================

import argparse
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch

from dataset import make_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SECTION 1 — CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=False,  help="Full path to .pt file", default="/ediss_data/ediss2/xai-texture/src/models/xai-model-run3-ft/checkpoints/D_layer1_init_seed42_best.pt")
    p.add_argument("--domain",       required=False,  default="fourier transform")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n_samples",    type=int, default=10)
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--out_dir",      default="output/inference_vis")
    return p.parse_args()


# ============================================================
# SECTION 2 — MODEL LOAD
# ============================================================

def load_model(checkpoint: str) -> torch.nn.Module:
    """
    Build a bare UNet (resnet18, no pretrained weights) and load checkpoint.
    Architecture must match whatever was used during training.
    No seed/injection needed — we're just restoring trained weights.
    """
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print(f"  Loaded: {checkpoint}")
    return model


# ============================================================
# SECTION 3 — METRICS
# ============================================================

def iou_score(pred: np.ndarray, mask: np.ndarray, eps=1e-6) -> float:
    inter = (pred & mask).sum()
    union = (pred | mask).sum()
    return float((inter + eps) / (union + eps))


def dice_score(pred: np.ndarray, mask: np.ndarray, eps=1e-6) -> float:
    inter = (pred & mask).sum()
    return float((2 * inter + eps) / (pred.sum() + mask.sum() + eps))


# ============================================================
# SECTION 4 — TENSOR HELPERS
# ============================================================

def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.transpose(img, (1, 2, 0))
    # undo ImageNet normalisation if needed
    if img.min() < 0 or img.max() <= 1.05:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)          # keep as float [0,1] for imshow


def tensor_to_mask(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().squeeze().numpy().astype(bool)


def logits_to_pred(logits: torch.Tensor, threshold: float):
    prob = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
    return (prob > threshold).astype(bool), prob


# ============================================================
# SECTION 5 — TP/FP/FN OVERLAY
# ============================================================

def build_overlay(image: np.ndarray,        # float [0,1] HxWx3
                  pred:  np.ndarray,         # bool HxW
                  mask:  np.ndarray,         # bool HxW
                  alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    regions = {
        "tp": ( pred &  mask, np.array([0.00, 0.78, 0.31])),  # green
        "fp": ( pred & ~mask, np.array([0.86, 0.16, 0.16])),  # red
        "fn": (~pred &  mask, np.array([0.16, 0.39, 0.86])),  # blue
    }
    for region, colour in regions.values():
        overlay[region] = (1 - alpha) * overlay[region] + alpha * colour
    return np.clip(overlay, 0, 1)


# ============================================================
# SECTION 6 — PER-SAMPLE FIGURE  (5 panels)
# ============================================================

def save_sample_figure(idx, image, mask, pred, prob, iou, dice, domain, out_dir):
    overlay = build_overlay(image, pred, mask)
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.5))
    fig.suptitle(
        f"{domain.upper()}  ·  Sample {idx+1}  ·  IoU={iou:.4f}  Dice={dice:.4f}",
        fontsize=12, fontweight="bold", y=1.02,
    )

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=10, fontweight="bold")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction", fontsize=10, fontweight="bold")

    axes[3].imshow(prob, cmap="RdYlGn", vmin=0, vmax=1)
    axes[3].set_title("Probability Map", fontsize=10, fontweight="bold")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn",
                              norm=plt.Normalize(0, 1)),
        ax=axes[3], fraction=0.046, pad=0.04,
    )

    axes[4].imshow(overlay)
    axes[4].set_title("TP / FP / FN Overlay", fontsize=10, fontweight="bold")
    legend_patches = [
        mpatches.Patch(color=(0.00, 0.78, 0.31), label="TP"),
        mpatches.Patch(color=(0.86, 0.16, 0.16), label="FP"),
        mpatches.Patch(color=(0.16, 0.39, 0.86), label="FN"),
    ]
    axes[4].legend(handles=legend_patches, loc="lower right",
                   fontsize=8, framealpha=0.7)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fname = out_dir / f"{domain}_sample{idx+1:02d}_iou{iou:.3f}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=180)
    plt.close()
    return fname


# ============================================================
# SECTION 7 — SUMMARY GRID  (all samples, 4 cols)
# ============================================================

def save_summary_grid(samples, domain, seed, out_dir):
    n   = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for ci, title in enumerate(["Original", "Ground Truth", "Prediction", "TP/FP/FN Overlay"]):
        axes[0, ci].set_title(title, fontsize=11, fontweight="bold", pad=6)

    mean_iou  = np.mean([s["iou"]  for s in samples])
    mean_dice = np.mean([s["dice"] for s in samples])
    fig.suptitle(
        f"{domain.upper()}  ·  seed={seed}  ·  "
        f"Mean IoU={mean_iou:.4f}  Mean Dice={mean_dice:.4f}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ri, s in enumerate(samples):
        overlay = build_overlay(s["image"], s["pred"], s["mask"])
        row = [s["image"], s["mask"], s["pred"], overlay]
        cmaps = [None, "gray", "gray", None]
        for ci, (img_data, cmap) in enumerate(zip(row, cmaps)):
            ax = axes[ri, ci]
            ax.imshow(img_data, cmap=cmap, vmin=(0 if cmap else None),
                      vmax=(1 if cmap else None))
            ax.axis("off")
        axes[ri, 0].set_ylabel(
            f"#{s['idx']+1}  IoU={s['iou']:.3f}\nDice={s['dice']:.3f}",
            fontsize=8, rotation=0, labelpad=72, va="center",
        )

    plt.tight_layout()
    fname = out_dir / f"{domain}_seed{seed}_summary.png"
    plt.savefig(fname, bbox_inches="tight", dpi=180)
    plt.close()
    return fname


# ============================================================
# SECTION 8 — METRICS CSV
# ============================================================

def save_csv(records, domain, seed, out_dir):
    import csv
    fname = out_dir / f"{domain}_seed{seed}_metrics.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    return fname


# ============================================================
# SECTION 9 — MAIN
# ============================================================

def run(args):
    out_dir = Path(args.out_dir) / f"{args.domain}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  INFERENCE  |  {args.domain}  |  seed={args.seed}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {args.n_samples}  |  Threshold: {args.threshold}")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}\n")

    model = load_model(args.checkpoint)

    _, _, test_loader = make_loaders(args.seed, batch_size=1)

    samples  = []
    records  = []

    for batch_idx, (images, masks) in enumerate(test_loader):
        if batch_idx >= args.n_samples:
            break

        with torch.no_grad():
            logits = model(images.to(DEVICE))

        image    = tensor_to_rgb(images[0])
        mask     = tensor_to_mask(masks[0])
        pred, prob = logits_to_pred(logits[0], args.threshold)

        iou  = iou_score(pred, mask)
        dice = dice_score(pred, mask)

        print(f"  Sample {batch_idx+1:02d}  IoU={iou:.4f}  Dice={dice:.4f}  "
              f"GT={mask.sum()}px  Pred={pred.sum()}px")

        save_sample_figure(batch_idx, image, mask, pred, prob,
                           iou, dice, args.domain, out_dir)

        samples.append({"idx": batch_idx, "image": image,
                        "mask": mask, "pred": pred,
                        "iou": iou, "dice": dice})
        records.append({"sample": batch_idx+1, "iou": round(iou, 4),
                        "dice": round(dice, 4), "pred_px": int(pred.sum()),
                        "gt_px": int(mask.sum()), "total_px": mask.size})

    grid_path = save_summary_grid(samples, args.domain, args.seed, out_dir)
    csv_path  = save_csv(records, args.domain, args.seed, out_dir)

    ious = [r["iou"] for r in records]
    print(f"\n{'─'*45}")
    print(f"  Mean IoU  : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"  Mean Dice : {np.mean([r['dice'] for r in records]):.4f}")
    print(f"  Best IoU  : {max(ious):.4f}  (sample {ious.index(max(ious))+1})")
    print(f"  Worst IoU : {min(ious):.4f}  (sample {ious.index(min(ious))+1})")
    print(f"  Grid  → {grid_path}")
    print(f"  CSV   → {csv_path}\n")


if __name__ == "__main__":
    run(parse_args())
