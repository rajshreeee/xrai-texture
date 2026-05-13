"""
Kernel Drift Analysis — CBIS, VGG11UNet, data-size ablation
Produces:
  1. Main table: Frob drift + Cosine sim per condition/fraction/layer
     (injected layer marked with * in output)
  2. C1 comparison CSV: injected enc1.conv[0] drift, Baseline vs Enc1-L1
  3. Main table CSV

Tracked layers:
  enc1.conv[0]  — injected layer for Enc1-L1 (marked *)
  enc2.conv[0]  — first conv of enc2 block (uninjected control, single-conv block)
  enc3.conv[0]  — first conv of enc3 block (uninjected control, 2-conv block)

Note: VGG11 enc1 has n_convs=1, so its Sequential is [Conv2d, BN, ReLU] only —
there is no enc1.conv[3] (unlike VGG16 where enc1 has 2 convs).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import random
import csv
import torch.nn.functional as nnF
from vgg11_unet import VGG11UNet
import config

# ── Paths ─────────────────────────────────────────────────────────────────────

CKPT_DIR   = Path(config.CHECKPOINT_DIR)
OUTPUT_DIR = Path(__file__).parent / "output" / "kernel_drift_datasize"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

FRACTIONS          = [10, 25, 50, 75, 100]
SEED               = 42
KERNEL_JITTER_SEED = 42

# ── Kernels (same as unet.py) ──────────────────────────────────────────────────

_RAW_KERNELS = {
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


def _resize_kernels(kernel_dict, target_size=(3, 3)):
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


KERNELS = [k for group in _resize_kernels(_RAW_KERNELS).values() for k in group]  # 5 kernels

# ── Checkpoint registry ───────────────────────────────────────────────────────

CHECKPOINTS = {
    "Baseline": {"inject_blocks": None,      "ckpt_prefix": "A_baseline"},
    "Enc1-L1":  {"inject_blocks": ["enc1"],  "ckpt_prefix": "B_enc1"},
}
CONDITIONS_ORDER = ["Baseline", "Enc1-L1"]

# ── Layers to analyse ─────────────────────────────────────────────────────────
# (block_name, conv_idx_in_Sequential, short_label)
# VGGBlock(n_convs=1) layout: conv[0]=Conv2d, conv[1]=BN, conv[2]=ReLU
# VGGBlock(n_convs=2) layout: conv[0]=Conv2d, conv[1]=BN, conv[2]=ReLU,
#                              conv[3]=Conv2d, conv[4]=BN, conv[5]=ReLU

LAYERS_TO_ANALYZE = [
    ("enc1", 0, "enc1_conv0"),   # injected for Enc1-L1; shape (64,   3, 3, 3)
    ("enc2", 0, "enc2_conv0"),   # uninjected control (1-conv block); shape (128,  64, 3, 3)
    ("enc3", 0, "enc3_conv0"),   # uninjected control (2-conv block); shape (256, 128, 3, 3)
]

INJECTED_LABEL = {"Enc1-L1": "enc1_conv0"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def inject_kernels(model, block_name, kernels_list, jitter_seed, scale_to_kaiming=True):
    block       = getattr(model, block_name)
    conv_weight = block.conv[0].weight
    n_out, n_in = conv_weight.shape[0], conv_weight.shape[1]
    n_kernels   = len(kernels_list)

    if scale_to_kaiming:
        fan_in      = n_in * 3 * 3
        kaiming_std = np.sqrt(2.0 / fan_in)
        all_vals    = np.concatenate([k.flatten() for k in kernels_list])
        current_std = float(np.std(all_vals))
        scale       = kaiming_std / current_std if current_std > 1e-8 else 1.0
    else:
        scale = 1.0

    rng = np.random.default_rng(jitter_seed)
    with torch.no_grad():
        for out_idx in range(n_out):
            base_k = kernels_list[out_idx % n_kernels]
            for in_idx in range(n_in):
                k = np.rot90(base_k, k=int(rng.integers(0, 4)))
                k = k * rng.choice([-1.0, 1.0])
                k = k * scale
                conv_weight[out_idx, in_idx, :, :] = torch.tensor(
                    k.copy(), dtype=torch.float32
                )


def build_model(seed, inject_blocks=None):
    set_all_seeds(seed)
    model = VGG11UNet(in_channels=3, out_channels=1)
    if inject_blocks is not None:
        for block_name in inject_blocks:
            inject_kernels(model, block_name, KERNELS, KERNEL_JITTER_SEED)
    return model


def get_filters(model, block_name, conv_idx):
    return getattr(model, block_name).conv[conv_idx].weight.detach().cpu().numpy()


# ── Metrics ───────────────────────────────────────────────────────────────────

def frob_drift(W_init, W_final):
    drifts = []
    for oi in range(W_init.shape[0]):
        delta = np.linalg.norm(W_final[oi] - W_init[oi])
        norm0 = np.linalg.norm(W_init[oi]) + 1e-8
        drifts.append(delta / norm0)
    return np.array(drifts)


def cosine_sim(W_init, W_final):
    sims = []
    for oi in range(W_init.shape[0]):
        ch_sims = []
        for ii in range(W_init.shape[1]):
            a = W_init[oi, ii].flatten()
            b = W_final[oi, ii].flatten()
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            ch_sims.append(np.dot(a, b) / denom)
        sims.append(np.mean(ch_sims))
    return np.array(sims)


# ── Core: compute drift for one condition/fraction ────────────────────────────

def compute_drift(condition, fraction_pct):
    cfg  = CHECKPOINTS[condition]
    ckpt = CKPT_DIR / f"{cfg['ckpt_prefix']}_frac{fraction_pct}_seed{SEED}_best.pt"

    if not ckpt.exists():
        print(f"  [MISSING] {condition} frac={fraction_pct}%: {ckpt}")
        return None

    m_init    = build_model(SEED, cfg["inject_blocks"])
    m_trained = build_model(SEED, cfg["inject_blocks"])
    state     = torch.load(ckpt, map_location="cpu", weights_only=True)
    m_trained.load_state_dict(state)
    m_trained.eval()

    result = {}
    for block_name, conv_idx, label in LAYERS_TO_ANALYZE:
        W0 = get_filters(m_init,    block_name, conv_idx)
        W1 = get_filters(m_trained, block_name, conv_idx)
        result[label] = {
            "frob":   frob_drift(W0, W1),
            "cosine": cosine_sim(W0, W1),
        }
    return result


# ── Collect all ───────────────────────────────────────────────────────────────

def collect_all():
    print("\n" + "=" * 60)
    print("  Collecting drift — CBIS, VGG11UNet, data-size")
    print("=" * 60)
    data = {}
    for cond in CONDITIONS_ORDER:
        data[cond] = {}
        for frac in FRACTIONS:
            print(f"  {cond:<12} frac={frac:>3}% ...", end=" ")
            d = compute_drift(cond, frac)
            if d is None:
                continue
            data[cond][frac] = d
            for _, _, label in LAYERS_TO_ANALYZE:
                print(f"{label}: frob={d[label]['frob'].mean():.4f} "
                      f"cos={d[label]['cosine'].mean():.4f}  ", end="")
            print()
    return data


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(data):
    summary = {}
    for cond in CONDITIONS_ORDER:
        summary[cond] = {}
        for frac in FRACTIONS:
            if frac not in data.get(cond, {}):
                continue
            summary[cond][frac] = {}
            for _, _, label in LAYERS_TO_ANALYZE:
                summary[cond][frac][label] = {
                    "frob_mean":   float(data[cond][frac][label]["frob"].mean()),
                    "cosine_mean": float(data[cond][frac][label]["cosine"].mean()),
                }
    return summary


# ── Print main table ──────────────────────────────────────────────────────────

def print_main_table(summary):
    labels = [label for _, _, label in LAYERS_TO_ANALYZE]

    print("\n" + "=" * 120)
    print("  MAIN TABLE — Kernel Drift (seed=42)")
    print("  * = injected layer for that condition")
    print("  Frob drift: lower = more anchored | Cosine sim: higher = more anchored")
    print("=" * 120)

    hdr = f"  {'Condition':<12} {'Frac':>5} | "
    for lbl in labels:
        hdr += f"{'Frob ' + lbl:^16}  {'Cos ' + lbl:^16}  "
    print(hdr)
    print("  " + "-" * 115)

    for cond in CONDITIONS_ORDER:
        inj_label = INJECTED_LABEL.get(cond)
        for frac in FRACTIONS:
            if cond not in summary or frac not in summary[cond]:
                continue
            s   = summary[cond][frac]
            row = f"  {cond:<12} {frac:>4}% | "
            for lbl in labels:
                marker = " *" if lbl == inj_label else "  "
                row += f"{s[lbl]['frob_mean']:.6f}{marker}  {s[lbl]['cosine_mean']:.6f}{marker}  "
            print(row)
        print()


# ── Print C1 comparison ───────────────────────────────────────────────────────

def print_c1_comparison(summary):
    print("  C1: enc1.conv[0] — Baseline vs Enc1-L1 (injected layer)")
    print("  " + "-" * 72)
    print(f"  {'Frac':>5} | {'Baseline Frob':>14} {'Enc1-L1 Frob':>14} "
          f"{'ΔFrob':>10} | {'Baseline Cos':>13} {'Enc1-L1 Cos':>13}")
    print("  " + "-" * 72)

    for frac in FRACTIONS:
        b_row = summary.get("Baseline", {}).get(frac, {}).get("enc1_conv0")
        e_row = summary.get("Enc1-L1",  {}).get(frac, {}).get("enc1_conv0")
        if b_row is None or e_row is None:
            continue
        print(f"  {frac:>4}% | {b_row['frob_mean']:>14.6f} {e_row['frob_mean']:>14.6f} "
              f"{e_row['frob_mean'] - b_row['frob_mean']:>+10.6f} | "
              f"{b_row['cosine_mean']:>13.6f} {e_row['cosine_mean']:>13.6f}")
    print()


# ── CSV exports ───────────────────────────────────────────────────────────────

def export_csvs(summary):
    labels = [label for _, _, label in LAYERS_TO_ANALYZE]

    main_path = OUTPUT_DIR / "main_table_drift_datasize.csv"
    with open(main_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["fraction_pct", "condition", "inject_block"]
        for lbl in labels:
            header += [f"{lbl}_frob", f"{lbl}_cosine"]
        w.writerow(header)
        for cond in CONDITIONS_ORDER:
            cfg = CHECKPOINTS[cond]
            inj = ",".join(cfg["inject_blocks"]) if cfg["inject_blocks"] else "none"
            for frac in FRACTIONS:
                if cond not in summary or frac not in summary[cond]:
                    continue
                row = [frac, cond, inj]
                for lbl in labels:
                    row += [f"{summary[cond][frac][lbl]['frob_mean']:.6f}",
                            f"{summary[cond][frac][lbl]['cosine_mean']:.6f}"]
                w.writerow(row)
    print(f"  Saved: {main_path}")

    c1_path = OUTPUT_DIR / "c1_enc1conv0_injected_vs_baseline.csv"
    with open(c1_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fraction_pct", "baseline_frob", "enc1l1_frob", "delta_frob",
                    "baseline_cosine", "enc1l1_cosine"])
        for frac in FRACTIONS:
            b = summary.get("Baseline", {}).get(frac, {}).get("enc1_conv0")
            e = summary.get("Enc1-L1",  {}).get(frac, {}).get("enc1_conv0")
            if b is None or e is None:
                continue
            w.writerow([frac,
                        f"{b['frob_mean']:.6f}", f"{e['frob_mean']:.6f}",
                        f"{e['frob_mean'] - b['frob_mean']:.6f}",
                        f"{b['cosine_mean']:.6f}", f"{e['cosine_mean']:.6f}"])
    print(f"  Saved: {c1_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data    = collect_all()
    summary = summarise(data)

    print_main_table(summary)
    print_c1_comparison(summary)
    export_csvs(summary)

    print(f"\nAll outputs → {OUTPUT_DIR}")
