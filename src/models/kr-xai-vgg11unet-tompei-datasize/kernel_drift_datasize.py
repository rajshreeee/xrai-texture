"""
Kernel Drift Analysis — TOMPEI, VGG11UNet, data-size ablation (kr variant)
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

# ── Kernels ───────────────────────────────────────────────────────────────────

KR_KERNELS = [
    np.array([
        [-0.00371635, -0.00391476, -0.00411817],
        [ 0.00164007,  0.00428494,  0.00119254],
        [ 0.00182289,  0.00264156,  0.00016729]
    ], dtype=np.float32),

    np.array([
        [ 0.00219253, -0.00211677,  0.00200776],
        [ 0.00228668, -0.00964566,  0.00172222],
        [ 0.00288709, -0.00198506,  0.00265121]
    ], dtype=np.float32),

    np.array([
        [-0.00187108,  0.00003464, -0.00066214],
        [-0.00096704,  0.00141499,  0.00207667],
        [-0.00146049,  0.00157941, -0.00014496]
    ], dtype=np.float32),

    np.array([
        [-0.00062489, -0.00180864, -0.00152716],
        [ 0.00129044,  0.00058043,  0.00068297],
        [ 0.00081856,  0.00033800,  0.00025030]
    ], dtype=np.float32),

    np.array([
        [ 0.00162122,  0.00090959,  0.00158802],
        [-0.00059961, -0.00285438, -0.00051475],
        [-0.00054948, -0.00002701,  0.00042640]
    ], dtype=np.float32),

    np.array([
        [-0.00578342, -0.00562701, -0.00593188],
        [ 0.00246649,  0.00240962,  0.00278980],
        [ 0.00337877,  0.00304987,  0.00324776]
    ], dtype=np.float32),

    np.array([
        [-0.00090438, -0.00020711, -0.00083958],
        [ 0.00000049,  0.00150385,  0.00004852],
        [-0.00053568,  0.00057249,  0.00036139]
    ], dtype=np.float32),

    np.array([
        [-0.00204773,  0.00169760, -0.00277730],
        [ 0.00344980,  0.00299332,  0.00312406],
        [-0.00378768,  0.00136014, -0.00401221]
    ], dtype=np.float32),

    np.array([
        [-0.00137177, -0.00054889, -0.00192962],
        [ 0.00069109,  0.00783324, -0.00041324],
        [-0.00151642, -0.00076535, -0.00197903]
    ], dtype=np.float32),

    np.array([
        [ 0.00230449, -0.00126131,  0.00217926],
        [ 0.00055780, -0.00718292,  0.00022351],
        [ 0.00191462, -0.00085925,  0.00212381]
    ], dtype=np.float32),
]


def _preprocess_kernels_kr(kernels):
    """Mean-center only — kernels are already 3×3, no resize or unit-variance normalization."""
    out = []
    for k in kernels:
        k_f = k - np.mean(k)
        out.append(k_f.astype(np.float32))
    return out


KERNELS = _preprocess_kernels_kr(KR_KERNELS)  # 10 kernels

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
    print("  Collecting drift — TOMPEI, VGG11UNet, data-size (kr variant)")
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
