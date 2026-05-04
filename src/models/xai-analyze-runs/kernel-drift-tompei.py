"""
Kernel Drift Analysis — CBIS-DDSM, U-Net ResNet-18
Produces:
  1. Main table: mean±std Frob drift + cosine sim, per condition/layer
     (injected layer bolded in output)
  2. Appendix table: per-seed breakdown
  3. CSV exports for both

Comparisons covered:
  C1: injected-layer vs baseline same-layer (cross-condition)
  C2: injected-layer vs non-injected-layer within same model
  C3: L1-injected vs L2-injected — does anchoring follow injection site?
"""

import torch
import numpy as np
import random
from pathlib import Path
import csv
import segmentation_models_pytorch as smp

# ── Paths ────────────────────────────────────────────────────────────────────

RUN3 = Path("/ediss_data/ediss2/xai-texture/src/models/xai-tompei-run1-ft/checkpoints")
RUN4 = Path("/ediss_data/ediss2/xai-texture/src/models/xai-tompei-run2-kr/checkpoints")
OUTPUT_DIR = Path("output/kernel_drift_tompei")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 256]

# ── Checkpoint registry ───────────────────────────────────────────────────────
# inject_layer: which layer was injected during this training run (None = baseline)
CHECKPOINTS = {
    "Baseline": {
        "inject_layer": None,
        "seeds": {
            42:  RUN3 / "A_baseline_seed42_best.pt",
            123: RUN3 / "A_baseline_seed123_best.pt",
            256: RUN3 / "A_baseline_seed256_best.pt",
        },
    },
    "Fourier-L1": {
        "inject_layer": "layer1",
        "seeds": {
            42:  RUN3 / "FT_D_layer1_init_seed42_best.pt",
            123: RUN3 / "FT_D_layer1_init_seed123_best.pt",
            256: RUN3 / "FT_D_layer1_init_seed256_best.pt",
        },
    },
    "Fourier-L2": {
        "inject_layer": "layer2",
        "seeds": {
            42:  RUN3 / "FT_B_layer2_init_seed42_best.pt",      # update if path differs
            123: RUN3 / "FT_B_layer2_init_seed123_best.pt",
            256: RUN3 / "FT_B_layer2_init_seed256_best.pt",
        },
    },
    "KR-L1": {
        "inject_layer": "layer1",
        "seeds": {
            42:  RUN4 / "KR_D_layer1_init_seed42_best.pt",
            123: RUN4 / "KR_D_layer1_init_seed123_best.pt",
            256: RUN4 / "KR_D_layer1_init_seed256_best.pt",
        },
    },
    "KR-L2": {
        "inject_layer": "layer2",
        "seeds": {
            42:  RUN4 / "KR_B_layer2_init_seed42_best.pt",      # update if path differs
            123: RUN4 / "KR_B_layer2_init_seed123_best.pt",
            256: RUN4 / "KR_B_layer2_init_seed256_best.pt",
        },
    },
}

# ── Reference kernels (epoch-0 injection values) ──────────────────────────────
FOURIER_KERNELS = [
    np.array([[-0.11768152, -0.02260669, -0.01690219],
              [-0.01119771,  0.08577861,  0.08007413],
              [-0.04162164,  0.02302924,  0.02112775]], dtype=np.float32),
    np.array([[-0.08016192,  0.01866784, -0.03404136],
              [-0.01592258,  0.08784867,  0.06808272],
              [-0.09663353,  0.02360934,  0.02855083]], dtype=np.float32),
    np.array([[-0.12475829, -0.03309914, -0.01782261],
              [-0.03309914,  0.05283131,  0.05856001],
              [-0.01782261,  0.05856001,  0.05665045]], dtype=np.float32),
    np.array([[-0.07598736, -0.00847928, -0.07451978],
              [-0.00847928,  0.09278284,  0.04875584],
              [-0.06571438,  0.05756124,  0.03408018]], dtype=np.float32),
    np.array([[-0.13138972, -0.01335755, -0.03084379],
              [-0.01335755,  0.07844524,  0.05221588],
              [-0.01991489,  0.05221588,  0.02598652]], dtype=np.float32),
]

KR_KERNELS = [
    np.array([[-0.07396229, -0.07791102, -0.08195926],
              [ 0.03264045,  0.08527829,  0.02373377],
              [ 0.03627891,  0.05257197,  0.00332938]], dtype=np.float32),
    np.array([[ 0.03350035, -0.03234279,  0.0306772 ],
              [ 0.0349389 , -0.14737907,  0.02631434],
              [ 0.04411276, -0.03033036,  0.04050867]], dtype=np.float32),
    np.array([[-0.08310985,  0.00153864, -0.02941101],
              [-0.0429541 ,  0.06285119,  0.09224177],
              [-0.06487222,  0.07015442, -0.00643885]], dtype=np.float32),
    np.array([[-0.03620493, -0.10478914, -0.08848073],
              [ 0.07476562,  0.033629  ,  0.03956997],
              [ 0.0474258 ,  0.01958307,  0.0145019 ]], dtype=np.float32),
    np.array([[ 0.07342777,  0.04119685,  0.07192409],
              [-0.02715734, -0.12927965, -0.02331389],
              [-0.02488687, -0.00122333,  0.01931237]], dtype=np.float32),
    np.array([[-0.0831344 , -0.08088607, -0.08526846],
              [ 0.03545483,  0.03463735,  0.04010229],
              [ 0.0485685 ,  0.04384069,  0.04668528]], dtype=np.float32),
    np.array([[-0.07474542, -0.01711728, -0.06938981],
              [ 0.0000405 ,  0.12429056,  0.00401009],
              [-0.04427301,  0.04731529,  0.02986825]], dtype=np.float32),
    np.array([[-0.0410655 ,  0.03404394, -0.05569641],
              [ 0.06918283,  0.06002851,  0.06265039],
              [-0.07595873,  0.02727646, -0.08046149]], dtype=np.float32),
    np.array([[-0.02805424, -0.01122542, -0.0394629 ],
              [ 0.01413357,  0.16019858, -0.00845122],
              [-0.0310125 , -0.01565227, -0.04047339]], dtype=np.float32),
    np.array([[ 0.0478375 , -0.02618276,  0.04523792],
              [ 0.01157903, -0.14910583,  0.00463971],
              [ 0.03974442, -0.01783664,  0.04408687]], dtype=np.float32),
]

DOMAIN_KERNELS = {
    "Fourier-L1": FOURIER_KERNELS,
    "Fourier-L2": FOURIER_KERNELS,
    "KR-L1":      KR_KERNELS,
    "KR-L2":      KR_KERNELS,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(seed, inject_layer=None, kernels=None):
    set_seed(seed)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=3, classes=1)
    if inject_layer and kernels:
        _inject(model, inject_layer, kernels)
    return model


def _inject(model, layer_name, kernels):
    target = getattr(model.encoder, layer_name)
    w = target[0].conv1.weight
    n_out, n_in, n_k = w.shape[0], w.shape[1], len(kernels)
    with torch.no_grad():
        for oi in range(n_out):
            k = torch.tensor(kernels[oi % n_k], dtype=torch.float32)
            for ii in range(n_in):
                w[oi, ii] = k


def get_filters(model, layer_name):
    return getattr(model.encoder, layer_name)[0].conv1.weight.detach().cpu().numpy()


# ── Metrics ───────────────────────────────────────────────────────────────────

def frob_drift(W_init, W_final):
    """Relative Frobenius drift per output filter → shape (n_out,)"""
    drifts = []
    for oi in range(W_init.shape[0]):
        delta = np.linalg.norm(W_final[oi] - W_init[oi])
        norm0 = np.linalg.norm(W_init[oi]) + 1e-8
        drifts.append(delta / norm0)
    return np.array(drifts)


def cosine_sim(W_init, W_final):
    """Mean cosine similarity across input channels, per output filter → shape (n_out,)"""
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


# ── Core: compute drift for one model/seed ────────────────────────────────────

def compute_drift(condition, seed):
    """Returns {layer: {frob: array, cosine: array}} or None if checkpoint missing."""
    cfg      = CHECKPOINTS[condition]
    inj_lay  = cfg["inject_layer"]
    kernels  = DOMAIN_KERNELS.get(condition)
    ckpt     = cfg["seeds"][seed]

    if not ckpt.exists():
        print(f"  [MISSING] {condition} seed={seed}: {ckpt}")
        return None

    m_init    = build_model(seed, inject_layer=inj_lay, kernels=kernels)
    m_trained = build_model(seed, inject_layer=inj_lay, kernels=kernels)
    state     = torch.load(ckpt, map_location="cpu")
    m_trained.load_state_dict(state)
    m_trained.eval()

    result = {}
    for layer in ["layer1", "layer2"]:
        W0  = get_filters(m_init,    layer)
        W1  = get_filters(m_trained, layer)
        assert np.abs(W1 - W0).max() > 1e-6, \
            f"CRITICAL: weights unchanged — {condition}/{layer}/seed={seed}"
        result[layer] = {
            "frob":   frob_drift(W0, W1),
            "cosine": cosine_sim(W0, W1),
        }
    return result


# ── Collect all ───────────────────────────────────────────────────────────────

def collect_all():
    print("\n" + "="*60)
    print("  Collecting drift — CBIS-DDSM, U-Net ResNet-18")
    print("="*60)
    data = {}
    for cond in CHECKPOINTS:
        data[cond] = {}
        for seed in SEEDS:
            print(f"  {cond:<14} seed={seed} ...", end=" ")
            d = compute_drift(cond, seed)
            if d is None:
                continue
            data[cond][seed] = d
            for layer in ["layer1", "layer2"]:
                f = d[layer]["frob"].mean()
                c = d[layer]["cosine"].mean()
                print(f"{layer}: frob={f:.4f} cos={c:.4f}  ", end="")
            print()
    return data


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise(data):
    """
    Returns: summary[condition][layer] = {
        frob_mean, frob_std, cosine_mean, cosine_std,
        per_seed: {seed: {frob_mean, cosine_mean}}
    }
    Means are computed per-seed first (mean over filters), then over seeds.
    """
    summary = {}
    for cond in data:
        summary[cond] = {}
        for layer in ["layer1", "layer2"]:
            per_seed_frob   = []
            per_seed_cosine = []
            seed_rows = {}
            for seed in SEEDS:
                if seed not in data[cond]:
                    continue
                f = data[cond][seed][layer]["frob"].mean()
                c = data[cond][seed][layer]["cosine"].mean()
                per_seed_frob.append(f)
                per_seed_cosine.append(c)
                seed_rows[seed] = {"frob": f, "cosine": c}

            summary[cond][layer] = {
                "frob_mean":   np.mean(per_seed_frob),
                "frob_std":    np.std(per_seed_frob),
                "cosine_mean": np.mean(per_seed_cosine),
                "cosine_std":  np.std(per_seed_cosine),
                "per_seed":    seed_rows,
            }
    return summary


# ── Print main table (C1 + C2 + C3) ─────────────────────────────────────────

def print_main_table(summary):
    """
    One row per condition.
    Columns: Frob(layer1), Frob(layer2), Cosine(layer1), Cosine(layer2)
    Injected layer is marked with * for easy spotting.
    """
    CONDITIONS_ORDER = ["Baseline", "Fourier-L1", "Fourier-L2", "KR-L1", "KR-L2"]
    INJ_LAYER = {c: CHECKPOINTS[c]["inject_layer"] for c in CHECKPOINTS}

    print("\n" + "="*120)
    print("  MAIN TABLE — Kernel Drift (mean ± std over 3 seeds)")
    print("  * = injected layer for that condition")
    print("  Frob drift: lower = more anchored | Cosine sim: higher = more anchored")
    print("="*120)

    hdr = (
        f"  {'Condition':<14} | {'Frob L1':^16} | {'Frob L2':^16} | "
        f"{'Cosine L1':^18} | {'Cosine L2':^18}"
    )
    print(hdr)
    print("  " + "-"*115)

    for cond in CONDITIONS_ORDER:
        if cond not in summary:
            continue
        inj = INJ_LAYER[cond]
        s = summary[cond]

        f1 = f"{s['layer1']['frob_mean']:.4f} ± {s['layer1']['frob_std']:.4f}"
        f2 = f"{s['layer2']['frob_mean']:.4f} ± {s['layer2']['frob_std']:.4f}"
        c1 = f"{s['layer1']['cosine_mean']:.6f} ± {s['layer1']['cosine_std']:.6f}"
        c2 = f"{s['layer2']['cosine_mean']:.6f} ± {s['layer2']['cosine_std']:.6f}"

        if inj == "layer1":
            f1 += " *"
            c1 += " *"
        elif inj == "layer2":
            f2 += " *"
            c2 += " *"

        print(f"  {cond:<14} | {f1:<18} | {f2:<18} | {c1:<20} | {c2:<20}")

    print()


# ── Print appendix per-seed table ────────────────────────────────────────────

def print_appendix_table(summary):
    CONDITIONS_ORDER = ["Baseline", "Fourier-L1", "Fourier-L2", "KR-L1", "KR-L2"]

    print("\n" + "="*85)
    print("  APPENDIX TABLE — Per-Seed Frob Drift and Cosine Similarity")
    print("="*85)
    hdr = (f"  {'Condition':<14} | {'Layer':<8} | "
           f"{'seed42 F':>8} {'seed42 C':>8} | "
           f"{'seed123 F':>9} {'seed123 C':>8} | "
           f"{'seed256 F':>9} {'seed256 C':>8}")
    print(hdr)
    print("  " + "-"*85)

    for cond in CONDITIONS_ORDER:
        if cond not in summary:
            continue
        for layer in ["layer1", "layer2"]:
            ps  = summary[cond][layer]["per_seed"]
            row = f"  {cond:<14} | {layer:<8} | "
            for seed in SEEDS:
                if seed in ps:
                    row += f"{ps[seed]['frob']:>8.4f} {ps[seed]['cosine']:>8.4f} | "
                else:
                    row += f"{'--':>8} {'--':>8} | "
            print(row)
        print()


# ── CSV exports ───────────────────────────────────────────────────────────────
def export_csvs(summary):
    CONDITIONS_ORDER = ["Baseline", "Fourier-L1", "Fourier-L2", "KR-L1", "KR-L2"]
    INJ_LAYER = {c: CHECKPOINTS[c]["inject_layer"] for c in CHECKPOINTS}

    # Main table CSV
    main_path = OUTPUT_DIR / "main_table_drift.csv"
    with open(main_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "inject_layer",
                    "frob_L1_mean", "frob_L1_std",
                    "frob_L2_mean", "frob_L2_std",
                    "cosine_L1_mean", "cosine_L1_std",
                    "cosine_L2_mean", "cosine_L2_std"])
        for cond in CONDITIONS_ORDER:
            if cond not in summary:
                continue
            inj = INJ_LAYER[cond]
            s   = summary[cond]
            
            w.writerow([
                cond, inj if inj else "none",
                f"{s['layer1']['frob_mean']:.6f}", f"{s['layer1']['frob_std']:.6f}",
                f"{s['layer2']['frob_mean']:.6f}", f"{s['layer2']['frob_std']:.6f}",
                f"{s['layer1']['cosine_mean']:.6f}", f"{s['layer1']['cosine_std']:.6f}",
                f"{s['layer2']['cosine_mean']:.6f}", f"{s['layer2']['cosine_std']:.6f}",
            ])
    print(f"  Saved: {main_path}")

# ── Win rate helper ───────────────────────────────────────────────────────────

def print_win_rates(summary):
    """
    For each injected condition: how many seeds does injected layer
    have lower Frob drift than baseline at same layer?
    """
    CONDITIONS_ORDER = ["Fourier-L1", "Fourier-L2", "KR-L1", "KR-L2"]
    INJ_LAYER = {c: CHECKPOINTS[c]["inject_layer"] for c in CHECKPOINTS}

    print("\n  WIN RATES vs Baseline (injected layer, lower Frob drift)")
    print("  " + "-"*50)
    for cond in CONDITIONS_ORDER:
        if cond not in summary:
            continue
        inj = INJ_LAYER[cond]
        wins = 0
        for seed in SEEDS:
            if seed not in summary[cond][inj]:
                continue
            if seed not in summary["Baseline"][inj]:
                continue
            if summary[cond][inj]["per_seed"][seed]["frob"] < \
               summary["Baseline"][inj]["per_seed"][seed]["frob"]:
                wins += 1
        print(f"  {cond:<14} {inj}:  {wins}/{len(SEEDS)} seeds beat baseline")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data    = collect_all()
    summary = summarise(data)

    print_main_table(summary)
    print_win_rates(summary)
    print_appendix_table(summary)
    export_csvs(summary)

    print(f"\nAll outputs → {OUTPUT_DIR}")