"""
kernel_drift_analysis.py

Two complementary drift comparisons:
  1. Same layer, different init  — injected vs baseline (main claim)
  2. Within-model contrast       — injected layer vs non-injected layers
                                   in the same model (rules out "model just
                                   trains less aggressively")

Metrics (scale-invariant):
  - Relative Frobenius drift : ‖W_final - W_init‖_F / ‖W_init‖_F
  - Cosine similarity        : cos(vec(W_init), vec(W_final))
  Both computed per output filter, then summarised over the layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pandas as pd
from scipy.stats import wilcoxon

import config
from unet import build_model, DEVICE

# ---------------------------------------------------------------------------
# Experiment registry  (group_name → inject_layer)
# Add / comment out conditions as your runs complete
# ---------------------------------------------------------------------------
CONDITIONS = {
    "A_baseline":       None,
    "KR_D_layer1_init": "layer1",
    "KR_B_layer2_init": "layer2",
}

SEEDS  = [42, 123, 256]
LAYERS = ["layer1", "layer2", "layer3", "layer4"]
CKPT   = Path(config.CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conv1_weight(model, layer_name):
    """(C_out, C_in, 3, 3) float32 numpy array for encoder.<layer>[0].conv1."""
    return getattr(model.encoder, layer_name)[0].conv1.weight.detach().cpu().numpy()


def relative_frobenius(w_init: np.ndarray, w_final: np.ndarray) -> np.ndarray:
    """Per output-filter relative Frobenius drift. Returns shape (C_out,)."""
    diff  = (w_final - w_init).reshape(w_final.shape[0], -1)
    denom = w_init.reshape(w_init.shape[0], -1)
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(denom, axis=1) + 1e-8)


def cosine_sim(w_init: np.ndarray, w_final: np.ndarray) -> np.ndarray:
    """Per output-filter cosine similarity. Returns shape (C_out,)."""
    wi = w_init.reshape(w_init.shape[0], -1)
    wf = w_final.reshape(w_final.shape[0], -1)
    num   = (wi * wf).sum(axis=1)
    denom = np.linalg.norm(wi, axis=1) * np.linalg.norm(wf, axis=1) + 1e-8
    return num / denom


def summarise(arr: np.ndarray) -> dict:
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "iqr":    float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


# ---------------------------------------------------------------------------
# Collect per-(condition, seed, layer) stats
# ---------------------------------------------------------------------------

def collect() -> pd.DataFrame:
    rows = []

    for cond, inject_layer in CONDITIONS.items():
        for seed in SEEDS:
            ckpt = CKPT / f"{cond}_seed{seed}_best.pt"
            if not ckpt.exists():
                print(f"  [skip] {ckpt.name}")
                continue

            # Reconstruct W_init deterministically from the same seed + injection
            m_init  = build_model(seed, inject_layer=inject_layer)

            # Load W_final
            m_final = build_model(seed, inject_layer=inject_layer)
            m_final.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m_final.eval()

            print(f"  [ok]   {ckpt.name}")

            for layer in LAYERS:
                try:
                    wi = get_conv1_weight(m_init,  layer)
                    wf = get_conv1_weight(m_final, layer)
                except AttributeError:
                    continue

                frob = relative_frobenius(wi, wf)
                cos  = cosine_sim(wi, wf)

                rows.append({
                    "condition":    cond,
                    "inject_layer": str(inject_layer),
                    "seed":         seed,
                    "layer":        layer,
                    "is_injected":  inject_layer == layer,
                    **{f"frob_{k}": v for k, v in summarise(frob).items()},
                    **{f"cos_{k}":  v for k, v in summarise(cos).items()},
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 1 — layer summary (mean ± std across seeds)
# ---------------------------------------------------------------------------

def table_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["condition", "layer"])[["frob_mean", "cos_mean"]]
        .agg(["mean", "std"])
        .round(4)
    )


# ---------------------------------------------------------------------------
# Table 2 — within-model contrast
# Pivot: for each injected model × seed, show frob_mean per layer
# so you can read off: is the injected layer the lowest-drift one?
# ---------------------------------------------------------------------------

def table_within_model(df: pd.DataFrame) -> pd.DataFrame:
    injected = df[df["inject_layer"] != "None"].copy()
    pivot = injected.pivot_table(
        index=["condition", "seed"],
        columns="layer",
        values="frob_mean",
    ).round(4)
    # Mark injected layer with asterisk in column header
    rename = {}
    for cond, inj in CONDITIONS.items():
        if inj and inj in pivot.columns:
            rename[inj] = f"{inj}*"
    return pivot.rename(columns=rename)


# ---------------------------------------------------------------------------
# Table 3 — paired Wilcoxon (injected vs baseline, per layer)
# With only 3 seeds this is descriptive; lead with effect size (rank-biserial r)
# ---------------------------------------------------------------------------

def table_paired_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "A_baseline" not in df["condition"].values:
        print("\n[info] No baseline checkpoints found — skipping paired stats.")
        return pd.DataFrame()

    rows = []
    for layer in LAYERS:
        for cond, inj in CONDITIONS.items():
            if inj is None:
                continue
            base = (df[(df["condition"] == "A_baseline") & (df["layer"] == layer)]
                    .sort_values("seed")["frob_mean"].values)
            inj_ = (df[(df["condition"] == cond) & (df["layer"] == layer)]
                    .sort_values("seed")["frob_mean"].values)
            n = min(len(base), len(inj_))
            if n < 2:
                continue
            diffs = inj_[:n] - base[:n]
            # Wilcoxon needs non-zero diffs
            if (diffs == 0).all():
                continue
            stat, p = wilcoxon(diffs)
            r_rb = 1 - (2 * stat) / (n * (n + 1))   # rank-biserial correlation
            rows.append({
                "condition": cond,
                "layer":     layer,
                "mean_delta_frob": diffs.mean().round(4),
                "r_rb":      round(r_rb, 3),
                "p_wilcoxon": round(p, 4),
                "n_seeds":   n,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Kernel Drift Analysis ===\n")
    print("Loading checkpoints...")
    df = collect()

    if df.empty:
        print("\nNo checkpoints loaded. Check CHECKPOINT_DIR and condition names.")
        return

    print("\n--- Table 1: Layer-level relative Frobenius drift & cosine sim (mean ± std across seeds) ---")
    t1 = table_layer_summary(df)
    print(t1.to_string())

    print("\n--- Table 2: Within-model contrast — rel. Frobenius per layer (* = injected layer) ---")
    t2 = table_within_model(df)
    print(t2.to_string())

    print("\n--- Table 3: Paired Wilcoxon injected vs baseline (effect size + p-value) ---")
    t3 = table_paired_stats(df)
    if not t3.empty:
        print(t3.to_string(index=False))

    # Save
    out = Path(__file__).parent / "kernel_drift_results.csv"
    df.to_csv(out, index=False)
    print(f"\nFull per-(condition, seed, layer) results → {out}")


if __name__ == "__main__":
    main()
