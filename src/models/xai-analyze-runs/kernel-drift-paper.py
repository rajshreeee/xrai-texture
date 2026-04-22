# ============================================================
#  KERNEL DRIFT — FOCUSED ANALYSIS
#  Two comparisons:
#    C1. Same layer, different init: injected-layer1 vs baseline-layer1
#    C2. Within-model: injected-layer1 vs uninjected-layer2 (same model)
#
#  Metrics per output filter:
#    - Relative Frobenius drift: ||W_final - W_init||_F / ||W_init||_F
#    - Cosine similarity: cos(vec(W_init), vec(W_final))
#
#  3 seeds → per-seed summaries + paired Wilcoxon across seeds
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import random
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats import median_abs_deviation

OUTPUT_DIR = Path("output/kernel_drift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 256]

RUN3 = Path("/ediss_data/ediss2/xai-texture/src/models/xai-model-run3-ft/checkpoints")
RUN4 = Path("/ediss_data/ediss2/xai-texture/src/models/xai-model-run4-kr/checkpoints")

# Each domain has one checkpoint per seed.
# TRAINED_INJECT_LAYER is the layer injected during that training run.
CHECKPOINTS = {
    "Baseline": {
        "inject_layer": None,
        "seeds": {
            42:  RUN3 / "A_baseline_seed42_best.pt",
            123: RUN3 / "A_baseline_seed123_best.pt",
            256: RUN3 / "A_baseline_seed256_best.pt",
        },
    },
    "Fourier": {
        "inject_layer": "layer1",
        "seeds": {
            42:  RUN3 / "D_layer1_init_seed42_best.pt",
            123: RUN3 / "D_layer1_init_seed123_best.pt",
            256: RUN3 / "D_layer1_init_seed256_best.pt",
        },
    },
    "Kernel Ranking": {
        "inject_layer": "layer1",
        "seeds": {
            42:  RUN4 / "D_layer1_init_seed42_best.pt",
            123: RUN4 / "KR_D_layer1_init_seed123_best.pt",
            256: RUN4 / "KR_D_layer1_init_seed256_best.pt",
        },
    },
}

fourier_kernels = [
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

kr_kernels = [
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
    "Fourier":        fourier_kernels,
    "Kernel Ranking": kr_kernels,
}

PALETTE = {
    "Baseline":       "#4e4e4e",
    "Fourier":        "#2166ac",
    "Kernel Ranking": "#d6604d",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


# ============================================================
# HELPERS
# ============================================================

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(seed, inject_layer=None, kernels_list=None):
    set_all_seeds(seed)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=3, classes=1)
    if inject_layer and kernels_list is not None:
        _inject(model, inject_layer, kernels_list)
    return model


def _inject(model, layer_name, kernels_list):
    target = getattr(model.encoder, layer_name)
    w = target[0].conv1.weight
    n_out, n_in, n_k = w.shape[0], w.shape[1], len(kernels_list)
    with torch.no_grad():
        for oi in range(n_out):
            k = torch.tensor(kernels_list[oi % n_k], dtype=torch.float32)
            for ii in range(n_in):
                w[oi, ii] = k


def get_filters(model, layer_name):
    target = getattr(model.encoder, layer_name)
    return target[0].conv1.weight.detach().cpu().numpy()


# ============================================================
# METRICS — per output filter
# ============================================================

def per_filter_frob_drift(W_init, W_final):
    """
    Relative Frobenius drift per output filter.
    ||W_final[oi] - W_init[oi]||_F / ||W_init[oi]||_F
    Shape: (n_out,)
    """
    out = []
    for oi in range(W_init.shape[0]):
        delta = np.linalg.norm(W_final[oi] - W_init[oi])
        norm0 = np.linalg.norm(W_init[oi]) + 1e-8
        out.append(delta / norm0)
    return np.array(out)


def per_filter_cosine(W_init, W_final):
    """
    Cosine similarity per output filter (mean over input channels).
    Shape: (n_out,)
    """
    out = []
    for oi in range(W_init.shape[0]):
        sims = []
        for ii in range(W_init.shape[1]):
            a = W_init[oi, ii].flatten()
            b = W_final[oi, ii].flatten()
            sims.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        out.append(float(np.mean(sims)))
    return np.array(out)


def layer_summary(arr):
    """Mean, median, IQR of a per-filter metric array."""
    q25, q75 = np.percentile(arr, [25, 75])
    return {"mean": arr.mean(), "median": np.median(arr),
            "iqr": q75 - q25, "std": arr.std()}


# ============================================================
# CORE: load one model, extract per-layer drift vs its own init
# ============================================================

def compute_model_drift(domain, seed):
    """
    Returns dict: layer_name → {frob: array(n_out), cosine: array(n_out)}
    Compares each layer's epoch-0 init against the trained checkpoint.
    init is reconstructed with the SAME injection used during training.
    """
    cfg       = CHECKPOINTS[domain]
    inj_layer = cfg["inject_layer"]
    ref_k     = DOMAIN_KERNELS.get(domain)
    ckpt_path = cfg["seeds"][seed]

    if not ckpt_path.exists():
        print(f"  [WARN] missing: {ckpt_path}")
        return None

    # Epoch-0 state — identical to what training started from
    m_init = build_model(seed, inject_layer=inj_layer, kernels_list=ref_k)

    # Trained state — load into same architecture
    m_trained = build_model(seed, inject_layer=inj_layer, kernels_list=ref_k)
    m_trained.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m_trained.eval()

    drift = {}
    for layer in ["layer1", "layer2"]:
        W_init  = get_filters(m_init,    layer)
        W_final = get_filters(m_trained, layer)

        # Sanity: confirm weights actually changed
        max_diff = np.abs(W_final - W_init).max()
        assert max_diff > 1e-6, \
            f"CRITICAL: no weight change detected — {domain}/{layer}/seed={seed}"

        drift[layer] = {
            "frob":   per_filter_frob_drift(W_init, W_final),
            "cosine": per_filter_cosine(W_init, W_final),
        }

    return drift


# ============================================================
# COLLECT ALL
# ============================================================

def collect():
    """
    Returns: data[domain][seed][layer] = {"frob": array, "cosine": array}
    """
    data = {d: {} for d in CHECKPOINTS}
    for domain in CHECKPOINTS:
        for seed in SEEDS:
            print(f"  {domain:<16} seed={seed} ...", end=" ")
            drift = compute_model_drift(domain, seed)
            if drift is None:
                continue
            data[domain][seed] = drift
            # Quick sanity print
            for layer in ["layer1", "layer2"]:
                f = drift[layer]["frob"]
                c = drift[layer]["cosine"]
                print(f"{layer} frob={f.mean():.4f} cos={c.mean():.4f}", end="  ")
            print()
    return data


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(data):
    print(f"\n{'='*80}")
    print("  DRIFT SUMMARY — mean (median) [IQR]  across filters, per seed")
    print(f"{'='*80}")
    hdr = f"  {'Domain':<16} {'Layer':<8} {'Seed':>5}  "
    hdr += f"{'Frob drift':>14}  {'Cosine sim':>12}"
    print(hdr)
    print(f"  {'-'*70}")

    for domain in ["Baseline", "Fourier", "Kernel Ranking"]:
        for layer in ["layer1", "layer2"]:
            for seed in SEEDS:
                if seed not in data[domain]:
                    continue
                d = data[domain][seed][layer]
                fs = layer_summary(d["frob"])
                cs = layer_summary(d["cosine"])
                print(f"  {domain:<16} {layer:<8} {seed:>5}  "
                      f"  {fs['mean']:.4f} ({fs['median']:.4f}) [{fs['iqr']:.4f}]"
                      f"  {cs['mean']:.4f} ({cs['median']:.4f}) [{cs['iqr']:.4f}]")
        print()


# ============================================================
# PAIRED TESTS + EFFECT SIZES  (across seeds, same-seed pairs)
# ============================================================

def paired_analysis(data):
    """
    C1: same layer, different init
        injected-layer1 vs baseline-layer1 (per seed → 3 paired means)
    C2: within-model contrast
        injected-layer1 vs injected-layer2 (per seed → 3 paired means)
    Uses Wilcoxon signed-rank on the 3 per-seed mean-drift values.
    Leads with rank-biserial correlation (effect size).
    """
    print(f"\n{'='*80}")
    print("  PAIRED ANALYSIS")
    print("  n=3 seeds → p-values are indicative only; lead with effect size (r_rb)")
    print(f"{'='*80}")

    def rank_biserial(x, y):
        """r_rb = 1 - 2*U / (n*m) where U from Mann-Whitney."""
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(x, y, alternative="two-sided")
        return 1 - 2 * u / (len(x) * len(y))

    def paired_summary(label, a_vals, b_vals, a_name, b_name):
        """a_vals, b_vals: list of per-seed mean drifts (length 3)."""
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        print(f"\n  {label}")
        print(f"    {a_name}: {[f'{v:.4f}' for v in a_vals]}  mean={np.mean(a_vals):.4f}")
        print(f"    {b_name}: {[f'{v:.4f}' for v in b_vals]}  mean={np.mean(b_vals):.4f}")
        print(f"    Δ (a−b):  {[f'{d:.4f}' for d in diffs]}  mean={np.mean(diffs):.4f}")
        r = rank_biserial(a_vals, b_vals)
        print(f"    Effect size r_rb = {r:.3f}  "
              f"({'large' if abs(r)>0.5 else 'medium' if abs(r)>0.3 else 'small'})")
        if len(set(diffs)) > 1:
            try:
                stat, p = wilcoxon(diffs)
                print(f"    Wilcoxon p = {p:.4f}  (n=3, treat as descriptive)")
            except Exception:
                print("    Wilcoxon: insufficient variation")
        else:
            print("    Wilcoxon: all differences identical — constant effect")

    for metric, label in [("frob", "Relative Frobenius Drift"),
                           ("cosine", "Cosine Similarity (higher = less drift)")]:
        print(f"\n  ── Metric: {label} ──")

        for domain in ["Fourier", "Kernel Ranking"]:
            # C1: injected-layer1 vs baseline-layer1
            inj_l1   = [data[domain][s]["layer1"][metric].mean() for s in SEEDS
                        if s in data[domain]]
            base_l1  = [data["Baseline"][s]["layer1"][metric].mean() for s in SEEDS
                        if s in data["Baseline"]]
            paired_summary(
                f"C1 [{domain}] injected-layer1 vs baseline-layer1",
                inj_l1, base_l1,
                f"{domain}/layer1", "Baseline/layer1"
            )

            # C2: within-model — injected-layer1 vs uninjected-layer2 (same model)
            inj_l1_  = [data[domain][s]["layer1"][metric].mean() for s in SEEDS
                        if s in data[domain]]
            inj_l2_  = [data[domain][s]["layer2"][metric].mean() for s in SEEDS
                        if s in data[domain]]
            paired_summary(
                f"C2 [{domain}] within-model: layer1 (injected) vs layer2 (not injected)",
                inj_l1_, inj_l2_,
                f"{domain}/layer1", f"{domain}/layer2"
            )


# ============================================================
# PLOTS
# ============================================================

def plot_drift_comparison(data):
    """
    2×2 grid:
      Row 1: Frob drift    — C1 (layer1: baseline vs injected)
                           — C2 (within-model: layer1 vs layer2)
      Row 2: Cosine sim    — same structure
    Each panel: strip plot (per-seed points) + mean bar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Kernel Drift Analysis\n"
                 "C1: same layer, different init | C2: within-model layer contrast",
                 fontweight="bold")

    metrics   = [("frob",   "Relative Frob. Drift  ‖ΔW‖/‖W₀‖  (lower = more anchored)"),
                 ("cosine", "Cosine Similarity cos(W₀, W_final)  (higher = more anchored)")]
    panels    = [("C1: Same layer, different init\n(layer1 comparison)",
                  "C2: Within-model layer contrast\n(injected model only)")]

    colors_c1 = {
        "Baseline":       PALETTE["Baseline"],
        "Fourier":        PALETTE["Fourier"],
        "Kernel Ranking": PALETTE["Kernel Ranking"],
    }

    for row, (metric, ylabel) in enumerate(metrics):
        ax_c1, ax_c2 = axes[row]

        # ── C1: layer1 across conditions ─────────────────────
        conditions_c1 = ["Baseline", "Fourier", "Kernel Ranking"]
        for xi, cond in enumerate(conditions_c1):
            vals = [data[cond][s]["layer1"][metric].mean()
                    for s in SEEDS if s in data[cond]]
            m    = np.mean(vals)
            ax_c1.bar(xi, m, color=colors_c1[cond], alpha=0.65, width=0.5,
                      zorder=1)
            ax_c1.scatter([xi] * len(vals), vals,
                          color=colors_c1[cond], s=60, zorder=3,
                          edgecolors="white", linewidths=0.8)
            ax_c1.text(xi, m + 0.001, f"{m:.4f}",
                       ha="center", va="bottom", fontsize=8)

        ax_c1.set_xticks(range(len(conditions_c1)))
        ax_c1.set_xticklabels(conditions_c1, rotation=10)
        ax_c1.set_ylabel(ylabel, fontsize=8.5)
        ax_c1.set_title("C1 · Layer1: baseline vs injected", fontsize=10)

        # ── C2: layer1 vs layer2 within each injected model ──
        x_pos = 0
        xticks, xlabels = [], []
        for domain in ["Fourier", "Kernel Ranking"]:
            for li, layer in enumerate(["layer1", "layer2"]):
                vals  = [data[domain][s][layer][metric].mean()
                         for s in SEEDS if s in data[domain]]
                m     = np.mean(vals)
                alpha = 0.85 if layer == "layer1" else 0.45
                hatch = None if layer == "layer1" else "//"
                ax_c2.bar(x_pos, m, color=PALETTE[domain], alpha=alpha,
                          width=0.5, hatch=hatch, zorder=1)
                ax_c2.scatter([x_pos] * len(vals), vals,
                              color=PALETTE[domain], s=60, zorder=3,
                              edgecolors="white", linewidths=0.8)
                ax_c2.text(x_pos, m + 0.001, f"{m:.4f}",
                           ha="center", va="bottom", fontsize=8)
                xticks.append(x_pos)
                xlabels.append(f"{domain.replace('Kernel Ranking','KR')}\n{layer}")
                x_pos += 1
            x_pos += 0.4  # gap between domains

        ax_c2.set_xticks(xticks)
        ax_c2.set_xticklabels(xlabels, fontsize=8)
        ax_c2.set_title("C2 · Within-model: injected layer1 vs uninjected layer2",
                        fontsize=10)

        # Shade legend (solid=injected layer, hatched=non-injected)
        from matplotlib.patches import Patch
        ax_c2.legend(handles=[
            Patch(facecolor="gray", alpha=0.85, label="layer1 (injected)"),
            Patch(facecolor="gray", alpha=0.45, hatch="//", label="layer2 (not injected)"),
        ], fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drift_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: drift_comparison.png")


def plot_filter_distributions(data):
    """
    Violin/box plot of per-filter Frob drift distributions, pooled across seeds.
    C1 (layer1) and C2 (within-model layer1 vs layer2) side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-Filter Drift Distributions (pooled across 3 seeds)\n"
                 "Each point = one output filter", fontweight="bold")

    # C1
    ax = axes[0]
    ax.set_title("C1 · Layer1 drift by condition")
    positions, all_vals, colors, labels = [], [], [], []
    for xi, cond in enumerate(["Baseline", "Fourier", "Kernel Ranking"]):
        vals = np.concatenate([data[cond][s]["layer1"]["frob"]
                               for s in SEEDS if s in data[cond]])
        positions.append(xi)
        all_vals.append(vals)
        colors.append(PALETTE[cond])
        labels.append(cond)

    vp = ax.violinplot(all_vals, positions=positions, showmedians=True,
                       showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Relative Frob. Drift ‖ΔW‖/‖W₀‖")

    # C2
    ax = axes[1]
    ax.set_title("C2 · Within-model: layer1 vs layer2")
    positions, all_vals, colors, labels = [], [], [], []
    xi = 0
    for domain in ["Fourier", "Kernel Ranking"]:
        for layer in ["layer1", "layer2"]:
            vals = np.concatenate([data[domain][s][layer]["frob"]
                                   for s in SEEDS if s in data[domain]])
            positions.append(xi)
            all_vals.append(vals)
            colors.append(PALETTE[domain])
            short = domain.replace("Kernel Ranking", "KR")
            labels.append(f"{short}\n{layer}")
            xi += 1
        xi += 0.3

    vp = ax.violinplot(all_vals, positions=positions, showmedians=True,
                       showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Relative Frob. Drift ‖ΔW‖/‖W₀‖")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drift_distributions.png", bbox_inches="tight")
    plt.close()
    print("  Saved: drift_distributions.png")


# ============================================================
# MAIN
# ============================================================

def run():
    print("\n" + "="*60)
    print("  KERNEL DRIFT — FOCUSED ANALYSIS (3 seeds)")
    print("="*60)

    print("\n[1] Loading models and computing drift...")
    data = collect()

    print("\n[2] Per-seed summary table...")
    print_table(data)

    print("\n[3] Paired analysis (C1 + C2)...")
    paired_analysis(data)

    print("\n[4] Plots...")
    plot_drift_comparison(data)
    plot_filter_distributions(data)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    run()