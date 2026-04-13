# ============================================================
#  KERNEL DRIFT & RETENTION ANALYSIS
#  Baseline | Fourier | Kernel Ranking  — seed=42
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import segmentation_models_pytorch as smp
import random
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CONFIGURE ────────────────────────────────────────────────
OUTPUT_DIR = Path("output/kernel_drift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED          = 42
INJECT_LAYERS = ["layer1", "layer2"]

# Full absolute paths to checkpoints
CHECKPOINTS = {
    "Baseline": Path("/ediss_data/ediss2/xai-texture/src/models/xai-model-run3-ft/checkpoints/A_baseline_seed42_best.pt"),
    "Fourier":  Path("/ediss_data/ediss2/xai-texture/src/models/xai-model-run3-ft/checkpoints/B_layer2_init_seed42_best.pt"),
    "Kernel Ranking": Path("/ediss_data/ediss2/xai-texture/src/models/xai-model-run4-kr/checkpoints/B_layer2_init_seed42_best.pt"),
}

# Injection layer used during training per domain
# (needed to exactly reconstruct epoch-0 state)
TRAINED_INJECT_LAYER = {
    "Baseline":       None,
    "Fourier":        "layer2",   # B_layer2 → injected into layer2
    "Kernel Ranking": "layer2",   # B_layer2 → injected into layer2
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

# Reference kernels used per domain (for texture-alignment metric)
DOMAIN_KERNELS = {
    "Fourier":        fourier_kernels,
    "Kernel Ranking": kr_kernels,
}

PALETTE = {
    "Baseline":       "#4e4e4e",
    "Fourier":        "#2166ac",
    "Kernel Ranking": "#d6604d",
}

# ============================================================
# HELPERS
# ============================================================

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_fresh_model(seed, inject_layer=None, kernels_list=None):
    set_all_seeds(seed)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    if inject_layer is not None and kernels_list is not None:
        _inject(model, inject_layer, kernels_list)
    return model


def _inject(model, layer_name, kernels_list):
    target      = getattr(model.encoder, layer_name)
    conv_weight = target[0].conv1.weight
    n_out, n_in = conv_weight.shape[0], conv_weight.shape[1]
    n_k         = len(kernels_list)
    with torch.no_grad():
        for oi in range(n_out):
            k = torch.tensor(kernels_list[oi % n_k], dtype=torch.float32)
            for ii in range(n_in):
                conv_weight[oi, ii] = k


def get_filters(model, layer_name):
    target = getattr(model.encoder, layer_name)
    return target[0].conv1.weight.detach().cpu().numpy()


# ============================================================
# METRICS
# ============================================================

def cosine_sim(a, b):
    a, b  = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-8))


def direction_displacement(f_init, f_trained):
    """1 − cosine_sim per output filter (mean over input channels). 0=unchanged."""
    disps = []
    for oi in range(f_init.shape[0]):
        sims = [cosine_sim(f_init[oi, ii], f_trained[oi, ii])
                for ii in range(f_init.shape[1])]
        disps.append(1.0 - float(np.mean(sims)))
    return np.array(disps)


def magnitude_change(f_init, f_trained):
    """(||trained|| − ||init||) / ||init||  per output filter."""
    changes = []
    for oi in range(f_init.shape[0]):
        ni = np.linalg.norm(f_init[oi])
        nt = np.linalg.norm(f_trained[oi])
        changes.append((nt - ni) / (ni + 1e-8))
    return np.array(changes)


def max_cosine_sim_to_references(f_trained, ref_kernels):
    """
    For each trained output filter: max cosine sim to any reference kernel
    (mean over input channels). Higher = more texture-operator-like.
    """
    sims = []
    for oi in range(f_trained.shape[0]):
        best = max(
            float(np.mean([cosine_sim(f_trained[oi, ii], ref)
                           for ii in range(f_trained.shape[1])]))
            for ref in ref_kernels
        )
        sims.append(best)
    return np.array(sims)


# ============================================================
# PRINT TABLE
# ============================================================

def print_drift_table(results):
    DIV = "=" * 78
    print(f"\n{DIV}")
    print("  KERNEL DRIFT ANALYSIS — seed=42")
    print("  Displacement = 1 - cosine_sim(init_filter, trained_filter)")
    print("  Lower displacement = filter stayed closer to initialization")
    print(DIV)
    print(f"  {'Domain':<20} {'Layer':<10} "
          f"{'Disp mean':>12} {'Disp std':>10} "
          f"{'Mag Δ mean':>12} {'Ref Sim mean':>14}")
    print(f"  {'-'*78}")
    for (domain, layer), m in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  {domain:<20} {layer:<10} "
              f"{m['disp_mean']:>12.4f} {m['disp_std']:>10.4f} "
              f"{m['mag_mean']:>+12.4f} {m['ref_sim_mean']:>14.4f}")
    print()


# ============================================================
# PLOTS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def plot_displacement_bars(results):
    domains = ["Baseline", "Fourier", "Kernel Ranking"]
    layers  = ["layer1", "layer2"]
    x, w    = np.arange(len(domains)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Filter Displacement from Initialization (seed=42)\n"
                 "Lower displacement = init structure better preserved post-training",
                 fontsize=12, fontweight="bold")

    specs = [
        ("disp_mean",    "disp_std",    "Directional Displacement\n(1 − cosine sim)",
         "Angular Displacement from Init"),
        ("ref_sim_mean", "ref_sim_std", "Max Cosine Sim to Ref Kernels",
         "Texture-Kernel Alignment of Trained Filters"),
    ]

    for ax, (m_key, s_key, ylabel, title) in zip(axes, specs):
        for li, layer in enumerate(layers):
            vals   = [results.get((d, layer), {}).get(m_key, float("nan")) for d in domains]
            errs   = [results.get((d, layer), {}).get(s_key, 0.0)          for d in domains]
            offset = (li - 0.5) * w
            bars   = ax.bar(
                x + offset, vals, w,
                yerr=errs, capsize=4,
                color=[PALETTE.get(d, "#999") for d in domains],
                alpha=0.88 if layer == "layer1" else 0.52,
                edgecolor="white",
                error_kw=dict(elinewidth=1.2, ecolor="#555"),
                label=layer,
            )
            for xi, v in zip(x + offset, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.004, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=12, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, title="Layer", title_fontsize=7)

    plt.tight_layout()
    fname = OUTPUT_DIR / "displacement_bars.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_filter_grids(filter_dict):
    domains = ["Baseline", "Fourier", "Kernel Ranking"]
    layers  = ["layer1", "layer2"]
    n_show  = 16   # 4×4 grid

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Learned Filter Visualization (seed=42, post-training)\n"
        "Structured init → oriented/banded patterns | Random init → noise-like",
        fontsize=12, fontweight="bold", y=1.01,
    )

    outer = gridspec.GridSpec(len(domains), len(layers), hspace=0.5, wspace=0.15)

    for ri, domain in enumerate(domains):
        for ci, layer in enumerate(layers):
            filters = filter_dict.get((domain, layer))
            if filters is None:
                ax = fig.add_subplot(outer[ri, ci])
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                continue

            inner = gridspec.GridSpecFromSubplotSpec(
                4, 4, subplot_spec=outer[ri, ci], hspace=0.05, wspace=0.05
            )
            for fi in range(min(n_show, filters.shape[0])):
                ax   = fig.add_subplot(inner[fi // 4, fi % 4])
                f    = filters[fi, 0]
                vmax = np.abs(f).max() + 1e-8
                ax.imshow(f, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
                ax.axis("off")

            # Panel title placed on the outer subplot
            title_ax = fig.add_subplot(outer[ri, ci])
            title_ax.set_title(
                f"{domain} / {layer}", fontsize=9, fontweight="bold",
                color=PALETTE.get(domain, "#333"), pad=6,
            )
            title_ax.axis("off")

    fname = OUTPUT_DIR / "filter_grids.png"
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {fname}")


def plot_displacement_distributions(disp_dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    fig.suptitle(
        "Per-Filter Directional Displacement Distribution (seed=42)\n"
        "Left of baseline = init structure retained | dashed = group mean",
        fontsize=12, fontweight="bold",
    )

    for ci, layer in enumerate(["layer1", "layer2"]):
        ax = axes[ci]
        for domain in ["Baseline", "Fourier", "Kernel Ranking"]:
            disps = disp_dict.get((domain, layer))
            if disps is None:
                continue
            ax.hist(disps, bins=20, alpha=0.55,
                    color=PALETTE[domain],
                    label=f"{domain}  μ={disps.mean():.3f}",
                    edgecolor="white", linewidth=0.5)
            ax.axvline(disps.mean(), color=PALETTE[domain], lw=2, ls="--", alpha=0.9)

        ax.set_title(f"{layer}")
        ax.set_xlabel("Directional Displacement (1 − cosine sim to init)")
        ax.set_ylabel("Filter count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = OUTPUT_DIR / "displacement_distributions.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_magnitude_change(results):
    """
    Separate bar chart for magnitude change — signed, so needs its own scale.
    Positive = filter grew after training, negative = shrank.
    """
    domains = ["Baseline", "Fourier", "Kernel Ranking"]
    layers  = ["layer1", "layer2"]
    x, w    = np.arange(len(domains)), 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Filter Magnitude Change from Initialization (seed=42)\n"
                 "Relative change: (||trained|| − ||init||) / ||init||",
                 fontsize=12, fontweight="bold")

    for li, layer in enumerate(layers):
        vals = [results.get((d, layer), {}).get("mag_mean", float("nan")) for d in domains]
        errs = [results.get((d, layer), {}).get("mag_std",  0.0)          for d in domains]
        ax.bar(x + (li - 0.5) * w, vals, w,
               yerr=errs, capsize=4,
               color=[PALETTE.get(d, "#999") for d in domains],
               alpha=0.88 if layer == "layer1" else 0.52,
               edgecolor="white",
               error_kw=dict(elinewidth=1.2, ecolor="#555"),
               label=layer)
        for xi, v in zip(x + (li - 0.5) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + (0.01 if v >= 0 else -0.04),
                        f"{v:+.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=12, ha="right")
    ax.set_ylabel("Relative Magnitude Change")
    ax.legend(fontsize=8, title="Layer", title_fontsize=7)

    plt.tight_layout()
    fname = OUTPUT_DIR / "magnitude_change.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# MAIN
# ============================================================

def run():
    print("\n" + "="*60)
    print("  KERNEL DRIFT ANALYSIS — seed=42")
    print("="*60)

    # ── 1. Reconstruct epoch-0 initializations ───────────────
    print("\n[1] Reconstructing epoch-0 initializations...")
    init_filters = {}

    for layer in INJECT_LAYERS:
        # Baseline: random, no injection
        m = build_fresh_model(SEED)
        init_filters[("Baseline", layer)] = get_filters(m, layer)
        print(f"  Baseline init [{layer}] shape={init_filters[('Baseline', layer)].shape}")

        # Injection domains
        for domain, ref_k in DOMAIN_KERNELS.items():
            m = build_fresh_model(SEED, inject_layer=layer, kernels_list=ref_k)
            init_filters[(domain, layer)] = get_filters(m, layer)
            print(f"  {domain} init [{layer}] shape={init_filters[(domain, layer)].shape}")

    # ── 2. Load trained checkpoints ──────────────────────────
    print("\n[2] Loading trained checkpoints...")
    trained_filters = {}

    for domain, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"  [WARN] Not found: {ckpt_path} — skipping {domain}")
            continue

        # Rebuild model with the SAME injection used during training
        # so that load_state_dict matches exactly (no key mismatch)
        inj_layer = TRAINED_INJECT_LAYER[domain]
        ref_k     = DOMAIN_KERNELS.get(domain)   # None for Baseline
        m         = build_fresh_model(SEED, inject_layer=inj_layer, kernels_list=ref_k)

        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()

        for layer in INJECT_LAYERS:
            trained_filters[(domain, layer)] = get_filters(m, layer)
            print(f"  {domain} trained [{layer}] shape={trained_filters[(domain, layer)].shape}")

    # ── 3. Compute metrics ────────────────────────────────────
    print("\n[3] Computing displacement metrics...")
    results   = {}
    disp_dict = {}

    for domain in CHECKPOINTS:
        ref_k = DOMAIN_KERNELS.get(domain)   # None for Baseline
        for layer in INJECT_LAYERS:
            key = (domain, layer)
            if key not in trained_filters or key not in init_filters:
                print(f"  [SKIP] {domain}/{layer} — missing data")
                continue

            f_i = init_filters[key]
            f_t = trained_filters[key]

            disp = direction_displacement(f_i, f_t)
            mag  = magnitude_change(f_i, f_t)
            disp_dict[key] = disp

            # Ref sim: only meaningful for injection domains
            if ref_k is not None:
                ref_sim = max_cosine_sim_to_references(f_t, ref_k)
                rsm, rss = ref_sim.mean(), ref_sim.std()
            else:
                rsm, rss = float("nan"), float("nan")

            results[key] = {
                "disp_mean": disp.mean(), "disp_std": disp.std(),
                "mag_mean":  mag.mean(),  "mag_std":  mag.std(),
                "ref_sim_mean": rsm,      "ref_sim_std": rss,
            }

    # ── 4. Print table ────────────────────────────────────────
    print_drift_table(results)

    print("  INTERPRETATION GUIDE:")
    print("  Disp lower in injection vs baseline → retention: filters resisted reorientation")
    print("  Disp similar in injection vs baseline → head-start: moved same amount from better origin")
    print("  Ref Sim higher in injection vs baseline → trained filters stayed texture-like\n")

    # ── 5. Generate plots ─────────────────────────────────────
    print("[4] Generating plots...")
    plot_displacement_bars(results)
    plot_magnitude_change(results)
    plot_filter_grids(trained_filters)
    plot_displacement_distributions(disp_dict)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
