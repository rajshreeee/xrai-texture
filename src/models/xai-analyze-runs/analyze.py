
# ============================================================
#  MULTI-DOMAIN KERNEL INJECTION ANALYSIS
#  Domains: Baseline | Fourier | Kernel Ranking | Laws (LTEM)
#  Layers:  layer1, layer2
#  Seeds:   42, 123, 256
#  Metrics: Train IoU, Test IoU
#  Reports: ECS Summary | E2T per-seed tables | Win Rate | Plots
# ============================================================

import re
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("output/injection_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SECTION 1 — CONFIGURATION
# ============================================================

LOG_ROOT = Path("/ediss_data/ediss2/xai-texture/src/models")

DOMAIN_FOLDERS = {
    "xai-model-run-baseline/logs": "Baseline",
    "xai-model-run3-ft/logs":      "Fourier",
    "xai-model-run4-kr/logs":      "Kernel Ranking",
    "xai-model-run6-ltem/logs":  "Laws (LTEM)",
}

LAYER_PATTERN = re.compile(r"layer(\d+)")
SEED_PATTERN  = re.compile(r"seed(\d+)")

SEEDS = [42, 123, 256]

AUC_EPOCH_START = 1
AUC_EPOCH_END   = 30

E2T_THRESHOLDS = [0.55, 0.60, 0.65]

# ── Palette: colorblind-safe, publication-grade ──────────────
PALETTE = {
    "Baseline":       "#4e4e4e",   # dark grey
    "Fourier":        "#2166ac",   # strong blue
    "Kernel Ranking": "#d6604d",   # warm red
    "Laws (LTEM)":    "#35978f",   # teal
}

LAYER_LS     = {"layer1": "-",  "layer2": "--", "none": "-"}
LAYER_MARKER = {"layer1": "o",  "layer2": "s",  "none": "o"}

# ── Global rcParams ──────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.4,
    "legend.fontsize":     9,
    "legend.frameon":      True,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "figure.facecolor":    "white",
    "axes.facecolor":      "#f9f9f9",
})

# ============================================================
# SECTION 2 — LOG PARSER
# ============================================================

def parse_log(log_path: Path) -> dict:
    ep_pat = re.compile(
        r"Epoch (\d+)/\d+ \|.*?"
        r"train_dice=[\d.]+ +\| train_iou=([\d.]+).*?"
        r"test_loss=[\d.]+ \| test_dice=[\d.]+ \| test_iou=([\d.]+)"
    )
    data = dict(epochs=[], train_iou=[], test_iou=[])
    with open(log_path) as f:
        for line in f:
            m = ep_pat.search(line)
            if m:
                data["epochs"].append(int(m.group(1)))
                data["train_iou"].append(float(m.group(2)))
                data["test_iou"].append(float(m.group(3)))
    return data

# ============================================================
# SECTION 3 — LOG LOADER
# ============================================================

def load_all(log_root: Path) -> dict:
    """results[domain][layer][seed] = {epochs, train_iou, test_iou}"""
    results = {}
    for folder, domain_label in DOMAIN_FOLDERS.items():
        folder_path = log_root / folder
        if not folder_path.exists():
            print(f"  [WARN] Not found: {folder_path}")
            continue
        results[domain_label] = {}
        for log_file in sorted(folder_path.glob("*.log")):
            seed_m  = SEED_PATTERN.search(log_file.name)
            layer_m = LAYER_PATTERN.search(log_file.name)
            if not seed_m:
                print(f"  [SKIP] No seed: {log_file.name}")
                continue
            seed  = int(seed_m.group(1))
            layer = f"layer{layer_m.group(1)}" if layer_m else "none"
            parsed = parse_log(log_file)
            if not parsed["epochs"]:
                print(f"  [WARN] No data: {log_file.name}")
                continue
            results[domain_label].setdefault(layer, {})
            results[domain_label][layer][seed] = parsed
            print(f"  Loaded [{domain_label}][{layer}][seed={seed}] "
                  f"-> {len(parsed['epochs'])} epochs")
    return results

# ============================================================
# SECTION 4 — METRIC HELPERS
# ============================================================

def ecs(series: list, start: int = AUC_EPOCH_START, end: int = AUC_EPOCH_END) -> float:
    window = series[start - 1 : end]
    return float(np.mean(window)) if window else float("nan")

def peak_delta(series_method: list, series_baseline: list,
               start: int = AUC_EPOCH_START, end: int = AUC_EPOCH_END) -> float:
    wm     = series_method[start - 1 : end]
    wb     = series_baseline[start - 1 : end]
    deltas = [m - b for m, b in zip(wm, wb)]
    return float(max(deltas)) if deltas else float("nan")

def epoch_to_threshold(series: list, threshold: float) -> Optional[int]:
    for i, v in enumerate(series):
        if v >= threshold:
            return i + 1
    return None

def _agg(d: dict) -> dict:
    vals = [v for v in d.values() if v is not None and not np.isnan(float(v))]
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "std":  float(np.std(vals))  if vals else float("nan"),
            **d}

def compute_metrics(results: dict, metric: str) -> dict:
    metrics = {}
    for domain, layers in results.items():
        metrics[domain] = {}
        for layer, seeds in layers.items():
            ecs_vals = {}
            e2t_vals = {t: {} for t in E2T_THRESHOLDS}
            curves   = {}
            for seed, data in seeds.items():
                series         = data[metric]
                curves[seed]   = np.array(series)
                ecs_vals[seed] = ecs(series)
                for t in E2T_THRESHOLDS:
                    e2t_vals[t][seed] = epoch_to_threshold(series, t)
            metrics[domain][layer] = {
                "ecs":    _agg(ecs_vals),
                "e2t":    {t: _agg(e2t_vals[t]) for t in E2T_THRESHOLDS},
                "curves": curves,
            }
    return metrics

def _bl_layer(metrics: dict, baseline: str = "Baseline") -> Optional[str]:
    layers = metrics.get(baseline, {})
    if "none" in layers:
        return "none"
    return list(layers.keys())[0] if layers else None

# ============================================================
# SECTION 5 — CONSOLE TABLES
# ============================================================

DIV  = "=" * 80
DIV2 = "-" * 80

def print_ecs_summary(metrics: dict, metric_label: str):
    bl_l      = _bl_layer(metrics)
    bl_curves = metrics["Baseline"][bl_l]["curves"]

    print(f"\n{DIV}")
    print(f"  ECS SUMMARY — {metric_label}  "
          f"(window ep {AUC_EPOCH_START}-{AUC_EPOCH_END} | seeds {SEEDS})")
    print(DIV)
    print(f"  {'Domain':<20} {'Layer':<10} "
          f"{'ECS mean+/-std':>16} {'dECS':>8} {'Peak d mean+/-std':>18} {'Win Rate':>10}")
    print(f"  {DIV2}")

    for domain in metrics:
        for layer in sorted(metrics[domain]):
            m         = metrics[domain][layer]
            e         = m["ecs"]
            gr_curves = m["curves"]

            if domain == "Baseline":
                print(f"  {domain:<20} {layer:<10} "
                      f"{e['mean']:>7.4f}+/-{e['std']:>6.4f} "
                      f"{'---':>8} {'---':>18} {'---':>10}  << baseline")
                continue

            d_ecs_seeds, pk_seeds, wins = [], [], 0
            for seed in sorted(SEEDS):
                if seed in bl_curves and seed in gr_curves:
                    bl_e = ecs(bl_curves[seed].tolist())
                    gr_e = ecs(gr_curves[seed].tolist())
                    d    = gr_e - bl_e
                    d_ecs_seeds.append(d)
                    pk_seeds.append(peak_delta(gr_curves[seed].tolist(),
                                               bl_curves[seed].tolist()))
                    if d > 0:
                        wins += 1

            d_mean = np.mean(d_ecs_seeds) if d_ecs_seeds else float("nan")
            pk_m   = np.mean(pk_seeds)    if pk_seeds    else float("nan")
            pk_s   = np.std(pk_seeds)     if pk_seeds    else float("nan")
            flag   = "UP" if d_mean > 0.001 else ("DN" if d_mean < -0.001 else "~")
            wr     = f"{wins}/{len(d_ecs_seeds)}"
            print(f"  {domain:<20} {layer:<10} "
                  f"{e['mean']:>7.4f}+/-{e['std']:>6.4f} "
                  f"{d_mean:>+7.4f} {flag} "
                  f"{pk_m:>+7.4f}+/-{pk_s:>6.4f} "
                  f"{wr:>10}")
    print()


def print_e2t_per_seed_tables(metrics: dict, metric_label: str):
    bl_l   = _bl_layer(metrics)
    if not bl_l:
        return
    bl_e2t = metrics["Baseline"][bl_l]["e2t"]

    for t in E2T_THRESHOLDS:
        print(f"\n{DIV}")
        print(f"  E2T >= {t} --- {metric_label}  "
              f"(epoch when IoU first crosses {t} | lower = faster)")
        print(DIV)
        seed_cols = "".join(f"  {'seed'+str(s):>8}" for s in SEEDS)
        print(f"  {'Domain':<20} {'Layer':<10}{seed_cols}  {'Mean d':>8}  {'Win Rate':>9}")
        print(f"  {DIV2}")

        bl_row = "".join(
            f"  {bl_e2t[t].get(s) or 'N/A':>8}"
            if not isinstance(bl_e2t[t].get(s), int)
            else f"  {bl_e2t[t][s]:>8}"
            for s in SEEDS
        )
        print(f"  {'Baseline':<20} {bl_l:<10}{bl_row}  {'---':>8}  {'---':>9}")

        for domain in metrics:
            if domain == "Baseline":
                continue
            for layer in sorted(metrics[domain]):
                gr_e2t          = metrics[domain][layer]["e2t"][t]
                per_seed_deltas = []
                wins            = 0
                seed_vals       = []
                for seed in SEEDS:
                    gr_val = gr_e2t.get(seed)
                    bl_val = bl_e2t[t].get(seed)
                    seed_vals.append(gr_val)
                    if gr_val is not None and bl_val is not None:
                        d = gr_val - bl_val
                        per_seed_deltas.append(d)
                        if d < 0:
                            wins += 1
                row_cells = "".join(
                    f"  {v:>8}" if isinstance(v, int) else f"  {'N/A':>8}"
                    for v in seed_vals
                )
                d_mean = np.mean(per_seed_deltas) if per_seed_deltas else float("nan")
                wr     = f"{wins}/{len(per_seed_deltas)}"
                flag   = "UP" if d_mean < -0.5 else ("DN" if d_mean > 0.5 else "~")
                print(f"  {domain:<20} {layer:<10}{row_cells}  "
                      f"{d_mean:>+7.1f} {flag}  {wr:>9}")
    print()

# ============================================================
# SECTION 6 — PLOTS
# ============================================================

# ── 1. Convergence curves ────────────────────────────────────

def plot_mean_curves(metrics: dict, metric_label: str, early_only: bool = True):
    max_ep = AUC_EPOCH_END + 5 if early_only else 80
    suffix = "early" if early_only else "full"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    ep_str    = f"Epochs 1-{max_ep}"
    fig.suptitle(
        f"{metric_label} — Mean +/- Std  "
        f"({ep_str if early_only else ep_str})",
        fontsize=13, fontweight="bold", y=1.02
    )

    for col, target_layer in enumerate(["layer1", "layer2"]):
        ax = axes[col]

        # ECS window band
        if early_only:
            ax.axvspan(AUC_EPOCH_START, AUC_EPOCH_END,
                       color="#fff59d", alpha=0.45, zorder=0)

        # Baseline — FIX: use _bl_layer, not bl_l from outer scope
        bl_l = _bl_layer(metrics)
        if bl_l:
            bl_c  = metrics["Baseline"][bl_l]["curves"]
            mat   = np.array([bl_c[s][:max_ep] for s in sorted(bl_c)])
            mu, sd = mat.mean(0), mat.std(0)
            x      = np.arange(1, len(mu) + 1)
            c      = PALETTE["Baseline"]
            ax.plot(x, mu, color=c, lw=2, ls=LAYER_LS.get(bl_l, "-"),
                    label="Baseline", zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, zorder=2)

        # Injection variants
        for domain, layers in metrics.items():
            if domain == "Baseline" or target_layer not in layers:
                continue
            c_dict = layers[target_layer]["curves"]
            mat    = np.array([c_dict[s][:max_ep] for s in sorted(c_dict)])
            mu, sd = mat.mean(0), mat.std(0)
            x      = np.arange(1, len(mu) + 1)
            c      = PALETTE.get(domain, "#333")
            ax.plot(x, mu, color=c, lw=2, ls=LAYER_LS[target_layer],
                    label=domain, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, zorder=2)

        ax.set_title(f"{'Layer 1' if target_layer == 'layer1' else 'Layer 2'} Injection",
                     pad=8)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel(metric_label)

        # Legend: domain colors + layer linestyle note
        dom_handles = [Line2D([0],[0], color=PALETTE.get(d,"#888"), lw=2,
                               ls=LAYER_LS.get(bl_l,"-") if d=="Baseline"
                               else LAYER_LS[target_layer], label=d)
                       for d in list(metrics.keys())
                       if d == "Baseline" or target_layer in metrics[d]]
        ax.legend(handles=dom_handles, loc="lower right", fontsize=8,
                  title=f"{'solid = layer1' if target_layer=='layer1' else 'dashed = layer2'}",
                  title_fontsize=7)

    fname = OUTPUT_DIR / f"curves_{suffix}_{metric_label.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── 2. ECS delta and Peak delta ──────────────────────────────

def plot_ecs_and_peak_delta(metrics: dict, metric_label: str):
    # FIX: get baseline curves using _bl_layer
    bl_l      = _bl_layer(metrics)
    bl_curves = metrics["Baseline"][bl_l]["curves"]  # {seed: np.array}

    domains = [d for d in metrics if d != "Baseline"]
    layers  = ["layer1", "layer2"]
    x       = np.arange(len(domains))
    width   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"ECS delta and Peak delta IoU vs Baseline — {metric_label}",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, mode in zip(axes, ["ecs", "peak"]):
        for li, layer in enumerate(layers):
            vals, errs = [], []
            for domain in domains:
                # FIX: check this layer exists for this domain
                if layer not in metrics[domain]:
                    vals.append(float("nan"))
                    errs.append(0.0)
                    continue

                gr_curves = metrics[domain][layer]["curves"]
                per_seed  = []
                for seed in sorted(SEEDS):
                    if seed not in bl_curves or seed not in gr_curves:
                        continue
                    bl_s = bl_curves[seed].tolist()
                    gr_s = gr_curves[seed].tolist()
                    if mode == "ecs":
                        per_seed.append(ecs(gr_s) - ecs(bl_s))
                    else:
                        per_seed.append(peak_delta(gr_s, bl_s))

                vals.append(np.mean(per_seed) if per_seed else float("nan"))
                errs.append(np.std(per_seed)  if per_seed else 0.0)

            offset = (li - 0.5) * width
            ax.bar(x + offset, vals, width,
                   yerr=errs,
                   color=[PALETTE.get(d, "#999") for d in domains],
                   alpha=0.88 if layer == "layer1" else 0.52,
                   edgecolor="white", linewidth=0.8, capsize=4,
                   error_kw=dict(elinewidth=1.2, ecolor="#333"),
                   label=layer)

        ax.axhline(0, color="#444", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=15, ha="right")
        ax.set_ylabel("delta IoU")
        ax.set_title("delta ECS vs Baseline" if mode == "ecs"
                     else f"Peak delta IoU (ep {AUC_EPOCH_START}-{AUC_EPOCH_END})")

        dom_patches = [mpatches.Patch(color=PALETTE.get(d, "#999"), label=d)
                       for d in domains]
        lay_patches = [
            mpatches.Patch(facecolor="#888", alpha=0.88, label="layer1 (solid)"),
            mpatches.Patch(facecolor="#888", alpha=0.52, label="layer2 (faded)"),
        ]
        ax.legend(handles=dom_patches + [mpatches.Patch(color="none", label="")] + lay_patches,
                  fontsize=8, loc="upper left",
                  title="Domain          Layer", title_fontsize=7)

    fname = OUTPUT_DIR / f"ecs_peak_delta_{metric_label.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── 3. E2T Win Rate ──────────────────────────────────────────

def plot_e2t_winrate(metrics: dict, metric_label: str):
    bl_l   = _bl_layer(metrics)
    bl_e2t = metrics["Baseline"][bl_l]["e2t"]

    domains = [d for d in metrics if d != "Baseline"]
    layers  = ["layer1", "layer2"]
    x       = np.arange(len(domains))
    width   = 0.35

    fig, axes = plt.subplots(1, len(E2T_THRESHOLDS), figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        f"Win Rate — Fraction of Seeds Faster Than Baseline — {metric_label}",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ti, t in enumerate(E2T_THRESHOLDS):
        ax = axes[ti]

        for li, layer in enumerate(layers):
            win_rates = []
            for domain in domains:
                # FIX: check layer exists
                if layer not in metrics[domain]:
                    win_rates.append(float("nan"))
                    continue
                gr_e2t = metrics[domain][layer]["e2t"][t]
                wins, total = 0, 0
                for seed in SEEDS:
                    bl_v = bl_e2t[t].get(seed)
                    gr_v = gr_e2t.get(seed)
                    if bl_v is not None and gr_v is not None:
                        total += 1
                        if gr_v < bl_v:
                            wins += 1
                win_rates.append(wins / total if total else float("nan"))

            offset = (li - 0.5) * width
            bars = ax.bar(x + offset, win_rates, width,
                          color=[PALETTE.get(d, "#999") for d in domains],
                          alpha=0.88 if layer == "layer1" else 0.52,
                          edgecolor="white", linewidth=0.8,
                          label=layer)
            for xi, wr in zip(x + offset, win_rates):
                if not np.isnan(wr):
                    ax.text(xi, wr + 0.04, f"{wr:.0%}",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.axhline(0.5, ls="--", color="#777", lw=1.2)
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(f"IoU >= {t}", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=15, ha="right")
        if ti == 0:
            ax.set_ylabel("Win Rate")
            ax.text(-0.4, 0.52, "50% line", fontsize=7, color="#777", style="italic")

        dom_patches = [mpatches.Patch(color=PALETTE.get(d, "#999"), label=d)
                       for d in domains]
        lay_patches = [
            mpatches.Patch(facecolor="#888", alpha=0.88, label="layer1"),
            mpatches.Patch(facecolor="#888", alpha=0.52, label="layer2"),
        ]
        ax.legend(handles=dom_patches + [mpatches.Patch(color="none", label="")] + lay_patches,
                  fontsize=7, loc="upper right",
                  title="Domain    Layer", title_fontsize=7)

    fname = OUTPUT_DIR / f"e2t_winrate_{metric_label.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── 4. E2T Delta Heatmap ─────────────────────────────────────

def plot_e2t_heatmap(metrics: dict, metric_label: str):
    bl_l   = _bl_layer(metrics)
    bl_e2t = metrics["Baseline"][bl_l]["e2t"]

    row_labels, matrix = [], []

    for domain in metrics:
        if domain == "Baseline":
            continue
        for layer in sorted(metrics[domain]):
            row_labels.append(f"{domain}\n{layer}")
            row = []
            for t in E2T_THRESHOLDS:
                gr_e2t = metrics[domain][layer]["e2t"][t]
                deltas = []
                for seed in SEEDS:
                    bl_v = bl_e2t[t].get(seed)
                    gr_v = gr_e2t.get(seed)
                    if bl_v is not None and gr_v is not None:
                        deltas.append(gr_v - bl_v)
                row.append(np.mean(deltas) if deltas else np.nan)
            matrix.append(row)

    z      = np.array(matrix, dtype=float)
    absmax = np.nanmax(np.abs(z))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(z, cmap="RdYlGn_r", vmin=-absmax, vmax=absmax, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("delta epochs vs baseline\n(negative = faster)", fontsize=9)

    ax.set_xticks(range(len(E2T_THRESHOLDS)))
    ax.set_xticklabels([f"IoU >= {t}" for t in E2T_THRESHOLDS], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(
        f"Mean delta Epochs to Threshold vs Baseline — {metric_label}\n"
        f"Green = faster | Values = mean across {len(SEEDS)} seeds",
        fontsize=11, fontweight="bold"
    )

    for ri in range(len(row_labels)):
        for ci in range(len(E2T_THRESHOLDS)):
            val = z[ri, ci]
            if not np.isnan(val):
                txt_color = "white" if abs(val) > absmax * 0.55 else "black"
                ax.text(ci, ri, f"{val:+.1f}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)

    fname = OUTPUT_DIR / f"e2t_heatmap_{metric_label.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# SECTION 7 — MAIN RUNNER
# ============================================================

def run(log_root: Path = LOG_ROOT):
    print(f"\nLoading logs from: {log_root}")
    results = load_all(log_root)

    if not results:
        print("No data — check LOG_ROOT.")
        return

    print(f"\nDomains: {list(results.keys())}")

    for metric, label in [("train_iou", "Train IoU"), ("test_iou", "Test IoU")]:
        print(f"\n{'#'*70}\n  METRIC: {label}\n{'#'*70}")
        metrics = compute_metrics(results, metric=metric)

        print_ecs_summary(metrics, metric_label=label)
        print_e2t_per_seed_tables(metrics, metric_label=label)

        print(f"\n  Generating plots for {label}...")
        plot_mean_curves(metrics,        metric_label=label, early_only=True)
        plot_mean_curves(metrics,        metric_label=label, early_only=False)
        plot_ecs_and_peak_delta(metrics, metric_label=label)
        plot_e2t_winrate(metrics,        metric_label=label)
        plot_e2t_heatmap(metrics,        metric_label=label)

    print(f"\nAll outputs -> {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
