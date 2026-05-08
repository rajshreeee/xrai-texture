"""
Plot val Dice vs epoch for any set of unet.py log files.

Usage:
    # plot calibration logs
    python plot_calibration.py --logs logs/A_baseline_calib_frac25_*.log logs/A_baseline_calib_frac100_*.log

    # plot existing (pre-calibration) logs
    python plot_calibration.py --logs logs/A_baseline_frac25_*.log logs/A_baseline_frac100_*.log

    # plot everything in logs/
    python plot_calibration.py

Output: logs/calibration_curves.png
"""
import argparse
import re
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


EPOCH_RE   = re.compile(r"Epoch\s+(\d+)/\d+.*?val_dice=([\d.]+)")
BEST_RE    = re.compile(r"best_val_dice=([\d.]+) at epoch (\d+)")
STOP_RE    = re.compile(r"Early stop at epoch (\d+)")
MIN_DELTA  = 0.001


def parse_log(path):
    epochs, val_dices = [], []
    best_dice, best_epoch, stop_epoch = None, None, None

    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                val_dices.append(float(m.group(2)))
            m = BEST_RE.search(line)
            if m:
                best_dice  = float(m.group(1))
                best_epoch = int(m.group(2))
            m = STOP_RE.search(line)
            if m:
                stop_epoch = int(m.group(1))

    label = Path(path).stem
    # shorten label for legend
    label = re.sub(r"_seed\d+_\d{8}_\d{6}$", "", label)
    return {
        "label":       label,
        "epochs":      epochs,
        "val_dices":   val_dices,
        "best_dice":   best_dice,
        "best_epoch":  best_epoch,
        "stop_epoch":  stop_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="*", default=None,
                        help="Log file paths (globs expanded). Default: all logs/*.log")
    parser.add_argument("--out", default="logs/calibration_curves.png")
    args = parser.parse_args()

    if args.logs is None:
        paths = sorted(glob.glob("logs/*.log"))
    else:
        paths = []
        for pattern in args.logs:
            paths.extend(sorted(glob.glob(pattern)))
        paths = sorted(set(paths))

    if not paths:
        print("No log files found.")
        return

    runs = [parse_log(p) for p in paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        if not run["epochs"]:
            print(f"  [warn] no epoch data in {paths[i]}")
            continue
        c = colors[i % len(colors)]
        ax.plot(run["epochs"], run["val_dices"], color=c, linewidth=1.5, label=run["label"])

        if run["best_epoch"] and run["best_dice"] is not None:
            ax.axvline(run["best_epoch"], color=c, linestyle="--", linewidth=0.8, alpha=0.7)
            # plateau band
            ax.axhspan(run["best_dice"] - MIN_DELTA, run["best_dice"] + MIN_DELTA,
                       color=c, alpha=0.06)

        if run["stop_epoch"]:
            ax.axvline(run["stop_epoch"], color=c, linestyle=":", linewidth=0.8, alpha=0.7)

    # legend entries for line styles
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="best epoch"),
        plt.Line2D([0], [0], color="gray", linestyle=":",  linewidth=0.8, label="early-stop epoch"),
        mpatches.Patch(color="gray", alpha=0.15, label=f"±{MIN_DELTA} plateau band"),
    ]
    labels += ["best epoch", "early-stop epoch", f"±{MIN_DELTA} plateau band"]
    ax.legend(handles=handles, labels=labels, fontsize=8, loc="lower right")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Dice")
    ax.set_title("Val Dice vs Epoch — convergence calibration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # print summary table
    print(f"\n{'Label':<45} {'BestEp':>7} {'BestDice':>9} {'StopEp':>7}")
    print("-" * 72)
    for run in runs:
        if not run["epochs"]:
            continue
        print(f"{run['label']:<45} {str(run['best_epoch'] or '?'):>7} "
              f"{str(round(run['best_dice'], 4) if run['best_dice'] else '?'):>9} "
              f"{str(run['stop_epoch'] or '–'):>7}")


if __name__ == "__main__":
    main()
