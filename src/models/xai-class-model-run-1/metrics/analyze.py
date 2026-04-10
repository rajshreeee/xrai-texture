import re
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    # New format with train_dice/train_iou
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| train_loss=[\d.]+ \| train_dice=[\d.]+ \| train_iou=([\d.]+) \| val_loss=[\d.]+ \| val_dice=([\d.]+) \| val_iou=([\d.]+)"
    )
    # Old format fallback (no train_dice/train_iou)
    epoch_pattern_old = re.compile(
        r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| val_dice=([\d.]+) \| val_iou=([\d.]+)"
    )
    best_pattern = re.compile(r"best_val_dice=([\d.]+)")

    data = {"epochs": [], "train_loss": [], "train_iou": [],
            "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = None

    with open(log_path) as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                data["epochs"].append(int(m.group(1)))
                data["train_iou"].append(float(m.group(2)))
                data["val_dice"].append(float(m.group(3)))
                data["val_iou"].append(float(m.group(4)))
                data["train_loss"].append(0.0)
                data["val_loss"].append(0.0)
            else:
                m2 = epoch_pattern_old.search(line)
                if m2:
                    data["epochs"].append(int(m2.group(1)))
                    data["train_loss"].append(float(m2.group(2)))
                    data["train_iou"].append(0.0)
                    data["val_loss"].append(float(m2.group(3)))
                    data["val_dice"].append(float(m2.group(4)))
                    data["val_iou"].append(float(m2.group(5)))

            b = best_pattern.search(line)
            if b:
                best_dice = float(b.group(1))

    data["best_dice"] = best_dice
    return data


def load_all_logs(log_dir):
    log_dir = Path(log_dir)
    file_pattern = re.compile(r"^(.+)_seed(\d+)_\d{8}_\d{6}\.log$")
    all_data = {}

    for log_file in sorted(log_dir.glob("*.log")):
        m = file_pattern.match(log_file.name)
        if not m:
            print(f"  Skipping unrecognized file: {log_file.name}")
            continue

        group_name = m.group(1)
        seed       = int(m.group(2))
        parsed     = parse_log_file(log_file)

        if not parsed["epochs"]:
            print(f"  Warning: no epoch data in {log_file.name}")
            continue

        if group_name not in all_data:
            all_data[group_name] = {
                "val_dice": [], "val_iou": [], "val_loss": [],
                "train_iou": [], "train_loss": [],
                "epochs": parsed["epochs"],
                "best_dice": [], "seeds": []
            }

        for key in ["val_dice", "val_iou", "val_loss", "train_iou", "train_loss"]:
            all_data[group_name][key].append(np.array(parsed[key]))
        all_data[group_name]["best_dice"].append(parsed["best_dice"])
        all_data[group_name]["seeds"].append(seed)
        
        best_str = f"{parsed['best_dice']:.4f}" if parsed['best_dice'] is not None else "N/A"
        print(f"  Loaded: {log_file.name} → group={group_name}, seed={seed}, "
             f"epochs={len(parsed['epochs'])}, best_dice={best_str}")

    return all_data


def get_at_epoch(series, epochs, ep):
    if ep - 1 < len(series):
        return series[ep - 1]
    return None


def print_per_seed_tables(all_data, metric="val_iou", checkpoints=None):
    if checkpoints is None:
        checkpoints = list(range(5, 101, 5))

    baseline       = "A_baseline"
    compare_groups = [g for g in ["B_layer2_init", "C_layer3_init", "D_layer1_init"]
                      if g in all_data]
    label          = "Train IoU" if metric == "train_iou" else "Val IoU"

    if baseline not in all_data:
        print(f"ERROR: {baseline} not found in logs.")
        return

    for seed_idx, seed in enumerate(all_data[baseline]["seeds"]):
        print(f"\n{'='*70}")
        print(f"  SEED {seed} — {label} (per-layer vs baseline)")
        print(f"{'='*70}")

        a_series = all_data[baseline][metric][seed_idx]
        epochs_A = all_data[baseline]["epochs"]

        for grp in compare_groups:
            if seed not in all_data[grp]["seeds"]:
                print(f"  [Missing seed {seed} for {grp}]")
                continue

            grp_idx  = all_data[grp]["seeds"].index(seed)
            g_series = all_data[grp][metric][grp_idx]
            epochs_G = all_data[grp]["epochs"]
            short    = grp.replace("_init", "").replace("layer", "L")

            print(f"\n  {grp} vs {baseline}")
            print(f"  {'Epoch':>6} | {'A_baseline':>10} | {short:>12} | {'Δ':>8}")
            print(f"  {'-'*46}")

            for ep in checkpoints:
                a_val = get_at_epoch(a_series, epochs_A, ep)
                g_val = get_at_epoch(g_series, epochs_G, ep)
                if a_val is None or g_val is None:
                    continue
                delta = g_val - a_val
                flag  = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "~")
                print(f"  {ep:>6} | {a_val:>10.4f} | {g_val:>12.4f} | {delta:>+8.4f} {flag}")


def print_aggregate_tables(all_data, metric="val_iou", checkpoints=None):
    if checkpoints is None:
        checkpoints = list(range(5, 101, 5))

    baseline       = "A_baseline"
    compare_groups = [g for g in ["B_layer2_init", "C_layer3_init", "D_layer1_init"]
                      if g in all_data]
    label          = "Train IoU" if metric == "train_iou" else "Val IoU"
    epochs_A       = all_data[baseline]["epochs"]

    print(f"\n{'='*70}")
    print(f"  AGGREGATE (mean ± std across all seeds) — {label}")
    print(f"{'='*70}")

    for grp in compare_groups:
        epochs_G = all_data[grp]["epochs"]
        print(f"\n  {grp} vs {baseline}")
        print(f"  {'Epoch':>6} | {'A  mean±std':>16} | {'Grp mean±std':>16} | {'Δ mean':>8} | {'Δ std':>7}")
        print(f"  {'-'*64}")

        for ep in checkpoints:
            a_vals = [get_at_epoch(s, epochs_A, ep) for s in all_data[baseline][metric]]
            g_vals = [get_at_epoch(s, epochs_G, ep) for s in all_data[grp][metric]]
            a_vals = [v for v in a_vals if v is not None]
            g_vals = [v for v in g_vals if v is not None]
            if not a_vals or not g_vals:
                continue

            a_mean, a_std = np.mean(a_vals), np.std(a_vals)
            g_mean, g_std = np.mean(g_vals), np.std(g_vals)
            deltas        = [g - a for g, a in zip(g_vals, a_vals)]
            d_mean, d_std = np.mean(deltas), np.std(deltas)
            flag          = "▲" if d_mean > 0.001 else ("▼" if d_mean < -0.001 else "~")
            print(f"  {ep:>6} | {a_mean:.4f}±{a_std:.4f} | {g_mean:.4f}±{g_std:.4f} "
                  f"| {d_mean:>+8.4f} | {d_std:>7.4f} {flag}")

    # Best dice summary
    print(f"\n  {'─'*62}")
    print(f"  {'Group':<22} | {'Best Dice':>10} | {'± std':>7} | {'vs A':>8}")
    print(f"  {'─'*62}")
    a_best = [v for v in all_data[baseline]["best_dice"] if v is not None]
    a_bm   = np.mean(a_best)
    print(f"  {'A_baseline':<22} | {a_bm:>10.4f} | {np.std(a_best):>7.4f} |    —")
    for grp in compare_groups:
        g_best = [v for v in all_data[grp]["best_dice"] if v is not None]
        if not g_best:
            continue
        gm, gs = np.mean(g_best), np.std(g_best)
        print(f"  {grp:<22} | {gm:>10.4f} | {gs:>7.4f} | {gm - a_bm:>+8.4f}")


def run_analysis(log_dir):
    print(f"\nLoading logs from: {log_dir}")
    all_data = load_all_logs(log_dir)

    if not all_data:
        print("No logs found — check your log directory path.")
        return

    print(f"\nGroups found: {list(all_data.keys())}")
    for g, d in all_data.items():
        print(f"  {g}: {len(d['seeds'])} seeds → {d['seeds']}")

    print_per_seed_tables(all_data, metric="train_iou")
    print_per_seed_tables(all_data, metric="val_iou")
    print_aggregate_tables(all_data, metric="train_iou")
    print_aggregate_tables(all_data, metric="val_iou")


if __name__ == "__main__":
    # import sys
    # log_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs"
    run_analysis("/ediss_data/ediss2/xai-texture/src/models/xai-model-run-2/logs")