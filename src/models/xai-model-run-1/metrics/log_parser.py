import re
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse a training log file and extract per-epoch metrics.
    Returns dict with lists: epochs, train_loss, val_loss, val_dice, val_iou
    """
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| val_dice=([\d.]+) \| val_iou=([\d.]+)"
    )
    best_pattern = re.compile(r"best_val_dice=([\d.]+)")

    data = {"epochs": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = None

    with open(log_path) as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                data["epochs"].append(int(m.group(1)))
                data["train_loss"].append(float(m.group(2)))
                data["val_loss"].append(float(m.group(3)))
                data["val_dice"].append(float(m.group(4)))
                data["val_iou"].append(float(m.group(5)))

            b = best_pattern.search(line)
            if b:
                best_dice = float(b.group(1))

    data["best_dice"] = best_dice
    return data


def load_all_logs(log_dir):
    """
    Load all log files and organize by group.
    Returns: {group_name: {"val_dice": [series_seed1, ...], "epochs": [...], "best_dice": [...]}}
    """
    log_dir = Path(log_dir)
    # Filename format: {group_name}_seed{seed}_{timestamp}.log
    file_pattern = re.compile(r"^(.+)_seed(\d+)_\d{8}_\d{6}\.log$")

    all_data = {}

    for log_file in sorted(log_dir.glob("*.log")):
        m = file_pattern.match(log_file.name)
        if not m:
            print(f"  Skipping unrecognized file: {log_file.name}")
            continue

        group_name = m.group(1)
        seed       = int(m.group(2))

        parsed = parse_log_file(log_file)

        if not parsed["epochs"]:
            print(f"  Warning: no epoch data in {log_file.name}")
            continue

        if group_name not in all_data:
            all_data[group_name] = {
                "val_dice":   [],
                "val_loss":   [],
                "train_loss": [],
                "val_iou":    [],
                "epochs":     parsed["epochs"],
                "best_dice":  [],
                "seeds":      []
            }

        all_data[group_name]["val_dice"].append(np.array(parsed["val_dice"]))
        all_data[group_name]["val_loss"].append(np.array(parsed["val_loss"]))
        all_data[group_name]["train_loss"].append(np.array(parsed["train_loss"]))
        all_data[group_name]["val_iou"].append(np.array(parsed["val_iou"]))
        all_data[group_name]["best_dice"].append(parsed["best_dice"])
        all_data[group_name]["seeds"].append(seed)

        print(f"  Loaded: {log_file.name} → group={group_name}, seed={seed}, "
              f"epochs={len(parsed['epochs'])}, best_dice={parsed['best_dice']:.4f}")

    return all_data