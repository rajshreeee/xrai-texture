import re
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

LOG_DIR = Path("/ediss_data/ediss2/xai-texture/src/models/rand-xai-vgg11unet-tompei-datasize/logs")
FRACTIONS = [10, 25, 50, 75, 100]

# ============================================================
# REGEX
# ============================================================

filename_pattern = re.compile(r"(A_baseline|C_rand_enc1)_frac(\d+)_seed\d+")

epoch_line_pattern = re.compile(
    r"Epoch\s+(\d+)/(\d+).*?"
    r"test_dice=(\d+\.\d+).*?"
    r"test_iou=(\d+\.\d+)"
)

best_epoch_pattern = re.compile(
    r"best_val_dice=.*?at epoch (\d+)"
)

# ============================================================
# PARSE LOG FILES
# ============================================================

results = {}

for log_file in LOG_DIR.glob("*.log"):

    match = filename_pattern.search(log_file.name)
    if not match:
        continue

    model_name = match.group(1)
    fraction = int(match.group(2))

    with open(log_file, "r") as f:
        lines = f.readlines()

    best_epoch = None
    for line in reversed(lines):
        m = best_epoch_pattern.search(line)
        if m:
            best_epoch = int(m.group(1))
            break

    if best_epoch is None:
        continue

    best_test_dice = None
    best_test_iou  = None

    for line in lines:
        m = epoch_line_pattern.search(line)
        if not m:
            continue
        if int(m.group(1)) == best_epoch:
            best_test_dice = float(m.group(3))
            best_test_iou  = float(m.group(4))
            break

    curve = []
    for line in lines:
        m = epoch_line_pattern.search(line)
        if m:
            curve.append({
                "epoch": int(m.group(1)),
                "dice":  float(m.group(3)),
                "iou":   float(m.group(4))
            })

    if fraction not in results:
        results[fraction] = {}

    results[fraction][model_name] = {
        "dice":       best_test_dice,
        "iou":        best_test_iou,
        "best_epoch": best_epoch,
        "curve":      curve,
    }

# ============================================================
# TIME-TO-THRESHOLD
# ============================================================

def first_epoch_to_reach(curve, target):
    for p in curve:
        if p["dice"] >= target:
            return p["epoch"]
    return None

# ============================================================
# BUILD TABLE
# ============================================================

rows = []

for fraction in FRACTIONS:

    base = results.get(fraction, {}).get("A_baseline", {})
    enc  = results.get(fraction, {}).get("C_rand_enc1", {})

    base_best = base.get("dice")
    enc_best  = enc.get("dice")

    if base_best is None or enc_best is None:
        continue

    target = min(base_best, enc_best)

    base_epoch_to_target = first_epoch_to_reach(base.get("curve", []), target)
    enc_epoch_to_target  = first_epoch_to_reach(enc.get("curve",  []), target)

    rows.append({
        "Train Fraction":           f"{fraction}%",
        "Baseline Test Dice":       base_best,
        "Rand-Enc1 Test Dice":      enc_best,
        "Baseline Test IoU":        base.get("iou"),
        "Rand-Enc1 Test IoU":       enc.get("iou"),
        "Target Dice (min best)":   target,
        "Baseline Epoch @ Target":  base_epoch_to_target,
        "Rand-Enc1 Epoch @ Target": enc_epoch_to_target,
    })

df = pd.DataFrame(rows)

# ============================================================
# FORMATTING
# ============================================================

metric_cols = [
    "Baseline Test Dice",
    "Rand-Enc1 Test Dice",
    "Baseline Test IoU",
    "Rand-Enc1 Test IoU",
    "Target Dice (min best)",
]

for col in metric_cols:
    df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

# ============================================================
# OUTPUT
# ============================================================

print("\n=== FAIR PAPER TABLE (TIME-TO-THRESHOLD) ===\n")
print(df.to_string(index=False))

csv_path = LOG_DIR / "paper_results_fair_table.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved CSV → {csv_path}")
