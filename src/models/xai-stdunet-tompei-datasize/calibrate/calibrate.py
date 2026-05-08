"""
Calibration pilot: run A_baseline for frac25 and frac100 with max_epochs=100
so we can inspect convergence and set justified FRACTION_MAX_EPOCHS in unet.py.

Usage:
    python calibrate.py

Logs appear in logs/ with run names like A_baseline_calib_frac25_seed42.
After this finishes, run plot_calibration.py to inspect the curves.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from unet import train_one_run, BATCH_SIZE, MIN_DELTA, T_MAX
from dataset import make_loaders

CALIB_FRACTIONS  = [0.25, 1.0]
CALIB_SEED       = 42
CALIB_MAX_EPOCHS = 100
CALIB_PATIENCE   = 15


def main():
    results = []
    for fraction in CALIB_FRACTIONS:
        print(f"\n{'='*60}")
        print(f"Calibration run | fraction={fraction} ({int(fraction*100)}%)")
        train_loader, val_loader, test_loader = make_loaders(
            CALIB_SEED, BATCH_SIZE, train_fraction=fraction
        )
        n_train = len(train_loader.dataset)
        print(f"n_train={n_train} | max_epochs={CALIB_MAX_EPOCHS} | patience={CALIB_PATIENCE} | min_delta={MIN_DELTA}")

        result = train_one_run(
            group_name    = "A_baseline_calib",
            seed          = CALIB_SEED,
            inject_blocks = None,
            train_loader  = train_loader,
            val_loader    = val_loader,
            test_loader   = test_loader,
            freeze_epochs = 0,
            train_fraction= fraction,
            max_epochs    = CALIB_MAX_EPOCHS,
            t_max         = T_MAX,
            patience      = CALIB_PATIENCE,
            min_delta     = MIN_DELTA,
        )
        results.append(result)
        print(f"  → best_val_dice={result['best_val_dice']:.4f} at epoch {result['best_epoch']} | "
              f"stopped_early={result['stopped_early']}")

    df = pd.DataFrame(results)
    out = "logs/calibration_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nCalibration summary saved to {out}")
    print(df[["train_fraction", "best_val_dice", "best_epoch", "stopped_early"]].to_string(index=False))

    print("\nNext: run  python plot_calibration.py  to inspect convergence curves,")
    print("then update FRACTION_MAX_EPOCHS in unet.py and re-run the full experiment.")


if __name__ == "__main__":
    main()
