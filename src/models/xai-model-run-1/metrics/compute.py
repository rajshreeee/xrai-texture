import pandas as pd
from metrics import compute_convergence_epochs
from log_parser import load_all_logs

LOG_DIR = "/ediss_data/ediss2/xai-texture/src/models/xai-model/logs"

if __name__ == "__main__":
    all_data = load_all_logs(LOG_DIR)

    # Convergence epochs
    conv_results = compute_convergence_epochs(all_data, threshold=0.98, window=3)

    # Save summary table
    rows = []
    for group, s in conv_results.items():
        rows.append({
            "Group":           group,
            "Conv Epoch":      f"{s['mean_conv_epoch']:.1f} ± {s['std_conv_epoch']:.1f}",
            "Best Dice":       f"{s['mean_best_dice']:.4f} ± {s['std_best_dice']:.4f}",
            "Seeds completed": len(s['all_conv_epochs'])
        })
    df = pd.DataFrame(rows).set_index("Group")
    print("\n", df.to_string())
    df.to_csv("convergence_summary.csv")

    # Convergence plots (reuse plot_convergence from analyze.py, just pass all_data)
    plot_convergence(all_data, metric="val_dice",
                     title="Validation Dice — Kernel Init vs Baseline",
                     save_path="convergence_dice.png")
    plot_convergence(all_data, metric="val_loss",
                     title="Validation Loss — Kernel Init vs Baseline",
                     save_path="convergence_loss.png")