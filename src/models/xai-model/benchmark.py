import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Fetch all runs from WandB project ---
def fetch_wandb_runs(project_name="mammography-kernel-init"):
    api = wandb.Api()
    runs = api.runs(project_name)

    all_data = defaultdict(lambda: defaultdict(list))  # group → metric → list of series

    for run in runs:
        group = run.config.get("group")
        history = run.history(keys=["epoch", "val_loss", "val_dice"], 
                               pandas=True)

        if history.empty or group is None:
            continue

        # One series per seed
        all_data[group]["val_loss"].append(history["val_loss"].values)
        all_data[group]["val_dice"].append(history["val_dice"].values)
        all_data[group]["epochs"].append(history["epoch"].values)

    return all_data


# --- Plot mean ± std across seeds per group ---
def plot_convergence(all_data, metric="val_dice", title=None, save_path=None):
    GROUP_STYLES = {
        "A_baseline":     {"color": "#e74c3c", "label": "Baseline (Random Init)"},
        "B_layer2_init":  {"color": "#2ecc71", "label": "Kernel Init — Layer2"},
        "C_layer3_init":  {"color": "#3498db", "label": "Kernel Init — Layer3"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for group_name, style in GROUP_STYLES.items():
        if group_name not in all_data:
            continue

        series_list = all_data[group_name][metric]  # list of arrays, one per seed
        epochs      = all_data[group_name]["epochs"][0]  # same for all seeds

        # Stack into (n_seeds, n_epochs)
        matrix = np.stack(series_list, axis=0)
        mean   = matrix.mean(axis=0)
        std    = matrix.std(axis=0)

        ax.plot(epochs, mean,
                color=style["color"],
                label=style["label"],
                linewidth=2)

        ax.fill_between(epochs,
                        mean - std,
                        mean + std,
                        color=style["color"],
                        alpha=0.15)

    # Mark key convergence epochs
    for epoch in [5, 10, 20]:
        ax.axvline(x=epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(epoch + 0.3, ax.get_ylim()[0], f"ep{epoch}",
                fontsize=8, color="gray")

    metric_label = "Validation Dice" if metric == "val_dice" else "Validation Loss"
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title or f"{metric_label} — Mean ± Std (5 seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


# --- Early convergence summary table ---
def print_early_convergence_table(all_data, key_epochs=[5, 10, 20]):
    rows = []
    for group_name, data in all_data.items():
        series_list = data["val_dice"]
        epochs      = data["epochs"][0]
        matrix      = np.stack(series_list, axis=0)

        row = {"Group": group_name}
        for ep in key_epochs:
            idx = np.where(epochs == ep)[0]
            if len(idx) > 0:
                vals = matrix[:, idx[0]]
                row[f"Dice@ep{ep}"] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        
        # Best dice across all epochs
        best_per_seed = matrix.max(axis=1)
        row["Best Dice"] = f"{best_per_seed.mean():.4f} ± {best_per_seed.std():.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Group")
    print("\n=== Early Convergence Summary ===")
    print(df.to_string())
    df.to_csv("convergence_summary.csv")
    return df


# --- Run everything ---
if __name__ == "__main__":
    all_data = fetch_wandb_runs("mammography-kernel-init")

    plot_convergence(all_data, metric="val_dice",
                     title="Validation Dice — Kernel Init vs Baseline",
                     save_path="convergence_dice.png")

    plot_convergence(all_data, metric="val_loss",
                     title="Validation Loss — Kernel Init vs Baseline",
                     save_path="convergence_loss.png")

    print_early_convergence_table(all_data)