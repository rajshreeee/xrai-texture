import numpy as np

def epoch_to_convergence(dice_series, threshold=0.98, window=3):
    best   = max(dice_series)
    target = threshold * best
    for i in range(len(dice_series) - window + 1):
        if all(d >= target for d in dice_series[i : i + window]):
            return i + 1
    return None


def compute_convergence_epochs(all_data, threshold=0.98, window=3):
    results = {}
    for group_name, data in all_data.items():
        conv_epochs = []
        for dice_series in data["val_dice"]:
            ep = epoch_to_convergence(dice_series, threshold, window)
            if ep is not None:
                conv_epochs.append(ep)
            else:
                print(f"  Warning: {group_name} — one seed did not converge stably")

        results[group_name] = {
            "mean_conv_epoch": np.mean(conv_epochs) if conv_epochs else None,
            "std_conv_epoch":  np.std(conv_epochs)  if conv_epochs else None,
            "all_conv_epochs": conv_epochs,
            "mean_best_dice":  np.mean(data["best_dice"]),
            "std_best_dice":   np.std(data["best_dice"]),
        }

    print(f"\n=== Convergence Summary (threshold={threshold}, window={window}) ===")
    for group, s in results.items():
        print(f"  {group}:")
        print(f"    Conv epoch : {s['mean_conv_epoch']:.1f} ± {s['std_conv_epoch']:.1f}  → {s['all_conv_epochs']}")
        print(f"    Best Dice  : {s['mean_best_dice']:.4f} ± {s['std_best_dice']:.4f}")

    return results