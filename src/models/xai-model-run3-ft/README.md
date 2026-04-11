# Steps
1. change config.py
2. Update kernels


# Results

Here is a full research-paper-style framing of your results, based on thorough analysis of both training logs: 

## Experimental Setup

Both runs share identical hyperparameters — 80 epochs, learning rate 1×10⁻⁵, batch size 8, and seed 42 — differing only in initialization strategy. Group **A** (`A_baseline`) uses standard random initialization with no kernel injection (`inject_layer=None`), while Group **B** (`B_layer2_init`) injects Fourier-derived kernels into `layer2` of the backbone at initialization (`inject_layer=layer2`). Both are evaluated on the CBIS-DDSM mammography dataset using Dice coefficient and IoU as primary metrics. [ppl-ai-file-upload.s3.amazonaws]
## Peak Performance

The key quantitative findings at the best checkpoint are summarized below: 
| Metric | A: Baseline | B: Layer2 Kernel Init | Δ (B − A) |
|---|---|---|---|
| Best Val Dice | 0.7618 (ep. 70) | **0.7706** (ep. 80) | **+0.0088** |
| Best Val IoU | — | — | — |
| Best Test Dice | 0.7892 | **0.7948** | **+0.0056** |
| Final Epoch Test Dice | 0.7871 | **0.7941** | **+0.0070** |

Group B achieves a best validation Dice of **0.7706** versus **0.7618** for the baseline, and a best test Dice of **0.7948** versus **0.7892**, representing consistent improvements of ~0.9 and ~0.6 percentage points respectively. Both models converge smoothly without signs of overfitting at 80 epochs, as evidenced by validation and test Dice tracking together throughout training. ## Accelerated Early Convergence

The most pronounced effect of layer2 kernel injection is in *early-epoch convergence speed*:[chart:output/checkpoint_val_dice.png]

- **Epoch 5:** A=0.5886, B=**0.6117** → Δ = **+0.0231**
- **Epoch 10:** A=0.6278, B=**0.6744** → Δ = **+0.0466** (largest margin)
- **Epoch 15:** A=0.6799, B=**0.7128** → Δ = **+0.0329**
- **Epoch 20:** A=0.7016, B=0.7087 → Δ = +0.0071 (gap narrows)

The kernel-initialized model reaches a validation Dice of **0.7128 by epoch 15**, a performance level that the baseline does not achieve until approximately epoch 25. This represents an effective speedup of ~10 epochs in early training, which is particularly meaningful in compute-constrained medical imaging scenarios. 
## Research-Paper Framing

You can frame these findings as follows for your paper:

> **"Effect of Fourier Kernel Injection on Convergence and Segmentation Performance"**
>
> We investigate whether initializing the convolutional weights of `layer2` with Fourier transform-derived kernels provides a statistically meaningful advantage over random initialization on the CBIS-DDSM mammography segmentation task. Results demonstrate two key effects: (1) **accelerated early-epoch learning** — the kernel-initialized model (B) surpasses the baseline by up to **4.66 percentage points in validation Dice at epoch 10**, suggesting that structured frequency-domain priors provide a better optimization starting point; and (2) **improved final-epoch generalization** — Group B achieves a best test Dice of **0.7948 vs. 0.7892**, a gain of **+0.56 pp**, and a best validation Dice of **0.7706 vs. 0.7618** (+0.88 pp). The upward trajectory of Group B at epoch 80 further suggests that the kernel-initialized model has not yet saturated, hinting at additional performance potential with extended training.

## Notable Observations

- The early-convergence effect is especially strong in the **first 15 epochs**, aligning with the hypothesis that frequency-domain kernel priors encode low-level texture information directly relevant to mammographic tissue patterns. 
- Both runs were stable with no divergence, confirming that the kernel injection does not destabilize training dynamics at LR=1e-5.
- Group B's best epoch is 80 (still improving at the end), whereas A's best is epoch 70, suggesting Group B may need **extended training (e.g., 100–120 epochs)** for a fairer comparison and to reach its true peak.
- For the paper, you should note this is a **single-seed result (seed=42)** — reporting across multiple seeds (e.g., 3–5) would strengthen statistical validity of the claim.