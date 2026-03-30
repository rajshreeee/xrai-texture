import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# YOUR FFT KERNELS — all 5 layers
# ============================================================

all_layer_kernels = {
    "layer1": [
        np.array([[ 6,-7,-10, 5, 6],[ 6,-4,-10, 2, 6],[ 6, 0,-10, 0, 6],[ 6, 2,-10,-4, 6],[ 6, 5,-10,-7, 6]]),
        np.array([[-8,-8,-8,-8,-7],[-3, 0, 4, 7,10],[ 8, 9, 9, 9, 8],[10, 7, 4, 0,-3],[-7,-8,-8,-8,-8]]),
        np.array([[-8,-8,-8,-8,-7],[-3, 0, 4, 7,10],[ 8, 9, 9, 9, 8],[10, 7, 4, 0,-3],[-7,-8,-8,-8,-8]]),
        np.array([[ 6,-4,-10, 0, 6],[ 6,-3,-10, 0, 6],[ 7, 0,-10, 0, 7],[ 6, 0,-10,-3, 6],[ 6, 0,-10,-4, 6]]),
        np.array([[ 6,-4,-10, 0, 6],[ 6,-3,-10, 0, 6],[ 7, 0,-10, 0, 7],[ 6, 0,-10,-3, 6],[ 6, 0,-10,-4, 6]]),
        np.array([[ 7, 8, 8, 7, 4],[ 5, 0,-2,-6,-9],[-8,-10,-10,-10,-8],[-9,-6,-2, 0, 5],[ 4, 7, 8, 8, 7]]),
    ],
    "layer2": [
        np.array([[-10,-5,-3,-4,-9],[-5, 0, 4, 3,-2],[-3, 4, 7, 7, 3],[-4, 3, 7, 7, 3],[-9,-2, 3, 3, 0]]),
        np.array([[-10,-5,-2,-2,-3],[-5, 0, 2, 3, 0],[-2, 2, 4, 5, 2],[-2, 2, 5, 4, 2],[-3, 0, 2, 2, 0]]),
        np.array([[-10,-5,-2,-2,-3],[-5, 0, 2, 3, 0],[-2, 2, 4, 5, 2],[-2, 2, 5, 4, 2],[-3, 0, 2, 2, 0]]),
        np.array([[-10,-5,-3,-4,-9],[-5, 0, 4, 3,-2],[-3, 4, 7, 7, 3],[-4, 3, 7, 7, 3],[-9,-2, 3, 3, 0]]),
        np.array([[-9, 0, 4, 0,-10],[-5, 5, 8, 5,-5],[-4, 6,10, 6,-4],[-5, 5, 8, 5,-5],[-10, 0, 4, 0,-9]]),
        np.array([[-9,-5,-4,-5,-10],[ 0, 5, 6, 5, 0],[ 4, 8,10, 8, 4],[ 0, 5, 6, 5, 0],[-10,-5,-4,-5,-9]]),
    ],
    "layer3": [
        np.array([[-10,-6,-3,-2,-3],[-5, 0, 2, 3, 2],[-2, 2, 5, 6, 4],[-2, 2, 4, 5, 3],[-5, 0, 0, 0, 0]]),
        np.array([[-10,-6,-3,-2,-3],[-5, 0, 2, 3, 2],[-2, 2, 5, 6, 4],[-2, 2, 4, 5, 3],[-5, 0, 0, 0, 0]]),
        np.array([[-8,-4, 0,-2,-5],[-4, 0, 4, 4, 0],[-3, 3, 6, 6, 4],[-5, 0, 5, 6, 4],[-10,-4, 0, 2, 0]]),
        np.array([[-8,-4, 0,-2,-5],[-4, 0, 4, 4, 0],[-3, 3, 6, 6, 4],[-5, 0, 5, 6, 4],[-10,-4, 0, 2, 0]]),
        np.array([[ 0,-2,-5,-8,-10],[ 7, 5, 3, 0,-4],[ 4, 6, 7, 6, 4],[-4, 0, 3, 5, 7],[-10,-8,-5,-2, 0]]),
        np.array([[ 3, 0,-5,-8,-10],[ 6, 6, 3, 0,-5],[ 3, 6, 7, 6, 3],[-5, 0, 3, 6, 6],[-10,-8,-5, 0, 3]]),
    ],
    "layer4": [
        np.array([[-6,-7,-7,-7,-6],[ 0, 2, 2, 2, 2],[ 9,10,10,10, 9],[ 2, 2, 2, 2, 0],[-6,-7,-7,-7,-6]]),
        np.array([[-6,-7,-7,-7,-6],[ 0, 2, 2, 2, 2],[ 9,10,10,10, 9],[ 2, 2, 2, 2, 0],[-6,-7,-7,-7,-6]]),
        np.array([[-7,-8,-7,-7,-6],[ 0, 0, 2, 4, 4],[ 9,10,10,10, 9],[ 4, 4, 2, 0, 0],[-6,-7,-7,-8,-7]]),
        np.array([[-7,-8,-7,-7,-6],[ 0, 0, 2, 4, 4],[ 9,10,10,10, 9],[ 4, 4, 2, 0, 0],[-6,-7,-7,-8,-7]]),
        np.array([[-10,-6,-3,-2,-2],[-6,-2, 0, 2, 0],[-3, 0, 3, 4, 3],[-2, 2, 4, 5, 4],[-2, 0, 3, 4, 2]]),
        np.array([[-10,-6,-3,-2,-2],[-6,-2, 0, 2, 0],[-3, 0, 3, 4, 3],[-2, 2, 4, 5, 4],[-2, 0, 3, 4, 2]]),
    ],
    "classifier": [
        np.array([[-9,-4,-3,-5,-10],[-4, 0, 4, 3,-2],[-3, 4, 7, 7, 2],[-4, 3, 7, 7, 4],[-9,-2, 3, 4, 0]]),
        np.array([[-10,-5,-2,-2,-3],[-5, 0, 2, 2, 0],[-2, 2, 4, 4, 2],[ 0, 3, 4, 4, 2],[-3, 0, 2, 2, 0]]),
        np.array([[-10,-5,-2,-2,-3],[-5, 0, 2, 2, 0],[-2, 2, 4, 4, 2],[ 0, 3, 4, 4, 2],[-3, 0, 2, 2, 0]]),
        np.array([[-9,-4,-3,-5,-10],[-4, 0, 4, 3,-2],[-3, 4, 7, 7, 2],[-4, 3, 7, 7, 4],[-9,-2, 3, 4, 0]]),
        np.array([[-10, 4, 9, 5,-9],[-10, 5, 9, 5,-9],[-9, 5, 9, 5,-9],[-9, 5, 9, 5,-10],[-9, 5, 9, 4,-10]]),
        np.array([[ 6, 6, 6, 6, 6],[-3,-2, 0, 0, 0],[-10,-10,-10,-10,-10],[ 0, 0, 0,-2,-3],[ 6, 6, 6, 6, 6]]),
    ],
}


# ============================================================
# KERNEL PREPARATION UTILITY
# ============================================================

def prepare_fft_kernels(layer_kernels, in_channels=1, out_channels=64, kernel_size=5):
    """
    Builds a (out_channels, in_channels, K, K) weight tensor from FFT kernels.
    
    - Tiles kernels across out_channels by cycling through the list
    - Normalizes each kernel to zero mean, unit std (matching Conv init scale)
    - Pads with random noise if out_channels > available kernels (keeps structure
      in the first N filters, random fallback for the rest)
    """
    kernels = []
    n = len(layer_kernels)

    for i in range(out_channels):
        k = layer_kernels[i % n].astype(np.float32)
        # Zero mean, unit std normalization
        k = k - k.mean()
        std = k.std()
        if std > 1e-8:
            k = k / std
        # Scale to match typical He init magnitude for conv layers
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        k = k * scale
        kernels.append(k)

    # Stack → (out_channels, kernel_size, kernel_size)
    weight = np.stack(kernels, axis=0)

    # Add in_channels dim → (out_channels, in_channels, K, K)
    # For grayscale (in_channels=1): just unsqueeze
    # For RGB (in_channels=3): replicate across channels
    weight = np.stack([weight] * in_channels, axis=1)

    return torch.tensor(weight, dtype=torch.float32)


# ============================================================
# UNET BUILDING BLOCKS
# ============================================================

class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,          # binary segmentation
        base_filters=64,
        fft_kernels=None,        # list of 6 numpy arrays (5×5)
        freeze_first_layer=False,
        fft_kernel_size=5,
    ):
        """
        Parameters
        ----------
        fft_kernels : list of np.array | None
            If provided, injects these as the weights of the first conv layer.
            If None, uses default random (He) initialization.
        freeze_first_layer : bool
            If True, the first conv layer weights are frozen during training.
            If False, they are fine-tunable (warm start).
        """
        super().__init__()

        f = base_filters  # 64

        # ---- ENCODER ----
        # First conv uses FFT kernel size (5×5) if kernels provided, else 3×3
        first_kernel = fft_kernel_size if fft_kernels is not None else 3
        self.enc1_conv1 = nn.Conv2d(
            in_channels, f,
            kernel_size=first_kernel,
            padding=first_kernel // 2,
            bias=False
        )
        self.enc1_bn1   = nn.BatchNorm2d(f)
        self.enc1_relu1 = nn.ReLU(inplace=True)
        self.enc1_conv2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.enc1_bn2   = nn.BatchNorm2d(f)
        self.enc1_relu2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = DoubleConv(f,    f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = DoubleConv(f*2,  f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4  = DoubleConv(f*4,  f*8)
        self.pool4 = nn.MaxPool2d(2)

        # ---- BOTTLENECK ----
        self.bottleneck = DoubleConv(f*8, f*16)

        # ---- DECODER ----
        self.up4    = nn.ConvTranspose2d(f*16, f*8,  2, stride=2)
        self.dec4   = DoubleConv(f*16, f*8)
        self.up3    = nn.ConvTranspose2d(f*8,  f*4,  2, stride=2)
        self.dec3   = DoubleConv(f*8,  f*4)
        self.up2    = nn.ConvTranspose2d(f*4,  f*2,  2, stride=2)
        self.dec2   = DoubleConv(f*4,  f*2)
        self.up1    = nn.ConvTranspose2d(f*2,  f,    2, stride=2)
        self.dec1   = DoubleConv(f*2,  f)

        # ---- OUTPUT ----
        self.out_conv = nn.Conv2d(f, out_channels, 1)

        # ---- INJECT FFT KERNELS ----
        if fft_kernels is not None:
            self._inject_fft_kernels(fft_kernels, in_channels, f, fft_kernel_size)
            if freeze_first_layer:
                self.enc1_conv1.weight.requires_grad = False
                print(f"✓ FFT kernels injected into enc1_conv1 [FROZEN]")
            else:
                print(f"✓ FFT kernels injected into enc1_conv1 [fine-tunable]")
        else:
            print("✓ UNet initialized with random weights (baseline)")

    def _inject_fft_kernels(self, layer_kernels, in_channels, out_channels, kernel_size):
        weight = prepare_fft_kernels(
            layer_kernels,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        with torch.no_grad():
            self.enc1_conv1.weight.copy_(weight)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_relu2(self.enc1_bn2(self.enc1_conv2(
             self.enc1_relu1(self.enc1_bn1(self.enc1_conv1(x))))))
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # raw logits → apply sigmoid for binary


# ============================================================
# INSTANTIATE ALL 3 CONDITIONS
# ============================================================

# Condition 1: Random init baseline
model_random = UNet(in_channels=1, out_channels=1, fft_kernels=None)

# Condition 2: FFT kernels injected, FROZEN (pure prior)
model_fft_frozen = UNet(
    in_channels=1, out_channels=1,
    fft_kernels=all_layer_kernels["layer1"],  # swap layer to test each
    freeze_first_layer=True
)

# Condition 3: FFT kernels injected, fine-tunable (warm start)
model_fft_finetune = UNet(
    in_channels=1, out_channels=1,
    fft_kernels=all_layer_kernels["layer1"],
    freeze_first_layer=False
)


# ============================================================
# TRAINING LOOP (with convergence logging)
# ============================================================

import torch.optim as optim
from torch.utils.data import DataLoader

def dice_loss(pred, target, smooth=1e-6):
    pred   = torch.sigmoid(pred)
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, label="model"):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # respects frozen layers
        lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = dice_loss

    history = {"train_loss": [], "val_iou": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- VALIDATE ---
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds    = model(images)
                val_loss += criterion(preds, masks).item()
                val_iou  += iou_score(preds, masks).item()

        avg_train = total_loss / len(train_loader)
        avg_vloss = val_loss   / len(val_loader)
        avg_viou  = val_iou    / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_vloss)
        history["val_iou"].append(avg_viou)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{label}] Epoch {epoch:03d} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {avg_vloss:.4f} | "
                  f"Val IoU: {avg_viou:.4f}")

    return history


# ============================================================
# CONVERGENCE PLOT ACROSS ALL CONDITIONS
# ============================================================

import matplotlib.pyplot as plt

def plot_convergence(histories: dict, metric="val_iou"):
    """histories = {"Random": h1, "FFT Frozen": h2, "FFT Finetune": h3}"""
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, h), color in zip(histories.items(), colors):
        ax.plot(h[metric], label=label, color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
    ax.set_title("Convergence Comparison: FFT Kernel Init vs Random", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150)
    plt.show()


# ============================================================
# USAGE
# ============================================================

# histories = {}
# histories["Random"]       = train_model(model_random,      train_loader, val_loader, epochs=100, label="Random")
# histories["FFT Frozen"]   = train_model(model_fft_frozen,  train_loader, val_loader, epochs=100, label="FFT Frozen")
# histories["FFT Finetune"] = train_model(model_fft_finetune,train_loader, val_loader, epochs=100, label="FFT Finetune")
# plot_convergence(histories, metric="val_iou")
# plot_convergence(histories, metric="val_loss")