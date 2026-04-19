"""
RGB → Thermal Image Translation Training Script
==============================================

This script trains a NAFNet-based convolutional neural network to convert
RGB images into thermal images.

Key features:
- Lightweight NAFNet architecture
- Mixed precision training (AMP)
- Composite loss: L1 + SSIM + gradient consistency
- Best model checkpoint saving

Dependencies:
- PyTorch
- pytorch-msssim
- tqdm

Dataset:
- Expects paired RGB and thermal images
"""

# ----------------------------
# IMPORTS
# ----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from multiprocessing import freeze_support
from pytorch_msssim import SSIM

# Custom dataset (must return: rgb, thermal)
from NAFNET_model.rgb_thermal_dataset import RGBThermalDataset


# ----------------------------
# MODEL BUILDING BLOCK
# ----------------------------

class NAFBlock(nn.Module):
    """
    NAFBlock (Nonlinear Activation Free-inspired block)

    A simplified residual block that:
    - Applies two convolutions
    - Uses channel-wise attention (gating)
    - Scales residual with a learnable parameter (beta)

    This design improves stability and efficiency compared to traditional ResNet blocks.
    """

    def __init__(self, c):
        """
        Args:
            c (int): Number of input/output channels
        """
        super().__init__()

        # Two standard convolution layers
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        # Channel attention / gating mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(c, c // 4, 1),  # Channel reduction
            nn.GELU(),
            nn.Conv2d(c // 4, c, 1),  # Channel expansion
            nn.Sigmoid()              # Attention weights [0,1]
        )

        # Activation function
        self.act = nn.GELU()

        # Learnable residual scaling (starts at 0 for stability)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass with residual connection

        Args:
            x (Tensor): Input feature map

        Returns:
            Tensor: Output feature map
        """

        # Apply conv → activation → conv
        y = self.act(self.conv1(x))
        y = self.conv2(y)

        # Apply channel-wise gating
        y = y * self.gate(y)

        # Residual connection with learnable scaling
        return x + self.beta * y


# ----------------------------
# FULL MODEL
# ----------------------------

class NAFNet(nn.Module):
    """
    NAFNet model for image-to-image translation.

    Structure:
    - Stem: initial feature extraction
    - Body: stack of NAFBlocks
    - Head: output projection
    """

    def __init__(self, in_channels=3, out_channels=1, width=96, depth=20):
        """
        Args:
            in_channels (int): Input channels (RGB = 3)
            out_channels (int): Output channels (thermal = 1)
            width (int): Feature width
            depth (int): Number of NAFBlocks
        """
        super().__init__()

        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1)
        )

        # Core network body
        self.blocks = nn.Sequential(
            *[NAFBlock(width) for _ in range(depth)]
        )

        # Output layer
        self.head = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

        # NOTE: No tanh or sigmoid applied
        # Output range is handled via loss + optional clamping

    def forward(self, x):
        """
        Forward pass

        Args:
            x (Tensor): RGB input image

        Returns:
            Tensor: Predicted thermal image
        """

        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


# ----------------------------
# LOSS FUNCTIONS
# ----------------------------

def gradient_loss(pred, target):
    """
    Gradient consistency loss.

    Encourages predicted image gradients to match target gradients,
    improving edge sharpness and structural detail.

    Args:
        pred (Tensor): Predicted image
        target (Tensor): Ground truth image

    Returns:
        Tensor: Scalar loss value
    """

    # Compute horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Compute vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # L1 difference between gradients
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


# ----------------------------
# TRAINING SCRIPT
# ----------------------------

if __name__ == "__main__":

    # Required for Windows multiprocessing (DataLoader workers)
    freeze_support()

    # Select device (GPU if available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    # ----------------------------
    # DATASET & DATALOADER
    # ----------------------------

    dataset = RGBThermalDataset(
        rgb_dir="../Dataset/rgb_processed",
        thermal_dir="../Dataset/thermal",
        target_size=(256, 256),
        augment=True  # Data augmentation enabled
    )

    loader = DataLoader(
        dataset,
        batch_size=8,      # Reduced for memory stability
        shuffle=True,
        num_workers=4,
        pin_memory=True   # Faster GPU transfer
    )

    # ----------------------------
    # MODEL & TRAINING SETUP
    # ----------------------------

    model = NAFNet().to(DEVICE)

    # Structural similarity loss (perceptual quality)
    ssim_fn = SSIM(
        data_range=2.0,   # Because outputs are clamped to [-1, 1]
        size_average=True,
        channel=1
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        weight_decay=1e-4
    )

    # Learning rate scheduler (cosine annealing with restarts)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Mixed precision scaler (for faster + stable training on GPU)
    scaler = torch.amp.GradScaler("cuda")

    # Track best model
    best = float("inf")

    EPOCHS = 50

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        # Progress bar
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for rgb, thermal in loop:

            # Move data to device
            rgb = rgb.to(DEVICE)
            thermal = thermal.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda"):

                pred = model(rgb)

                # Clamp output for stability
                pred = torch.clamp(pred, -1, 1)

                # --- Loss components ---
                l1 = F.l1_loss(pred, thermal)                 # Pixel loss
                ssim_loss = 1 - ssim_fn(pred, thermal)        # Perceptual loss
                grad = gradient_loss(pred, thermal)           # Edge consistency

                # Weighted total loss
                loss = 0.6 * l1 + 0.2 * ssim_loss + 0.2 * grad

            # Backpropagation (scaled for AMP)
            scaler.scale(loss).backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Step scheduler AFTER each epoch
        scheduler.step()

        # Compute average loss
        avg_loss = running_loss / len(loader)
        print(f"Epoch avg: {avg_loss:.5f}")

        # Save best model
        if avg_loss < best:
            best = avg_loss
            torch.save(model.state_dict(), "../best_rgb2thermal.pth")
            print("Saved best model")

    print("Training complete.")