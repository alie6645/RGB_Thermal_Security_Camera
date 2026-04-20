"""
RGB → Thermal Evaluation & Visualization Script
==============================================

This script evaluates a trained NAFNet model by:
1. Running inference on a held-out test split
2. Generating visual comparisons for selected samples

Each output image contains 4 panels:
    [ RGB | Ground Truth Thermal | Predicted (Grayscale) | Predicted (Colored) ]

Features:
- Automatic dataset split (80/20)
- GPU acceleration (if available)
- Thermal normalization for visualization
- Colormap-based heatmap rendering

Dependencies:
- PyTorch
- NumPy
- Pillow (PIL)
- matplotlib

Expected structure:
- Dataset/rgb_processed/
- Dataset/thermal/
- best_rgb2thermal.pth
"""

# ----------------------------
# IMPORTS
# ----------------------------

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import random_split
import matplotlib.cm as cm  # Provides colormaps for heatmap visualization

# Custom modules
from rgb_thermal_dataset import RGBThermalDataset
from train import NAFNet


# ----------------------------
# CONFIGURATION
# ----------------------------

# Colormap used for predicted thermal visualization
# Recommended: "inferno" (most realistic thermal look)
COLORMAP = "inferno"  # Alternatives: magma, plasma, viridis, jet

# Select compute device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------------
# DATASET PREPARATION
# ----------------------------

# Load full dataset (NO augmentation for evaluation)
full_dataset = RGBThermalDataset(
    rgb_dir="../Dataset/rgb_processed",
    thermal_dir="../Dataset/thermal",
    augment=False
)

# Split dataset into train/test (80/20 split)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Only use test set for evaluation
_, test_dataset = random_split(full_dataset, [train_size, test_size])
dataset = test_dataset


# ----------------------------
# MODEL LOADING
# ----------------------------

# Initialize model architecture
model = NAFNet().to(device)

# Load trained weights
model.load_state_dict(
    torch.load("../best_rgb2thermal.pth", map_location=device, weights_only=True)
)

# Set model to evaluation mode (disables dropout, etc.)
model.eval()


# ----------------------------
# OUTPUT DIRECTORY
# ----------------------------

# Directory where visual results will be saved
out_dir = "../evaluation_results"
os.makedirs(out_dir, exist_ok=True)


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def to_uint8(img):
    """
    Convert floating-point image in [0,1] range to uint8 [0,255].

    Args:
        img (np.ndarray): Image with values in [0,1]

    Returns:
        np.ndarray: uint8 image
    """
    img = np.clip(img, 0, 1)  # Ensure valid range
    return (img * 255).astype(np.uint8)


def normalize(img):
    """
    Min-max normalize an image to [0,1].

    This is important for visualization since thermal values
    may not already be in a displayable range.

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Normalized image
    """
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def apply_colormap(img, cmap_name="inferno"):
    """
    Convert a grayscale image into a colored heatmap.

    Args:
        img (np.ndarray): Normalized image [0,1]
        cmap_name (str): Name of matplotlib colormap

    Returns:
        np.ndarray: RGB image (uint8)
    """
    cmap = cm.get_cmap(cmap_name)

    # Apply colormap → returns RGBA, so discard alpha channel
    colored = cmap(img)[:, :, :3]

    return to_uint8(colored)


# ----------------------------
# SAMPLE SELECTION
# ----------------------------

# Number of samples to visualize
num_to_show = 10

dataset_size = len(dataset)

# Evenly sample indices across dataset
indices = np.linspace(0, dataset_size - 1, num_to_show).astype(int).tolist()

print("Sampling indices:", indices)


# ----------------------------
# INFERENCE LOOP
# ----------------------------

for i, idx in enumerate(indices):

    # Get one sample (RGB input + thermal ground truth)
    rgb, thermal_gt = dataset[idx]

    # Add batch dimension and move to device
    rgb_input = rgb.unsqueeze(0).to(device)

    # Run model inference (no gradients needed)
    with torch.no_grad():
        pred = model(rgb_input)[0].cpu()

    # ----------------------------
    # CONVERT TENSORS → NUMPY
    # ----------------------------

    # RGB: (C,H,W) → (H,W,C)
    rgb_img = rgb.detach().cpu().permute(1, 2, 0).numpy()

    # Thermal images: remove channel dimension
    gt_img = thermal_gt.squeeze().numpy()
    pred_img = pred.squeeze().numpy()

    # ----------------------------
    # NORMALIZATION
    # ----------------------------

    # Normalize thermal images for visualization
    gt_img = normalize(gt_img)
    pred_img = normalize(pred_img)

    # ----------------------------
    # CONVERT TO IMAGE FORMAT
    # ----------------------------

    # RGB image
    rgb_img = to_uint8(rgb_img)
    rgb_pil = Image.fromarray(rgb_img)

    # Ground truth thermal (grayscale → RGB for consistent display)
    gt_gray = to_uint8(gt_img)
    gt_pil = Image.fromarray(gt_gray).convert("RGB")

    # Predicted thermal (grayscale)
    pred_gray = to_uint8(pred_img)
    pred_gray_pil = Image.fromarray(pred_gray).convert("RGB")

    # Predicted thermal (colored heatmap)
    pred_color = apply_colormap(pred_img, COLORMAP)
    pred_color_pil = Image.fromarray(pred_color)

    # ----------------------------
    # CREATE VISUAL GRID
    # ----------------------------

    # Each image has same width/height
    w, h = rgb_pil.size

    # Create blank canvas: 4 images side-by-side
    grid = Image.new("RGB", (w * 4, h))

    # Paste images in order
    grid.paste(rgb_pil, (0, 0))              # Original RGB input
    grid.paste(gt_pil, (w, 0))               # Ground truth thermal
    grid.paste(pred_gray_pil, (w * 2, 0))    # Predicted (grayscale)
    grid.paste(pred_color_pil, (w * 3, 0))   # Predicted (colored)

    # ----------------------------
    # SAVE RESULT
    # ----------------------------

    save_path = os.path.join(out_dir, f"sample_{i:02d}_idx_{idx}.png")
    grid.save(save_path)

    print("Saved:", save_path)


print("Evaluation complete.")