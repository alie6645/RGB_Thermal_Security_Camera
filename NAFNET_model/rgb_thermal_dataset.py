"""
RGB-Thermal Paired Dataset Loader
================================

This module defines a PyTorch Dataset for loading paired RGB and thermal images.

Key Features:
- Automatically matches RGB and thermal images by filename
- Supports resizing to a fixed resolution
- Normalizes data to [-1, 1] (compatible with most neural networks)
- Optional safe geometric augmentation (flip only)
- Ensures aligned transformations between RGB and thermal images

Expected directory structure:
    Dataset/
        rgb_processed/
            0001.png
            0002.png
            ...
        thermal/
            0001.png
            0002.png
            ...

IMPORTANT:
- Filenames must match between RGB and thermal images
- Example: "frame_001.png" ↔ "frame_001.png"
"""

# ----------------------------
# IMPORTS
# ----------------------------

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


# ----------------------------
# HELPER FUNCTION
# ----------------------------

def numeric_sort_key(filename):
    """
    Extract a numeric key from a filename for proper sorting.

    This ensures files like:
        image_2.png comes before image_10.png

    instead of lexicographic order:
        image_10.png comes before image_2.png

    Args:
        filename (str): File name

    Returns:
        int or str: Numeric value if found, otherwise original name
    """
    name = os.path.splitext(filename)[0]

    try:
        # Extract digits and convert to integer
        return int("".join(filter(str.isdigit, name)))
    except ValueError:
        # Fallback if no digits are present
        return name


# ----------------------------
# DATASET CLASS
# ----------------------------

class RGBThermalDataset(Dataset):
    """
    PyTorch Dataset for paired RGB → Thermal image translation.

    Each sample returns:
        (rgb_tensor, thermal_tensor)

    Shapes:
        RGB:     (3, H, W)
        Thermal: (1, H, W)
    """

    def __init__(self, rgb_dir, thermal_dir, target_size=(256, 256), augment=False):
        """
        Initialize dataset.

        Args:
            rgb_dir (str): Path to RGB images
            thermal_dir (str): Path to thermal images
            target_size (tuple): Resize images to (width, height)
            augment (bool): Whether to apply data augmentation
        """

        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.target_size = target_size
        self.augment = augment

        # ----------------------------
        # LOAD FILE LISTS
        # ----------------------------

        rgb_files = os.listdir(rgb_dir)
        thermal_files = os.listdir(thermal_dir)

        # Create mapping: filename (without extension) → thermal file
        thermal_map = {
            os.path.splitext(f)[0]: f
            for f in thermal_files
        }

        # ----------------------------
        # MATCH RGB ↔ THERMAL PAIRS
        # ----------------------------

        self.pairs = []

        for r in rgb_files:
            key = os.path.splitext(r)[0]

            # Only keep files that exist in both folders
            if key in thermal_map:
                self.pairs.append((r, thermal_map[key]))

        # Sort pairs numerically for reproducibility
        self.pairs = sorted(
            self.pairs,
            key=lambda x: numeric_sort_key(x[0])
        )

        # Safety check
        if len(self.pairs) == 0:
            raise ValueError("No RGB-Thermal pairs found!")

        print(f"Loaded {len(self.pairs)} pairs")


    def __len__(self):
        """
        Return number of samples in dataset.
        """
        return len(self.pairs)


    def __getitem__(self, idx):
        """
        Load and process a single RGB-thermal pair.

        Steps:
        1. Load images from disk
        2. Convert color formats
        3. Resize to target size
        4. Normalize to [-1, 1]
        5. Apply optional augmentation
        6. Convert to PyTorch tensors

        Args:
            idx (int): Index of sample

        Returns:
            tuple:
                rgb_tensor (Tensor): shape (3, H, W)
                thermal_tensor (Tensor): shape (1, H, W)
        """

        # Get filenames
        rgb_file, thermal_file = self.pairs[idx]

        # ----------------------------
        # LOAD IMAGES
        # ----------------------------

        # OpenCV loads images in BGR format by default
        rgb = cv2.imread(os.path.join(self.rgb_dir, rgb_file))
        if rgb is None:
            raise ValueError(f"Failed to load RGB: {rgb_file}")

        # Convert BGR → RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load thermal image as grayscale
        thermal = cv2.imread(
            os.path.join(self.thermal_dir, thermal_file),
            cv2.IMREAD_GRAYSCALE
        )
        if thermal is None:
            raise ValueError(f"Failed to load thermal: {thermal_file}")

        # ----------------------------
        # RESIZE
        # ----------------------------

        rgb = cv2.resize(
            rgb,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )

        thermal = cv2.resize(
            thermal,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------
        # NORMALIZATION
        # ----------------------------

        # Convert to float in [0,1]
        rgb = rgb.astype(np.float32) / 255.0
        thermal = thermal.astype(np.float32) / 255.0

        # Scale to [-1, 1]
        rgb = rgb * 2 - 1
        thermal = thermal * 2 - 1

        # ----------------------------
        # DATA AUGMENTATION
        # ----------------------------

        # IMPORTANT:
        # Only geometric transforms are used to preserve pixel alignment
        # between RGB and thermal images.

        if self.augment:

            # Horizontal flip (50% probability)
            if np.random.rand() < 0.5:
                rgb = np.fliplr(rgb).copy()
                thermal = np.fliplr(thermal).copy()

            # Vertical flip (20% probability)
            if np.random.rand() < 0.2:
                rgb = np.flipud(rgb).copy()
                thermal = np.flipud(thermal).copy()

        # ----------------------------
        # MEMORY LAYOUT FIX
        # ----------------------------

        # Ensure arrays are contiguous in memory (required for PyTorch)
        rgb = np.ascontiguousarray(rgb)
        thermal = np.ascontiguousarray(thermal)

        # ----------------------------
        # CONVERT TO TENSORS
        # ----------------------------

        # RGB: (H, W, C) → (C, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)

        # Thermal: (H, W) → (1, H, W)
        thermal_tensor = torch.from_numpy(thermal).unsqueeze(0)

        return rgb_tensor, thermal_tensor