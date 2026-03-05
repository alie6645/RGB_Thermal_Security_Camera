import os
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

class ExperimentDataset(Dataset):
    def __init__(self, rgb_dir, therm_dir, transform=None, target_transform=None, len=1):
        self.rgb_dir = rgb_dir
        self.therm_dir = therm_dir
        self.transform = transform
        self.target_transform = target_transform
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, str(idx) + ".jpg")
        therm_path = os.path.join(self.therm_dir, str(idx) +  ".jpg")
        rgbimage = decode_image(rgb_path)
        label = decode_image(therm_path, mode="GRAY")
        if self.transform:
            rgbimage = self.transform(rgbimage)
        if self.target_transform:
            label = self.target_transform(label)
        rgbimage = rgbimage.float()
        label = label.float()
        return rgbimage, label