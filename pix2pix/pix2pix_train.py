import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import kagglehub

# Download dataset from KaggleHub
download_path = kagglehub.dataset_download("balraj98/cityscapes-pix2pix-dataset")
print("Downloaded dataset path:", download_path)

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")


class CityscapesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Missing dataset directory: {self.data_dir.resolve()}")

        self.files = sorted(
            [
                p for p in self.data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in: {self.data_dir.resolve()}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # Assumes horizontally concatenated pix2pix pair:
        # left = real image, right = segmented image
        real_image = image.crop((0, 0, width // 2, height))
        segmented_image = image.crop((width // 2, 0, width, height))

        if self.transform:
            real_image = self.transform(real_image)
            segmented_image = self.transform(segmented_image)

        return segmented_image, real_image


SIZE = 256
BATCH_SIZE = 4

root_dir = Path(download_path)
train_dir = root_dir / "train"
val_dir = root_dir / "val"

print("cwd:", Path.cwd())
print("train_dir:", train_dir.resolve())
print("val_dir:", val_dir.resolve())

data_transforms = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = CityscapesDataset(train_dir, transform=data_transforms)
val_dataset = CityscapesDataset(val_dir, transform=data_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


def display_sample_pairs(dataset, num_pairs=3, title="Sample Pairs"):
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
    fig.suptitle(title, fontsize=16)

    if num_pairs == 1:
        axes = [axes]

    for i in range(num_pairs):
        segmented_image, real_image = dataset[i]

        segmented_image = segmented_image.permute(1, 2, 0).cpu().numpy()
        real_image = real_image.permute(1, 2, 0).cpu().numpy()

        segmented_image = segmented_image * 0.5 + 0.5
        real_image = real_image * 0.5 + 0.5

        axes[i][0].imshow(segmented_image.clip(0, 1))
        axes[i][0].set_title("Input: Segmented Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(real_image.clip(0, 1))
        axes[i][1].set_title("Target: Real Image")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()


print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

display_sample_pairs(train_dataset, num_pairs=3, title="Sample Pairs from Train Dataset")
display_sample_pairs(val_dataset, num_pairs=3, title="Sample Pairs from Val Dataset")