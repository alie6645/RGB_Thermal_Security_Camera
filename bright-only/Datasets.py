from ExperimentData import ExperimentDataset
from torchvision.transforms import v2

transforms = v2.RandomHorizontalFlip(1)

# Expects directories to contain files named "0.jpg", "1.jpg", ... for rgb
# or "0.png", "1.png", ... for thermal

rgb_train_dir = "/var/tmp/u1447122/rgb_bright_processed"
therm_train_dir = "/var/tmp/u1447122/therm_bright_processed"
train_len = 2015

rgb_test_dir = "/var/tmp/u1447122/rgb_bright_processed/test"
therm_test_dir = "/var/tmp/u1447122/therm_bright_processed/test"
test_len = 485

data_bright_train = ExperimentDataset(
rgb_train_dir,
therm_train_dir,
len=train_len,
transform=transforms,
target_transform=transforms
)

data_bright_test = ExperimentDataset(
rgb_train_dir,
therm_train_dir,
len=test_len,
transform=transforms,
target_transform=transforms
)
