from ExperimentData import ExperimentDataset
from torchvision.transforms import v2

transforms = v2.RandomHorizontalFlip(1)

data_shrink_train = ExperimentDataset(
"/var/tmp/u1447122/rgb",
"/var/tmp/u1447122/thermal",
len=200,
transform=transforms,
target_transform=transforms
)

data_same_train = ExperimentDataset(
"/var/tmp/u1447122/mrgb/train",
"/var/tmp/u1447122/mtherm/train",
len=1300,
transform=transforms,
target_transform=transforms
)

data_same_test = ExperimentDataset(
"/var/tmp/u1447122/mrgb",
"/var/tmp/u1447122/mtherm",
len=290,
transform=transforms,
target_transform=transforms
)

data_processed_train = ExperimentDataset(
"/var/tmp/u1447122/rgb_processed",
"/var/tmp/u1447122/thermal_processed",
len=1100,
transform=transforms,
target_transform=transforms
)

data_processed_test = ExperimentDataset(
"/var/tmp/u1447122/rgb_processed/test",
"/var/tmp/u1447122/thermal_processed/test",
len=200,
transform=transforms,
target_transform=transforms
)

data_processed_small_train = ExperimentDataset(
"/var/tmp/u1447122/rgb_processed_small",
"/var/tmp/u1447122/therm_processed_small",
len=1100,
transform=transforms,
target_transform=transforms
)

data_processed_small_test = ExperimentDataset(
"/var/tmp/u1447122/rgb_processed_small/test",
"/var/tmp/u1447122/therm_processed_small/test",
len=200,
transform=transforms,
target_transform=transforms
)
