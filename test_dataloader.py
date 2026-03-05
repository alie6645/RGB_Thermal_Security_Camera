from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy
from ExperimentData import ExperimentDataset

data = ExperimentDataset(
"olddata\\tiny\\rgb",
"olddata\\tiny\\therm"
)

loader = DataLoader(data, batch_size=10, shuffle=True)

rgb, therm = next(iter(loader))
rgb = rgb.squeeze()
rgbarr = rgb[0][0].numpy()
therm = therm.squeeze()
thermarr = therm[0].numpy()

cv2.imshow("rgb", rgbarr)
cv2.waitKey(0)
cv2.imshow("therm", thermarr)
cv2.waitKey(0)

