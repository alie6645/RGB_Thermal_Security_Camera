import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet.unet_model import UNet
from unet.shrinknet_model import ShrinkNet
from nafnet.NAFNet_arch import NAFNet
from ExperimentData import ExperimentDataset
import cv2
import random

test_data = ExperimentDataset(
"olddata/medium/rgb",
"olddata/medium/therm",
len=1000
)

shrink_data = ExperimentDataset(
"olddata/full/rgb",
"olddata/medium/therm",
len=1000
)

loader = DataLoader(test_data, batch_size=1, shuffle=False)
shrinkloader = DataLoader(shrink_data, batch_size=1, shuffle=False)

device = "cpu"
print(f"Using {device} device")

def minmax(x):
    min, max = x.min(), x.max()
    return (x-min) / (max-min)

def display(model, rgb, func, name):
    pred = model(rgb)
    pred = pred.squeeze()
    pred = func(pred)
    predim = pred.detach().numpy()
    cv2.imshow(name, predim)
    cv2.imwrite("sample/" + name + ".jpg", predim * 255)

it = iter(loader)
shrit = iter(shrinkloader)
num = random.randint(0, 1000);
print("using photo: " + str(num))
for x in range(num):
    rgb, therm = next(it)
    big_rgb, big_therm = next(shrit)

torch.no_grad()

model = UNet(3, 1).to(device)
model.eval()
model.load_state_dict(torch.load("model.pth", weights_only=True))
display(model, rgb, lambda x: x, "unet")

model = ShrinkNet(3, 1).to(device)
model.eval()
model.load_state_dict(torch.load("shrink.pth", weights_only=True))
display(model, big_rgb, lambda x: x, "shrinknet")

model = NAFNet().to("cpu")
model.eval()
model.load_state_dict(torch.load("naf.pth", weights_only=True))
display(model, rgb, minmax, "nafnet")

therm = therm.squeeze()
rgbim = rgb[0][0].numpy()/255
cv2.imshow("rgb", rgbim)
thermim = therm.numpy()
cv2.imshow("therm", thermim)
cv2.imwrite("sample/rgb.jpg", rgbim * 255)
cv2.imwrite("sample/therm.jpg", thermim * 255)
cv2.waitKey(0)
