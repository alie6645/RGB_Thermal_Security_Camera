import time
import torch
from torch.utils.data import DataLoader
from ExperimentData import ExperimentDataset
from unet.unet_model import UNet
from unet.shrinknet_model import ShrinkNet
from nafnet.NAFNet_arch import NAFNet

def infer_time(model, loader, iters=64):
    model.eval()
    total = 0
    rgb, therm = next(iter(loader))
    for x in range(iters):
        start = time.time_ns()
        pred = model(rgb)
        end = time.time_ns()
        total = total + end - start
    return total/iters

tiny_data = ExperimentDataset(
"olddata\\tiny\\test\\rgb",
"olddata\\tiny\\test\\therm",
len=100
)

small_data = ExperimentDataset(
"olddata\\small\\rgb",
"olddata\\tiny\\therm",
len=100
)



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.no_grad()

loader = DataLoader(tiny_data, batch_size=16, shuffle=True)
model = NAFNet().to(device)
model.load_state_dict(torch.load("naf.pth", weights_only=True))
nafnet_time = infer_time(model, loader)

model = UNet(3, 1).to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
unet_time = infer_time(model, loader)

loader = DataLoader(small_data, batch_size=16, shuffle=True)
model = ShrinkNet(3, 1).to(device)
model.load_state_dict(torch.load("shrink.pth", weights_only=True))
shrink_time = infer_time(model, loader)

print("nafnet: " + str(nafnet_time) + " ns")
print("unet: " + str(unet_time) + " ns")
print("shrinknet: " + str(shrink_time) + " ns")