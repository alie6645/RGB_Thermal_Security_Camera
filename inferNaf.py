import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nafnet.NAFNet_arch import NAFNet
from ExperimentData import ExperimentDataset
import cv2

test_data = ExperimentDataset(
"olddata\\dataset_1-6-2024\\class1",
"olddata\\dataset_1-6-2024\\class2",
len=17000
)

loader = DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NAFNet().to(device)
model.load_state_dict(torch.load("naf.pth", weights_only=True))
print(model)

torch.no_grad()
model.eval()

def minmax(x):
    min, max = x.min(), x.max()
    return (x-min) / (max-min)

rgb, therm = next(iter(loader))
pred = model(rgb)
therm = therm.squeeze()
pred = pred.squeeze()
pred = minmax(pred)
rgbim = rgb[0][0].numpy()/255
cv2.imshow("rgb", rgbim)
thermim = therm.numpy()
cv2.imshow("therm", thermim)
predim = pred.detach().numpy()
print(predim.dtype)
cv2.imshow("pred", predim)
cv2.waitKey(0)
cv2.imwrite("sample\\naf.tiff", predim)




