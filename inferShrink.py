import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet.shrinknet_model import ShrinkNet
from ExperimentData import ExperimentDataset
import cv2

test_data = ExperimentDataset(
"olddata\\small\\test\\rgb",
"olddata\\tiny\\test\\therm",
len=1
)

loader = DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = ShrinkNet(3, 1).to(device)
model.load_state_dict(torch.load("shrink.pth", weights_only=True))

torch.no_grad()
model.eval()

rgb, therm = next(iter(loader))
pred = model(rgb)
pred = torch.reshape(pred, (1, 1, 32, 32))
therm = therm.squeeze()
pred = pred.squeeze()
pred = F.sigmoid(pred)
rgbim = cv2.resize(rgb[0][0].numpy()/255, (320, 320))
cv2.imshow("rgb", rgbim)
thermim = cv2.resize(therm.numpy(), (320, 320))
cv2.imshow("therm", thermim)
predim = cv2.resize(pred.detach().numpy(), (320, 320))
cv2.imshow("pred", predim)
cv2.waitKey(0)
cv2.imwrite("sample\\therm.tiff", thermim)
cv2.imwrite("sample\\shrink.tiff", predim)


