import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from ExperimentData import ExperimentDataset
import torchvision.transforms as transforms
from unet.unet_model import UNet

data = ExperimentDataset(
"olddata\\dataset_1-6-2024\\class1",
"olddata\\dataset_1-6-2024\\class2",
len=10000
)

batchsize = 50
epochs = 10

loader = DataLoader(data, batch_size=batchsize, shuffle=True)

for X, y in loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = UNet(3, 1).to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:  
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(loader, model, loss_fn, optimizer)
print("Done!")
end = time.time()
print("time elapsed: " + (end - start) + " s") 

torch.save(model.state_dict(), "model.pth")
print("Saved Model to model.pth")