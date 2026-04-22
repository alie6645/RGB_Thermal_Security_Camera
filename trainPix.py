import time
import torch
from torch.utils.data import DataLoader
import Datasets
from pix2pix.model import Pix2Pix

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data = Datasets.data_bright_train
batchsize = 1
epochs = 10

train_loader = DataLoader(data, batch_size=batchsize, shuffle=True)

model = Pix2Pix(in_channels=3, out_channels=1, device=device)
model.setup()

# -------------------------
# Training Loop
# -------------------------
def train(img_dataloader, model):
    for idx, (x, y) in enumerate(img_dataloader):
        model.set_input(x, y)
        model.optimize_parameters()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    model.netG.to(model.device)
    model.netD.to(model.device)

    train(train_loader, model)

    model.save_networks(epoch)

print("Done!")
