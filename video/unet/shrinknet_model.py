import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.unet_model import UNet
from unet.unet_parts import *

class ShrinkNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ShrinkNet, self).__init__()
        self.unet = UNet(32, n_classes)
        self.start = (Down(n_channels, 32))

    def forward(self, x):
        x = self.start(x)
        x = self.unet(x)
        return x;