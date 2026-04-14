import cv2
import torch
import torchvision.transforms.functional as F
from nafnet.NAFNet_arch import NAFNet
from unet.unet_model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_frame(frame_bgr):
    width = 200
    height = 200

    # Resize to (width x height)
    frame_bgr = cv2.resize(frame_bgr, (200, 200))

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # To tensor in [0,1], shape [3,H,W]
    tensor = F.to_tensor(img_rgb)  # already [0,1] float32

    # Add batch dimension (expected by model):
    return tensor.unsqueeze(0).to(device)

def postprocess_tensor(tensor):
    # Clamp to [0,1] just in case
    tensor = tensor.clamp(0.0, 1.0)

    # Remove batch dimension
    img = tensor.squeeze(0).cpu()

    # To HWC uint8
    img = F.to_pil_image(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

import numpy as np
import os

def main(model, file):
    # Your model
    model = model.to(device).eval()
    model.load_state_dict(torch.load(os.path.join("models", file), map_location=device))

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame)

        with torch.no_grad():
            out = model(inp) 

        out_frame = postprocess_tensor(out)

        cv2.imshow("RGB to Thermal", out_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

import sys

if __name__ == "__main__":
    model_type = sys.argv[1]
    if model_type == 'n':
        main(NAFNet(), "naf.pth")
    elif model_type == 'u':
        main(UNet(3, 1), "unet.pth")
    else:
        print("options: n -> NAFNet, u -> UNet")
