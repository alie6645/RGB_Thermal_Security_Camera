import cv2
import torch
import numpy as np
from torchvision import transforms

def process_video_with_model(model, input_path, output_path, device="cpu"):
    """
    Reads an MP4, crops to square then resizes to 400 x 400,
    runs inference (model outputs 1-channel grayscale 0–1),
    and writes a grayscale MP4.
    """

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer (grayscale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (400, 400), isColor=False)

    preprocess = transforms.Compose([
        transforms.ToTensor(),   # HWC → CHW, float32 0–1
    ])

    crop_size = min(width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----- Center crop -----
        h, w, _ = frame.shape
        cropped = frame[0: crop_size, 0: crop_size]
        cropped = cv2.resize(cropped, (400, 400))

        # BGR → RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # To tensor
        tensor = preprocess(rgb).unsqueeze(0).to(device)  # [1,3,400,400]

        with torch.no_grad():
            output = model(tensor)  # [1,1,400,400], values 0–1

        # Convert to numpy grayscale
        gray = output.squeeze().cpu().numpy()  # [400,400]

        # Scale to 0–255
        gray = np.clip(gray, 0, 1)
        gray = (gray * 255).astype("uint8")



        # Write frame
        out.write(gray)

    cap.release()
    out.release()

from nafnet.NAFNet_arch import NAFNet
from unet.unet_model import UNet

models = {
    "nafnet" : (NAFNet(), "models/400x400/naf.pth"),
    "nafnet50" : (NAFNet(), "models/naf50.pth"),
    "nafnet100" : (NAFNet(), "models/naf100.pth"),
    "unet"   : (UNet(3, 1), "models/unet.pth")
}

import sys
if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    model_param = models[sys.argv[3]]
    model = model_param[0]
    model.load_state_dict(torch.load(model_param[1], map_location=torch.device("cpu")))
    process_video_with_model(model, input, output)
