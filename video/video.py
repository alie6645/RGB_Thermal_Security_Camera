import cv2
import torch
import torchvision.transforms.functional as F
from nafnet.NAFNet_arch import NAFNet
from unet.unet_model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_frame(frame_bgr, size):
    width = 200
    height = 200

    # Resize to (width x height)
    frame_bgr = cv2.resize(frame_bgr, size)

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

def main(model, file, size):
    # Your model
    model = model.to(device).eval()
    size_name = str(size[0]) + "x" + str(size[1])
    model.load_state_dict(torch.load(os.path.join("models", size_name, file), map_location=device))

    cap = cv2.VideoCapture(0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_size = min(width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Square crop
        h, w, _ = frame.shape
        frame = frame[0: crop_size, 0: crop_size]

        inp = preprocess_frame(frame, size)

        with torch.no_grad():
            out = model(inp)

        out_frame = postprocess_tensor(out)

        # --- SIDE BY SIDE VIEW ---
        h = min(frame.shape[0], out_frame.shape[0])
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
        out_resized = cv2.resize(out_frame, (int(out_frame.shape[1] * h / out_frame.shape[0]), h))

        combined = np.hstack((frame_resized, out_resized))

        cv2.imshow("Original (Left) | Thermal (Right)", combined)
        # --------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: " + sys.argv[0] + " model_type[\'n\' or \'u\'] model_size[\'s\' or \'l\']")
        sys.exit()
    
    model_type = sys.argv[1]
    if model_type == 'n':
        model_type = NAFNet()
        file = "naf.pth"
    elif model_type == 'u':
        model_type = UNet(3, 1)
        file = "unet.pth"
    else:
        print("model type options: n -> NAFNet, u -> UNet")
    
    model_size = sys.argv[2]
    if model_size == 's':
        model_size = (200, 200)
    elif model_size == 'l':
        model_size = (400, 400)
    else:
        print("model size options: s -> 200 x 200, l -> 400 x 400")

    main(model_type, file, model_size)
