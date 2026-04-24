Contains scripts used for RGB to thermal neural network translation \[paper](https://www.overleaf.com/read/xwymdkgvqxsc#39ed2b)

# RGB Thermal Security Camera

This project explores RGB-to-thermal image translation for low-light security camera applications. The goal is to take normal RGB camera frames and generate thermal-like grayscale predictions that make people and warm objects easier to detect in dark scenes.

The repository contains several experimental pipelines:

- TensorFlow pix2pix-style RGB → thermal training
- PyTorch NAFNet / UNet inference scripts
- Camera capture and preprocessing scripts
 

The project was developed using paired RGB and thermal images captured from a custom camera setup.

---

## Project Goal

Low-light RGB cameras often fail to clearly show people, animals, or objects in dark environments. Thermal cameras are better for this, but they are more expensive and lower resolution.

This project investigates whether a neural network can learn a mapping from RGB camera images to thermal-like images using paired RGB/thermal training data.

The intended use case is a security camera system that can:

1. Capture RGB images from a normal camera.
2. Predict a thermal-like image using a trained model.
3. Highlight human silhouettes or warm objects in low-light scenes.

# Operating Camera
Need to open camera to charge battery and to access hdmi port and usb c port for powering if battery is unplugged.
Once attached to hdmi, use dietpi-config command to access network adapter settings to setup wifi connect. alternatively, can squeeze ethernet cable into adapter slot and plug into router.
Next, obtain ip address from router or hotspot and run "ssh dietpi@{ip address}"

default credentials are username: dietpi, password: thermal.

can begin collecting data but using command .~/thermal/sync_capture_200.sh shell script.
can run demo script using .~/thermal/video_to_thermal.sh shell script.

can copy off files over the network from the orange pi by using "scp -r dietpi@{ip address}:/path/to/folder C:/path/on/windows/to/folder"
---

## Repository Structure

```text
RGB_Thermal_Security_Camera/
├── pix2pix/
│   ├── preprocessing_images.py
│   ├── pix2pix_train.py
│   └── pix2pix_test.py
│
├── video/
│   └── web cam video models
│
├── camera_scripts/
│   ├── Scripts that live on the orange pi 5 camera platform.
    ├── preprocessing_images.py - run across dataset of image pairs in order to do preprocessing with alignment and scaling to account for FOV and angle differences
│   ├── ov5642_regs.py - register values for ov5642 camera module for SPI and i2c communication to retrieve image frames
│   ├── sync_capture_200.py - main data capture script, can be run with sync_capture_200.sh on the camera in order to capture 200 images, 2 seconds to take each image.
│   └── video_to_thermal.py - script to capture video feed from ov5642 arducam module and run each frame through an inference module trained using scripts in this repo.
                              generates .avi video file with the inferenced thermal video. can be adjusted to different models, but runs 200x200 nafnet model since thats the best perfomance for the orange pi.
├── nafnet/
│   └── NAFNet model experiments
│
├── NAFNET_model/
│   └── NAFNet model code / checkpoints
│
└─── unet/
    └── UNet model experiments



samples: example results



video: code for video demos

