#!/usr/bin/env python3
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import spidev
from smbus2 import SMBus, i2c_msg

from unet.unet_model import UNet

# ============================================================
# CONFIG
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_ROOT = Path("/home/dietpi/thermal/video_runs")
NUM_FRAMES = 90
TARGET_FPS = 3.0

SAVE_INPUT_VIDEO = True
SAVE_OUTPUT_VIDEO = True
SAVE_SIDE_BY_SIDE_VIDEO = True
SAVE_FRAMES = False

MODEL_SIZE = (400, 400)
OV_FLIP_CODE = -1

DEFAULT_SPI_BUS = 4
DEFAULT_SPI_DEV = 1
DEFAULT_SPI_HZ = 500_000
DEFAULT_SPI_MODE = 0
DEFAULT_I2C_BUS = 5
OV5642_I2C_ADDR = 0x3C

KNOWN_GOOD_ARDUCHIP_TIM = 0x00
KNOWN_GOOD_4740 = 0x21

MANUAL_GAIN_HIGH_PROFILE = [
    (0x3503, 0x00),
    (0x350A, 0x00),
    (0x350B, 0x10),
    (0x3500, 0x00),
    (0x3501, 0x10),
    (0x3502, 0x00),
]

ARDUCHIP_TEST1  = 0x00
ARDUCHIP_TIM    = 0x03
ARDUCHIP_FIFO   = 0x04
ARDUCHIP_TRIG   = 0x41
FIFO_SIZE1      = 0x42
FIFO_SIZE2      = 0x43
FIFO_SIZE3      = 0x44

FIFO_CLEAR_MASK     = 0x01
FIFO_START_MASK     = 0x02
FIFO_RDPTR_RST_MASK = 0x10
FIFO_WRPTR_RST_MASK = 0x20
CAP_DONE_MASK       = 0x08

SINGLE_FIFO_READ = 0x3D


# ============================================================
# OUTPUT
# ============================================================

def make_run_dir():
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"ov5642_infer_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================
# MODEL
# ============================================================

def build_model():
    weight_path = os.path.join("400x400", "unet.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    model = UNet(3, 1).to(device).eval()
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)

    print(f"[MODEL] loaded {weight_path} on {device}")
    return model


def preprocess_frame(frame_bgr, size):
    frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(img_rgb)  # [3,H,W], float32 [0,1]
    return tensor.unsqueeze(0).to(device)


def postprocess_tensor(tensor):
    tensor = tensor.clamp(0.0, 1.0)
    img = tensor.squeeze(0).detach().cpu()

    # If model output is [1,H,W], repeat to 3 channels for mp4 writing
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    img = F.to_pil_image(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


# ============================================================
# OV5642 LOW LEVEL
# ============================================================

def spi_write(spi, addr, val):
    spi.xfer2([addr | 0x80, val & 0xFF])


def spi_read(spi, addr):
    return spi.xfer2([addr & 0x7F, 0x00])[1]


def spi_sanity(spi):
    patterns = [0x55, 0xAA, 0x5A]
    for p in patterns:
        spi_write(spi, ARDUCHIP_TEST1, p)
        time.sleep(0.01)
        g = spi_read(spi, ARDUCHIP_TEST1)
        print(f"[OV] SPI TEST1: 0x{p:02X}->0x{g:02X}")
        if g != p:
            return False
    return True


def fifo_length(spi):
    b1 = spi_read(spi, FIFO_SIZE1)
    b2 = spi_read(spi, FIFO_SIZE2)
    b3 = spi_read(spi, FIFO_SIZE3) & 0x7F
    return (b3 << 16) | (b2 << 8) | b1


def fifo_prepare(spi):
    spi_write(spi, ARDUCHIP_FIFO, FIFO_CLEAR_MASK)
    time.sleep(0.005)
    spi_write(spi, ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK)
    time.sleep(0.005)
    spi_write(spi, ARDUCHIP_FIFO, 0x00)
    time.sleep(0.005)
    spi_write(spi, ARDUCHIP_FIFO, FIFO_CLEAR_MASK)
    time.sleep(0.005)
    spi_write(spi, ARDUCHIP_FIFO, 0x00)
    time.sleep(0.005)


def fifo_start(spi):
    spi_write(spi, ARDUCHIP_FIFO, FIFO_START_MASK)


def poll_capture_done(spi, timeout_s=8.0):
    t0 = time.time()
    saw_low = False

    while time.time() - t0 < 2.0:
        trig = spi_read(spi, ARDUCHIP_TRIG)
        if (trig & CAP_DONE_MASK) == 0:
            saw_low = True
            break
        time.sleep(0.01)

    if not saw_low:
        return False, spi_read(spi, ARDUCHIP_TRIG), "CAP_DONE never went low after prepare"

    t1 = time.time()
    while time.time() - t1 < timeout_s:
        trig = spi_read(spi, ARDUCHIP_TRIG)
        if trig & CAP_DONE_MASK:
            return True, trig, None
        time.sleep(0.01)

    return False, spi_read(spi, ARDUCHIP_TRIG), "Capture never completed"


def read_fifo_single_command_chunks(spi, nbytes):
    out = bytearray()
    chunk = 4095
    remaining = nbytes

    while remaining > 0:
        k = min(chunk, remaining)
        rx = spi.xfer2([SINGLE_FIFO_READ] + [0x00] * k)
        out.extend(rx[1:])
        remaining -= k
        time.sleep(0.0005)

    return bytes(out)


def i2c_write16(bus, dev, reg, val):
    msg = i2c_msg.write(dev, [(reg >> 8) & 0xFF, reg & 0xFF, val & 0xFF])
    bus.i2c_rdwr(msg)


def i2c_read16(bus, dev, reg):
    write = i2c_msg.write(dev, [(reg >> 8) & 0xFF, reg & 0xFF])
    read = i2c_msg.read(dev, 1)
    bus.i2c_rdwr(write, read)
    return list(read)[0]


def write_reg_list_16(bus, dev, regs, delay_every=0):
    n = 0
    for reg, val in regs:
        if reg == 0xFFFF and val == 0xFF:
            break
        i2c_write16(bus, dev, reg, val)
        n += 1
        if delay_every and (n % delay_every == 0):
            time.sleep(0.002)


def ov5642_detect(bus):
    high = i2c_read16(bus, OV5642_I2C_ADDR, 0x300A)
    low = i2c_read16(bus, OV5642_I2C_ADDR, 0x300B)
    return high, low


def ov5642_soft_reset(bus):
    i2c_write16(bus, OV5642_I2C_ADDR, 0x3008, 0x80)
    time.sleep(0.15)


def ov5642_init_jpeg_640x480(bus):
    from ov5642_regs import OV5642_JPEG_INIT, OV5642_640x480_JPEG

    if len(OV5642_JPEG_INIT) <= 1 or len(OV5642_640x480_JPEG) <= 1:
        raise RuntimeError("ov5642_regs.py still contains placeholder tables.")

    ov5642_soft_reset(bus)
    write_reg_list_16(bus, OV5642_I2C_ADDR, OV5642_JPEG_INIT, delay_every=32)
    time.sleep(0.05)
    write_reg_list_16(bus, OV5642_I2C_ADDR, OV5642_640x480_JPEG, delay_every=32)
    time.sleep(0.05)


def ov5642_apply_transport_fix(bus):
    i2c_write16(bus, OV5642_I2C_ADDR, 0x4740, KNOWN_GOOD_4740)
    time.sleep(0.02)


def ov5642_apply_manual_gain_high(bus):
    for reg, val in MANUAL_GAIN_HIGH_PROFILE:
        i2c_write16(bus, OV5642_I2C_ADDR, reg, val)
        time.sleep(0.01)


def extract_jpeg(payload: bytes):
    soi = payload.find(b"\xFF\xD8")
    if soi < 0:
        return None, -1, -1

    eoi = payload.find(b"\xFF\xD9", soi + 2)
    if eoi < 0:
        return None, soi, -1

    return payload[soi:eoi + 2], soi, eoi


def ov5642_open_and_init():
    spi = spidev.SpiDev()
    spi.open(DEFAULT_SPI_BUS, DEFAULT_SPI_DEV)
    spi.mode = DEFAULT_SPI_MODE
    spi.max_speed_hz = DEFAULT_SPI_HZ

    if not spi_sanity(spi):
        spi.close()
        raise RuntimeError("SPI sanity failed.")

    spi_write(spi, ARDUCHIP_TIM, KNOWN_GOOD_ARDUCHIP_TIM)

    with SMBus(DEFAULT_I2C_BUS) as bus:
        high, low = ov5642_detect(bus)
        if (high, low) != (0x56, 0x42):
            spi.close()
            raise RuntimeError(f"Wrong sensor ID: 0x{high:02X} 0x{low:02X}")

        ov5642_init_jpeg_640x480(bus)
        ov5642_apply_transport_fix(bus)
        ov5642_apply_manual_gain_high(bus)

    return spi


def capture_ov5642_frame(spi):
    fifo_prepare(spi)
    fifo_start(spi)

    done, trig, err = poll_capture_done(spi, timeout_s=8.0)
    flen = fifo_length(spi)

    if not done:
        raise RuntimeError(err or f"Capture failed, TRIG=0x{trig:02X}")
    if flen == 0:
        raise RuntimeError("FIFO length is 0.")

    payload = read_fifo_single_command_chunks(spi, flen)

    jpeg, soi, eoi = extract_jpeg(payload)
    if jpeg is None:
        raise RuntimeError(f"No complete JPEG found. SOI={soi}, EOI={eoi}")

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode OV JPEG.")

    img = cv2.flip(img, OV_FLIP_CODE)
    return img


# ============================================================
# MAIN
# ============================================================

def main():
    run_dir = make_run_dir()
    frames_in_dir = run_dir / "input_frames"
    frames_out_dir = run_dir / "output_frames"
    frames_in_dir.mkdir(exist_ok=True)
    frames_out_dir.mkdir(exist_ok=True)

    model = build_model()
    spi = ov5642_open_and_init()

    out_w, out_h = MODEL_SIZE

    input_writer = None
    output_writer = None
    side_writer = None

    frame_period = 1.0 / TARGET_FPS
    next_deadline = time.perf_counter()

    print(f"[RUN] saving to {run_dir}")

    try:
        for i in range(NUM_FRAMES):
            t0 = time.perf_counter()

            frame = capture_ov5642_frame(spi)

            # standardize incoming frame to 400x400 before inference
            inp = preprocess_frame(frame, MODEL_SIZE)

            with torch.no_grad():
                out = model(inp)

            out_frame = postprocess_tensor(out)

            # standardize saved rgb frame to 400x400 too
            in_frame = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_LINEAR)
            side = np.concatenate([in_frame, out_frame], axis=1)

            if SAVE_FRAMES:
                cv2.imwrite(str(frames_in_dir / f"frame_{i:04d}.jpg"), in_frame)
                cv2.imwrite(str(frames_out_dir / f"pred_{i:04d}.png"), out_frame)

            if input_writer is None and SAVE_INPUT_VIDEO:
                input_writer = cv2.VideoWriter(
                    str(run_dir / "input_rgb.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    TARGET_FPS,
                    (out_w, out_h),
                )

            if output_writer is None and SAVE_OUTPUT_VIDEO:
                output_writer = cv2.VideoWriter(
                    str(run_dir / "predicted_thermal.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    TARGET_FPS,
                    (out_w, out_h),
                )

            if side_writer is None and SAVE_SIDE_BY_SIDE_VIDEO:
                side_writer = cv2.VideoWriter(
                    str(run_dir / "side_by_side.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    TARGET_FPS,
                    (out_w * 2, out_h),
                )

            if input_writer is not None:
                input_writer.write(in_frame)

            if output_writer is not None:
                output_writer.write(out_frame)

            if side_writer is not None:
                side_writer.write(side)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[{i+1:03d}/{NUM_FRAMES}] {dt_ms:.1f} ms")

            next_deadline += frame_period
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_deadline = time.perf_counter()

    finally:
        try:
            spi.close()
        except Exception:
            pass

        if input_writer is not None:
            input_writer.release()
        if output_writer is not None:
            output_writer.release()
        if side_writer is not None:
            side_writer.release()

    print("[DONE]")


if __name__ == "__main__":
    main()