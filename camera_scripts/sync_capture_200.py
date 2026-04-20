#!/usr/bin/env python3
import time
import cv2
import numpy as np
import spidev

from pathlib import Path
from threading import Thread, Condition
from smbus2 import SMBus, i2c_msg

from seekcamera import (
    SeekCameraIOType,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("/home/dietpi/thermal/data")

PAIR_COUNT = 200                 # number of paired captures
SYNC_DELAY_S = 2.0               # delay before each synchronized trigger
INTERVAL_BETWEEN_PAIRS_S = 0.5   # extra pause after each pair

SEEK_FILENAME = "seek.png"
OV_JPG_FILENAME = "ov5642.jpg"
OV_RAW_FILENAME = "ov5642.raw"

# OV flip options:
#   0  = vertical flip
#   1  = horizontal flip
#  -1  = both directions (180-degree rotation)
OV_FLIP_CODE = -1

# OV5642 / ArduCam defaults
DEFAULT_SPI_BUS = 4
DEFAULT_SPI_DEV = 1
DEFAULT_SPI_HZ = 500_000
DEFAULT_SPI_MODE = 0
DEFAULT_I2C_BUS = 5
OV5642_I2C_ADDR = 0x3C

KNOWN_GOOD_ARDUCHIP_TIM = 0x00
KNOWN_GOOD_4740 = 0x21

MANUAL_GAIN_HIGH_PROFILE = [
    (0x3503, 0x00),  # manual exposure + manual gain
    (0x350A, 0x00),  # gain = 1x
    (0x350B, 0x10),
    (0x3500, 0x00),  # exposure = ~1/2000s
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
# COMMON
# ============================================================
def wait_until(target_t):
    while True:
        now = time.perf_counter()
        dt = target_t - now
        if dt <= 0:
            return
        if dt > 0.01:
            time.sleep(dt / 2)
        else:
            pass


def make_session_dir():
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = DATA_ROOT / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def make_pair_dir(session_dir: Path, pair_index: int):
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    pair_dir = session_dir / f"pair_{pair_index:03d}_{ts}_{ms:03d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    return pair_dir


# ============================================================
# SEEK CAMERA PART
# ============================================================
class SeekState:
    def __init__(self):
        self.cond = Condition()
        self.frame = None
        self.camera = None
        self.connected = False
        self.frame_count = 0
        self.last_frame_time = None


def seek_on_frame(_camera, camera_frame, st: SeekState):
    with st.cond:
        st.frame = camera_frame.thermography_float
        st.frame_count += 1
        st.last_frame_time = time.time()
        st.cond.notify_all()


def seek_on_event(camera: SeekCamera, event_type, event_status, st: SeekState):
    print(f"[SEEK] event={event_type} chip={camera.chipid}")

    if event_type == SeekCameraManagerEvent.CONNECT:
        st.camera = camera
        st.connected = True
        st.frame = None
        camera.register_frame_available_callback(seek_on_frame, st)
        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        if st.camera == camera:
            try:
                camera.capture_session_stop()
            except Exception:
                pass
            st.camera = None
            st.connected = False

    elif event_type == SeekCameraManagerEvent.ERROR:
        print(f"[SEEK] ERROR: {event_status}")


def seek_to_grayscale(tfloat: np.ndarray) -> np.ndarray:
    t = np.asarray(tfloat, dtype=np.float32)
    lo, hi = np.percentile(t, (2, 98))
    if hi <= lo:
        lo = float(np.min(t))
        hi = float(np.max(t))
        if hi <= lo:
            hi = lo + 1.0

    norm = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return gray


def wait_for_seek_ready(st: SeekState, timeout_s=10.0):
    t0 = time.time()
    while not st.connected and time.time() - t0 < timeout_s:
        time.sleep(0.05)

    if not st.connected:
        return False, "Seek camera did not connect."

    with st.cond:
        while st.frame is None and time.time() - t0 < timeout_s:
            st.cond.wait(timeout=0.25)

    if st.frame is None:
        return False, "Seek connected but never delivered a frame."

    return True, None


def capture_seek_at_trigger(st: SeekState, trigger_time, out_path: Path, result: dict):
    wait_until(trigger_time)

    with st.cond:
        if st.frame is None:
            result["seek_error"] = "No Seek frame available at trigger time."
            return

        tf = st.frame.data.copy()

    img = seek_to_grayscale(tf)
    ok = cv2.imwrite(str(out_path), img)

    result["seek_ok"] = bool(ok)
    result["seek_path"] = str(out_path)
    result["seek_shape"] = img.shape


# ============================================================
# OV5642 PART
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
    return i2c_read16(bus, OV5642_I2C_ADDR, 0x300A), i2c_read16(bus, OV5642_I2C_ADDR, 0x300B)


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


def capture_ov5642_at_trigger(spi, trigger_time, jpg_path: Path, raw_path: Path, result: dict):
    fifo_prepare(spi)

    wait_until(trigger_time)
    fifo_start(spi)

    done, trig, err = poll_capture_done(spi, timeout_s=8.0)
    flen = fifo_length(spi)

    if not done:
        result["ov_error"] = err or f"Capture failed, TRIG=0x{trig:02X}"
        return
    if flen == 0:
        result["ov_error"] = "FIFO length is 0."
        return

    payload = read_fifo_single_command_chunks(spi, flen)
    raw_path.write_bytes(payload)

    jpeg, soi, eoi = extract_jpeg(payload)
    if jpeg is None:
        result["ov_error"] = f"No complete JPEG found. SOI={soi}, EOI={eoi}"
        return

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        result["ov_error"] = "Failed to decode OV JPEG."
        return

    img = cv2.flip(img, OV_FLIP_CODE)

    ok = cv2.imwrite(str(jpg_path), img)
    if not ok:
        result["ov_error"] = "Failed to save flipped OV image."
        return

    result["ov_ok"] = True
    result["ov_path"] = str(jpg_path)
    result["ov_raw_path"] = str(raw_path)
    result["ov_shape"] = img.shape


# ============================================================
# MAIN
# ============================================================
def main():
    session_dir = make_session_dir()
    print(f"Session dir: {session_dir}")

    seek_state = SeekState()

    with SeekCameraManager(SeekCameraIOType.USB) as mgr:
        mgr.register_event_callback(seek_on_event, seek_state)

        ok, err = wait_for_seek_ready(seek_state, timeout_s=10.0)
        if not ok:
            raise RuntimeError(err)

        spi = ov5642_open_and_init()

        try:
            for i in range(1, PAIR_COUNT + 1):
                pair_dir = make_pair_dir(session_dir, i)
                trigger_time = time.perf_counter() + SYNC_DELAY_S
                results = {}

                print(f"\n=== PAIR {i}/{PAIR_COUNT} ===")
                print(f"Trigger in {SYNC_DELAY_S:.3f} s")
                print(f"Output -> {pair_dir}")

                t_seek = Thread(
                    target=capture_seek_at_trigger,
                    args=(seek_state, trigger_time, pair_dir / SEEK_FILENAME, results),
                    daemon=True,
                )
                t_ov = Thread(
                    target=capture_ov5642_at_trigger,
                    args=(spi, trigger_time, pair_dir / OV_JPG_FILENAME, pair_dir / OV_RAW_FILENAME, results),
                    daemon=True,
                )

                t_seek.start()
                t_ov.start()
                t_seek.join()
                t_ov.join()

                for k, v in results.items():
                    print(f"{k}: {v}")

                if "seek_error" in results or "ov_error" in results:
                    print("Pair completed with an error.")
                else:
                    print("Pair completed successfully.")

                if i < PAIR_COUNT:
                    time.sleep(INTERVAL_BETWEEN_PAIRS_S)

        finally:
            try:
                spi.close()
            except Exception:
                pass

            if seek_state.camera is not None:
                try:
                    seek_state.camera.capture_session_stop()
                except Exception:
                    pass


if __name__ == "__main__":
    main()