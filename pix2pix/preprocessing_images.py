#!/usr/bin/env python3

import csv
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("/home/donpc/projects/RGB_Thermal_Security_Camera/RGB Thermal Dataset")
OUT_DIR = Path("/home/donpc/projects/RGB_Thermal_Security_Camera/processed_640x480")
OUT_MANIFEST = OUT_DIR / "manifest.csv"

TARGET_W = 640
TARGET_H = 480

PREVIEW_LIMIT = 100
ALPHAS = [0.60]

RGB_NAME = "ov5642.jpg"
THERMAL_NAME = "seek.png"

# ------------------------------------------------------------
# RGB transform knobs
# ------------------------------------------------------------
RGB_SCALE_X = 1.0
RGB_SCALE_Y = 1.0
RGB_ROT_DEG = 0.0
RGB_TX = 0
RGB_TY = 0

# ------------------------------------------------------------
# Thermal transform knobs
# ------------------------------------------------------------
THERMAL_SCALE_X = 1.4
THERMAL_SCALE_Y = 1.4
THERMAL_ROT_DEG = 0.0
THERMAL_TX = 40
THERMAL_TY = 0

# ------------------------------------------------------------
# Optional center crop before transform/resize
# ------------------------------------------------------------
RGB_CENTER_CROP_FRAC = 1.0
THERMAL_CENTER_CROP_FRAC = 1.0

# ------------------------------------------------------------
# Thermal normalization
# ------------------------------------------------------------
THERMAL_P_LOW = 1.0
THERMAL_P_HIGH = 99.0

# ------------------------------------------------------------
# Hot region preview threshold
# ------------------------------------------------------------
HEAT_THRESH = 0.60


# ============================================================
# UTILS
# ============================================================

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pair_dirs(root: Path):
    return sorted([p for p in root.rglob("pair_*") if p.is_dir()])


def safe_imread_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load color image: {path}")
    return img


def safe_imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load grayscale image: {path}")
    return img


def center_crop_fraction(img: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 0.999:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(w * frac))
    new_h = max(1, int(h * frac))

    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    x1 = x0 + new_w
    y1 = y0 + new_h

    return img[y0:y1, x0:x1]


def apply_rgb_transform(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, RGB_ROT_DEG, 1.0)

    M[0, 0] *= RGB_SCALE_X
    M[0, 1] *= RGB_SCALE_X
    M[1, 0] *= RGB_SCALE_Y
    M[1, 1] *= RGB_SCALE_Y

    M[0, 2] += center[0] - (M[0, 0] * center[0] + M[0, 1] * center[1])
    M[1, 2] += center[1] - (M[1, 0] * center[0] + M[1, 1] * center[1])

    M[0, 2] += RGB_TX
    M[1, 2] += RGB_TY

    transformed = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return transformed


def apply_thermal_transform(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, THERMAL_ROT_DEG, 1.0)

    M[0, 0] *= THERMAL_SCALE_X
    M[0, 1] *= THERMAL_SCALE_X
    M[1, 0] *= THERMAL_SCALE_Y
    M[1, 1] *= THERMAL_SCALE_Y

    M[0, 2] += center[0] - (M[0, 0] * center[0] + M[0, 1] * center[1])
    M[1, 2] += center[1] - (M[1, 0] * center[0] + M[1, 1] * center[1])

    M[0, 2] += THERMAL_TX
    M[1, 2] += THERMAL_TY

    transformed = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed


def resize_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def resize_thermal(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)


def normalize_thermal(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)

    lo = np.percentile(img_f, THERMAL_P_LOW)
    hi = np.percentile(img_f, THERMAL_P_HIGH)

    if hi <= lo:
        hi = lo + 1.0

    out = (img_f - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out


def thermal_to_u8(thermal_norm: np.ndarray) -> np.ndarray:
    return (thermal_norm * 255.0).astype(np.uint8)


def thermal_colormap(thermal_norm: np.ndarray) -> np.ndarray:
    thermal_u8 = thermal_to_u8(thermal_norm)
    return cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)


def blend_overlay(rgb_bgr: np.ndarray, thermal_color: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, thermal_color, alpha, 0.0)


def brighten_rgb_shadow_reveal(rgb_bgr: np.ndarray) -> np.ndarray:
    """
    Brighten dark areas in the RGB image to try to reveal people/shadows
    that correspond to dark blobs or hidden detail.
    """
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    v_norm = v / 255.0
    gamma = 0.55
    v_bright = np.power(v_norm, gamma).astype(np.float32)

    dark_boost = np.where(v_norm < 0.45, 0.18, 0.0).astype(np.float32)
    v_bright = np.clip(v_bright + dark_boost, 0.0, 1.0).astype(np.float32)

    v_bright = np.clip((v_bright - 0.5) * 0.9 + 0.5, 0.0, 1.0).astype(np.float32)
    s_out = np.clip(s * 0.95, 0, 255).astype(np.float32)
    v_out = (v_bright * 255.0).astype(np.float32)

    hsv_out = cv2.merge([h.astype(np.float32), s_out, v_out])
    hsv_out = np.clip(hsv_out, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

def make_pair_output_dir(session_name: str, pair_name: str) -> Path:
    pair_out_dir = OUT_DIR / session_name / pair_name
    pair_out_dir.mkdir(parents=True, exist_ok=True)
    return pair_out_dir


def process_pair_dir(pair_dir: Path, idx: int):
    rgb_path = pair_dir / RGB_NAME
    thermal_path = pair_dir / THERMAL_NAME

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
    if not thermal_path.exists():
        raise FileNotFoundError(f"Missing thermal image: {thermal_path}")

    rgb_original = safe_imread_color(rgb_path)
    thermal_original = safe_imread_gray(thermal_path)

    rgb_in_shape = f"{rgb_original.shape[1]}x{rgb_original.shape[0]}"
    thermal_in_shape = f"{thermal_original.shape[1]}x{thermal_original.shape[0]}"

    rgb = center_crop_fraction(rgb_original, RGB_CENTER_CROP_FRAC)
    thermal = center_crop_fraction(thermal_original, THERMAL_CENTER_CROP_FRAC)

    rgb = apply_rgb_transform(rgb)
    thermal = apply_thermal_transform(thermal)

    rgb_resized = resize_rgb(rgb)
    thermal_resized = resize_thermal(thermal)

    thermal_norm = normalize_thermal(thermal_resized)
    thermal_u8 = thermal_to_u8(thermal_norm)
    thermal_color = thermal_colormap(thermal_norm)

    bright_rgb = brighten_rgb_shadow_reveal(rgb_resized)
    overlap = blend_overlay(rgb_resized, thermal_color, ALPHAS[0])

    session_name = pair_dir.parent.name
    pair_name = pair_dir.name
    pair_out_dir = make_pair_output_dir(session_name, pair_name)

    input_rgb_out = pair_out_dir / "input_rgb.png"
    ground_truth_overlap_out = pair_out_dir / "ground_truth_overlap.png"
    bright_rgb_out = pair_out_dir / "brightened_rgb.png"
    thermal_gray_out = pair_out_dir / "thermal_gray.png"
    thermal_color_out = pair_out_dir / "thermal_color.png"

    cv2.imwrite(str(input_rgb_out), rgb_resized)
    cv2.imwrite(str(ground_truth_overlap_out), overlap)
    cv2.imwrite(str(bright_rgb_out), bright_rgb)
    cv2.imwrite(str(thermal_gray_out), thermal_u8)
    cv2.imwrite(str(thermal_color_out), thermal_color)

    return {
        "session": session_name,
        "pair": pair_name,
        "rgb_input": str(rgb_path),
        "thermal_input": str(thermal_path),
        "pair_output_dir": str(pair_out_dir),
        "input_rgb_output": str(input_rgb_out),
        "ground_truth_overlap_output": str(ground_truth_overlap_out),
        "brightened_rgb_output": str(bright_rgb_out),
        "thermal_gray_output": str(thermal_gray_out),
        "thermal_color_output": str(thermal_color_out),
        "rgb_shape_in": rgb_in_shape,
        "thermal_shape_in": thermal_in_shape,
        "shape_out": f"{TARGET_W}x{TARGET_H}",
    }


def write_manifest(rows):
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session",
                "pair",
                "rgb_input",
                "thermal_input",
                "pair_output_dir",
                "input_rgb_output",
                "ground_truth_overlap_output",
                "brightened_rgb_output",
                "thermal_gray_output",
                "thermal_color_output",
                "rgb_shape_in",
                "thermal_shape_in",
                "shape_out",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()
    pair_dirs = find_pair_dirs(DATASET_ROOT)

    print(f"Found {len(pair_dirs)} pair directories under {DATASET_ROOT}")

    rows = []
    failures = 0

    for idx, pair_dir in enumerate(pair_dirs):
        try:
            row = process_pair_dir(pair_dir, idx)
            rows.append(row)
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(pair_dirs)}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {pair_dir}: {e}")

    write_manifest(rows)

    print("Done.")
    print(f"Successful: {len(rows)}")
    print(f"Failed: {failures}")
    print(f"Outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()