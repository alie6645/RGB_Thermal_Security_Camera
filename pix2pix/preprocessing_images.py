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
OUT_RGB_DIR = OUT_DIR / "rgb"
OUT_THERMAL_DIR = OUT_DIR / "thermal"
OUT_PREVIEW_DIR = OUT_DIR / "previews"
OUT_MANIFEST = OUT_DIR / "manifest.csv"

TARGET_W = 640
TARGET_H = 480

PREVIEW_LIMIT = 100
ALPHAS = [0.25, 0.40, 0.60]

RGB_NAME = "ov5642.jpg"
THERMAL_NAME = "seek.png"

# ------------------------------------------------------------
# RGB transform knobs
# You said RGB needs to shift right and down by ~100 px
# ------------------------------------------------------------
RGB_SCALE_X = 1.0
RGB_SCALE_Y = 1   # try 0.80 to 0.90 and tune
RGB_ROT_DEG = 0.0
RGB_TX = 0
RGB_TY = 0

# ------------------------------------------------------------
# Thermal transform knobs
# Leave identity for now unless inspection says otherwise
# ------------------------------------------------------------
# ------------------------------------------------------------
# Thermal transform knobs
# ------------------------------------------------------------
THERMAL_SCALE_X = 1.4
THERMAL_SCALE_Y = 1.4
THERMAL_ROT_DEG = 0.0
THERMAL_TX = 70
THERMAL_TY = 30

# ------------------------------------------------------------
# Optional center crop before transform/resize
# 1.0 means no crop
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
    for d in [OUT_DIR, OUT_RGB_DIR, OUT_THERMAL_DIR, OUT_PREVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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

    # Start with rotation matrix
    M = cv2.getRotationMatrix2D(center, RGB_ROT_DEG, 1.0)

    # Apply non-uniform scaling around the image center
    M[0, 0] *= RGB_SCALE_X
    M[0, 1] *= RGB_SCALE_X
    M[1, 0] *= RGB_SCALE_Y
    M[1, 1] *= RGB_SCALE_Y

    # Re-center after anisotropic scaling so scaling happens about the center
    M[0, 2] += center[0] - (M[0, 0] * center[0] + M[0, 1] * center[1])
    M[1, 2] += center[1] - (M[1, 0] * center[0] + M[1, 1] * center[1])

    # Then apply manual translation
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

    # Start with rotation matrix
    M = cv2.getRotationMatrix2D(center, THERMAL_ROT_DEG, 1.0)

    # Apply non-uniform scaling around the image center
    M[0, 0] *= THERMAL_SCALE_X
    M[0, 1] *= THERMAL_SCALE_X
    M[1, 0] *= THERMAL_SCALE_Y
    M[1, 1] *= THERMAL_SCALE_Y

    # Re-center after anisotropic scaling so scaling happens about the center
    M[0, 2] += center[0] - (M[0, 0] * center[0] + M[0, 1] * center[1])
    M[1, 2] += center[1] - (M[1, 0] * center[0] + M[1, 1] * center[1])

    # Then apply manual translation
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


def make_hot_mask(thermal_norm: np.ndarray, thresh: float = HEAT_THRESH) -> np.ndarray:
    mask = (thermal_norm >= thresh).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def side_by_side(rgb_bgr: np.ndarray, thermal_u8: np.ndarray, thermal_color: np.ndarray) -> np.ndarray:
    thermal_gray_bgr = cv2.cvtColor(thermal_u8, cv2.COLOR_GRAY2BGR)
    return np.hstack([rgb_bgr, thermal_gray_bgr, thermal_color])


def blend_overlay(rgb_bgr: np.ndarray, thermal_color: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, thermal_color, alpha, 0.0)


def edge_overlay(rgb_bgr: np.ndarray, thermal_norm: np.ndarray) -> np.ndarray:
    thermal_u8 = thermal_to_u8(thermal_norm)
    edges = cv2.Canny(thermal_u8, 50, 150)

    out = rgb_bgr.copy()
    out[edges > 0] = (0, 0, 255)
    return out


def checkerboard_overlay(rgb_bgr: np.ndarray, thermal_color: np.ndarray, tile: int = 32) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    out = np.zeros_like(rgb_bgr)

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            use_rgb = ((x // tile) + (y // tile)) % 2 == 0
            src = rgb_bgr if use_rgb else thermal_color
            out[y:y + tile, x:x + tile] = src[y:y + tile, x:x + tile]

    return out


def hotmask_overlay(rgb_bgr: np.ndarray, hot_mask: np.ndarray) -> np.ndarray:
    out = rgb_bgr.copy()
    red = np.zeros_like(rgb_bgr)
    red[:, :, 2] = 255
    alpha = 0.35

    hot = hot_mask > 0
    out[hot] = cv2.addWeighted(out[hot], 1.0 - alpha, red[hot], alpha, 0.0)
    return out


def save_preview_set(stem: str, rgb_bgr: np.ndarray, thermal_norm: np.ndarray):
    thermal_u8 = thermal_to_u8(thermal_norm)
    thermal_color = thermal_colormap(thermal_norm)
    hot_mask = make_hot_mask(thermal_norm)

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_side.png"),
                side_by_side(rgb_bgr, thermal_u8, thermal_color))

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_edges.png"),
                edge_overlay(rgb_bgr, thermal_norm))

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_checker.png"),
                checkerboard_overlay(rgb_bgr, thermal_color))

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_hotmask.png"),
                hotmask_overlay(rgb_bgr, hot_mask))

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_thermal_color.png"),
                thermal_color)

    cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_thermal_gray.png"),
                thermal_to_u8(thermal_norm))

    for alpha in ALPHAS:
        tag = str(alpha).replace(".", "")
        cv2.imwrite(str(OUT_PREVIEW_DIR / f"{stem}_blend_{tag}.png"),
                    blend_overlay(rgb_bgr, thermal_color, alpha))


def process_pair_dir(pair_dir: Path, idx: int):
    rgb_path = pair_dir / RGB_NAME
    thermal_path = pair_dir / THERMAL_NAME

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
    if not thermal_path.exists():
        raise FileNotFoundError(f"Missing thermal image: {thermal_path}")

    rgb = safe_imread_color(rgb_path)
    thermal = safe_imread_gray(thermal_path)

    rgb_in_shape = f"{rgb.shape[1]}x{rgb.shape[0]}"
    thermal_in_shape = f"{thermal.shape[1]}x{thermal.shape[0]}"

    rgb = center_crop_fraction(rgb, RGB_CENTER_CROP_FRAC)
    thermal = center_crop_fraction(thermal, THERMAL_CENTER_CROP_FRAC)

    rgb = apply_rgb_transform(rgb)
    thermal = apply_thermal_transform(thermal)

    rgb_resized = resize_rgb(rgb)
    thermal_resized = resize_thermal(thermal)

    thermal_norm = normalize_thermal(thermal_resized)
    thermal_u8 = thermal_to_u8(thermal_norm)

    session_name = pair_dir.parent.name
    pair_name = pair_dir.name
    stem = f"{session_name}__{pair_name}"

    rgb_out_path = OUT_RGB_DIR / f"{stem}.png"
    thermal_out_path = OUT_THERMAL_DIR / f"{stem}.png"

    cv2.imwrite(str(rgb_out_path), rgb_resized)
    cv2.imwrite(str(thermal_out_path), thermal_u8)

    if idx < PREVIEW_LIMIT:
        save_preview_set(stem, rgb_resized, thermal_norm)

    return {
        "session": session_name,
        "pair": pair_name,
        "rgb_input": str(rgb_path),
        "thermal_input": str(thermal_path),
        "rgb_output": str(rgb_out_path),
        "thermal_output": str(thermal_out_path),
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
                "rgb_output",
                "thermal_output",
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