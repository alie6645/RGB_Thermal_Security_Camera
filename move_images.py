import os
import shutil

# Root dataset folder
dataset_dir = r"C:\Users\stant\Downloads\RGB Thermal Dataset\Dataset"

# Output folders
rgb_dir = os.path.join(dataset_dir, "rgb")
thermal_dir = os.path.join(dataset_dir, "thermal")

# Create output directories
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(thermal_dir, exist_ok=True)

# Counter for numbering
counter = 1

# Walk through dataset
for session in sorted(os.listdir(dataset_dir)):
    session_path = os.path.join(dataset_dir, session)

    if not os.path.isdir(session_path) or session in ["rgb", "thermal"]:
        continue

    for pair in sorted(os.listdir(session_path)):
        pair_path = os.path.join(session_path, pair)

        if not os.path.isdir(pair_path):
            continue

        rgb_src = os.path.join(pair_path, "ov5642.jpg")
        thermal_src = os.path.join(pair_path, "seek.png")

        # Only process if BOTH images exist (keeps pairs aligned)
        if os.path.exists(rgb_src) and os.path.exists(thermal_src):

            # Zero-padded filename (e.g., 000001)
            filename = f"{counter:06d}"

            rgb_dst = os.path.join(rgb_dir, filename + ".jpg")
            thermal_dst = os.path.join(thermal_dir, filename + ".png")

            shutil.move(rgb_src, rgb_dst)
            shutil.move(thermal_src, thermal_dst)

            counter += 1

print(f"Done! Moved {counter-1} image pairs.")