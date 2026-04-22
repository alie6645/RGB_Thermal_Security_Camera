import cv2
import os

# edit to target rgb and thermal directories
# can also change the generated folder's name

rgb_dir_in = "data/smallrgb"
therm_dir_in = "data/thermal"
rgb_dir_out = "mrgb"
therm_dir_out = "mtherm"

def batch_crop_resize(
    input_dir,
    output_dir,
    crop_region,
    resize_to,
    func=None
):

    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    os.makedirs(output_dir, exist_ok=True)
    x, y, w, h = crop_region

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(extensions):
            continue

        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue
        
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, resize_to, interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, resized)

        print(f"Processed: {filename}")

if __name__ == "__main__":
    batch_crop_resize(rgb_dir_in, rgb_dir_out, (0, 0, 270, 320), (400, 400))
    batch_crop_resize(therm_dir_in, therm_dir_out, (0, 0, 200, 200), (400, 400))
    
