import cv2
import os

def resize_group(newsize, start, end, interval, dir_name, do_rgb, do_therm):
    count = 0
    notify = ((end - start)//interval) // 10
    progress = 0;
    for i in range (start, end, interval):
        name = str(i) + ".jpg"
        if do_rgb:
            rgbpath = os.path.join("dataset_1-6-2024", "class1", name)
            rgb = cv2.imread(rgbpath)
            rgb = cv2.resize(rgb, (newsize, newsize), interpolation=cv2.INTER_AREA)
        if do_therm:
            thermpath = os.path.join("dataset_1-6-2024", "class2", name)
            therm = cv2.imread(thermpath)
            therm = cv2.resize(therm, (newsize, newsize), interpolation=cv2.INTER_AREA)
        name = str(count) + ".jpg"
        count = count + 1
        if do_rgb:
            rgbpath = os.path.join(dir_name, "rgb", name)
            cv2.imwrite(rgbpath, rgb)
        if do_therm:
            thermpath = os.path.join(dir_name, "therm", name)
            cv2.imwrite(thermpath, therm)
        if (count % notify == 0):
            progress = (100 * count) // ((end-start)//interval)
            print(dir_name + ": " + str(progress) + "% complete")
    print(dir_name + ": DONE")

def datagen(newsize, len, interval, dir_name, do_rgb=True, do_therm=True):
    train_range = (len//8) * 7
    resize_group(newsize, 0, train_range, interval, dir_name, do_rgb, do_therm)
    testpath = os.path.join(dir_name, "test")
    resize_group(newsize, train_range, len, interval, testpath, do_rgb, do_therm)

if __name__ == "__main__":
    datagen(256, 10000, 5, "full")