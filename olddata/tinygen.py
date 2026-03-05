import cv2
import os
count = 0
for i in range (2443, 3443, 10):
    num = i - 2443
    name = str(i) + ".jpg"
    rgbpath = os.path.join("dataset_1-6-2024", "class1", name)
    thermpath = os.path.join("dataset_1-6-2024", "class2", name)
    rgb = cv2.imread(rgbpath)
    therm = cv2.imread(thermpath)
    rgb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    therm = cv2.resize(therm, (32, 32), interpolation=cv2.INTER_AREA)
    name = str(count) + ".jpg"
    count = count + 1
    rgbpath = os.path.join("tiny", "test", "rgb", name)
    thermpath = os.path.join("tiny", "test", "therm", name)
    cv2.imwrite(rgbpath, rgb)
    cv2.imwrite(thermpath, therm)
