import os

count = 0
max_num = 151373
for i in range(1, max_num + 1):
    rgbpath = os.path.join("class1", str(i) + ".jpg")
    thermpath = os.path.join("class2", str(i) + ".jpg")
    if os.path.exists(rgbpath):
        newrgb = os.path.join("class1", str(count) + ".jpg")
        newtherm = os.path.join("class2", str(count) + ".jpg")
        os.rename(rgbpath, newrgb)
        os.rename(thermpath, newtherm)
        count = count + 1
print(count)