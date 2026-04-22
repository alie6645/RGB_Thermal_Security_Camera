models are too large to store on GitHub. download the model files [here](https://drive.google.com/drive/folders/1nauK01mP7KUS5KA32KW5W71FTqJ187Kq?usp=sharing) and drop full folder into video/ directory

to run computer camera use:

small NAFNet:
```python video.py n s```

large NAFNet:
```python video.py n l```

small UNet:
```python video.py u s```

UNet:
```python video.py u l```

q to quit

to convert a given video use (only uses 400x400 models):
```python convert_video.py source target model```
model options: nafnet, nafnet50, nafnet100, unet