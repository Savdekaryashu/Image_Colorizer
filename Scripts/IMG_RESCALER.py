#op=output_folder
#ip=Input_folder
#img_pth


import os
from PIL import Image
import numpy as np

def resize_img(ip,op,size=(256,256,)):
    if not os.path.exists(op):
        os.makedirs(op)

    total_images = len(os.listdir(ip))
    for idx,i in enumerate(os.listdir(ip)):
        img_pth=os.path.join(ip,i)
        img = Image.open(img_pth)
        img_resized=img.resize(size)
        img_resized.save(os.path.join(op,i))
        print(f"Resizing {i} ({idx + 1}/{total_images})")

def img_gryscle(ip,op):
    if not os.path.exists(op):
        os.makedirs(op)

    total_images = len(os.listdir(ip))
    for idx,i in enumerate( os.listdir(ip)):
        img_pth=os.path.join(ip,i)
        img = Image.open(img_pth).convert('L')
        img.save(os.path.join(op,i))
        print(f"Resizing {i} ({idx + 1}/{total_images})")

def norm_img(ip,op,scale=255):
    if not os.path.exists(op):
        os.makedirs(op)

    total_images = len(os.listdir(ip))
    for idx,i in enumerate( os.listdir(ip)):
        img_pth=os.path.join(ip,i)
        img=Image.open(img_pth)
        img_array=np.array(img)/ scale
        img_norm=Image.fromarray((img_array * scale).astype(np.uint8))
        img_norm.save(os.path.join(op,i))
        print(f"Normalizing {i} ({idx + 1}/{total_images})")


input_dir = "/home/thugyash/ML/Github/Image_Colorizer/Data/DIV2K_train_HR2"  # Original DIV2K images
grayscale_dir = "/home/thugyash/ML/Github/Image_Colorizer/Data/GR"
resized_dir = "/home/thugyash/ML/Github/Image_Colorizer/Data/RS"
normalized_dir = "/home/thugyash/ML/Github/Image_Colorizer/Data/NM"

resize_img(input_dir, resized_dir)
img_gryscle(resized_dir, grayscale_dir)
norm_img(grayscale_dir, normalized_dir)