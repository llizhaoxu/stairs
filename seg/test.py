import os
from PIL import Image
import numpy as np
import cv2
if __name__ == '__main__':
    image_path ='/Users/li/PycharmProjects/stairs/seg/1.jpg'

    image = np.asarray(Image.open(image_path), dtype=np.float32)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(np.uint8(im_rgb)).save('lena_rgb_pillow.jpg')