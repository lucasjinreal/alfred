"""

image format convert in popular libs

PIL.Image -> OpenCV
OpenCV -> PIL.Image
Matplotlib.PLT -> OpenCV

"""
from PIL import Image
import numpy as np
import cv2


def cv2pil(image, inplace=True):
    if inplace:
        new_image = image
    else:
        new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(pil_image):
    new_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return new_image
