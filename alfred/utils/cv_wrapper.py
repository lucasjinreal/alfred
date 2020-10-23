"""

some opencv wrappers to make video inference
more simple

"""
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np


font_f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts/FZSSJW.TTF")

def put_cn_txt_on_img(img, txt, ori, font_scale, color):
    """
    put Chinese text on image
    """
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert os.path.exists(font_f), '{} not found'.format(font_f)
    font = ImageFont.truetype(font_f, 25)
    fillColor = color #(255,0,0)
    position = ori #(100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, txt, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img