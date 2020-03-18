"""

Process on image

"""


import cv2


def darken_image(ori_img, dark_factor=0.6):
    """
    this will darken origin image and return darken one
    """
    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
    hsv_img[...,2] = hsv_img[...,2]*dark_factor
    originimg = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return originimg