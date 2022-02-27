import cv2
import numpy as np


def draw_face_landmarks(img, pts, box=None, color=(255, 147, 23), size=8):
    if pts is not None:
        print(pts.shape)
        n = pts.shape[1]
        if n <= 106:
            for i in range(n):
                cv2.circle(
                    img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1
                )
        else:
            sep = 1
            for i in range(0, n, sep):
                cv2.circle(
                    img, (int(round(pts[0, i])), int(round(pts[1, i]))), 2, color, 1
                )

        if box is not None:
            line_color = (255, 127, 80)
            left, top, right, bottom = np.round(box).astype(np.int32)
            left_top = (left, top)
            right_top = (right, top)
            right_bottom = (right, bottom)
            left_bottom = (left, bottom)
            cv2.line(img, left_top, right_top, line_color, 1, cv2.LINE_AA)
            cv2.line(img, right_top, right_bottom, line_color, 1, cv2.LINE_AA)
            cv2.line(img, right_bottom, left_bottom, line_color, 1, cv2.LINE_AA)
            cv2.line(img, left_bottom, left_top, line_color, 1, cv2.LINE_AA)
    return img

