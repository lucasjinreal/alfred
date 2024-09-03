import os
import sys
import argparse
import cv2
import numpy as np
from natsort import natsorted


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate images in a specified shape."
    )
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory containing images"
    )
    parser.add_argument(
        "-s",
        "--shape",
        required=True,
        help="Shape to concatenate images, e.g., 3x3 or 3xN or Nx5",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=0,
        help="Padding size to add around each image",
    )
    return parser.parse_args()


def load_images(image_dir):
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    image_files = natsorted(image_files)
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]
    return images


def create_black_image(shape, dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


def concat_images(images, shape, padding):
    if "xN" in shape:
        rows = int(shape.split("x")[0])
        cols = (len(images) + rows - 1) // rows  # Calculate columns needed
    elif "Nx" in shape:
        cols = int(shape.split("x")[1])
        rows = (len(images) + cols - 1) // cols  # Calculate rows needed
    else:
        rows, cols = map(int, shape.split("x"))

    max_height = max(image.shape[0] for image in images) + 2 * padding
    max_width = max(image.shape[1] for image in images) + 2 * padding
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1

    white_image = create_black_image((max_height, max_width, channels)) + 255

    grid = []
    for r in range(rows):
        row_images = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(images):
                img = images[idx]
                img_padded = cv2.copyMakeBorder(
                    img,
                    padding,
                    padding,
                    padding,
                    padding,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
                img_padded = cv2.copyMakeBorder(
                    img_padded,
                    0,
                    max_height - img_padded.shape[0],
                    0,
                    max_width - img_padded.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            else:
                img_padded = white_image
            row_images.append(img_padded)
        grid.append(np.hstack(row_images))
    return np.vstack(grid)


def main():
    args = parse_args()
    images = load_images(args.dir)
    concatenated_image = concat_images(images, args.shape, args.padding)
    cv2.imwrite("concatenated_image.jpg", concatenated_image)
    print("Concatenated image saved as 'concatenated_image.jpg'")


# if __name__ == "__main__":
# main()
