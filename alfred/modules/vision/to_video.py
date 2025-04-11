import os
import cv2
import numpy as np
from natsort import natsorted
from colorama import Fore, init

init(autoreset=True)  # Auto-reset colorama after each print

class VideoCombiner(object):
    def __init__(self, img_dir):
        self.img_dir = os.path.abspath(img_dir)

        if not os.path.exists(self.img_dir):
            print(Fore.RED + "=> Error: " + f"img_dir {self.img_dir} does not exist.")
            exit(1)

        self._get_video_shape()

    def _get_video_shape(self):
        # Filter and sort image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_files = os.listdir(self.img_dir)
        
        # Filter image files with valid extensions
        self.all_images = [
            os.path.join(self.img_dir, f) 
            for f in all_files 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if not self.all_images:
            print(Fore.RED + "=> Error: " + f"No valid image files found in {self.img_dir}")
            exit(1)

        # Natural sort the images
        self.all_images = natsorted(self.all_images)

        # Get video shape from first image (more reliable than random)
        sample_img = self.all_images[0]
        img = cv2.imread(sample_img)
        
        if img is None:
            print(Fore.RED + "=> Error: " + f"Failed to read sample image {sample_img}")
            exit(1)
            
        self.video_shape = img.shape

    def combine(self, target_file="combined.mp4"):
        size = (self.video_shape[1], self.video_shape[0])
        print("=> Target video frame size:", size)
        print(f"=> Total {len(self.all_images)} frames to process")

        # Create video writer with correct parameters
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better compatibility than DIVX
        video_writer = cv2.VideoWriter(target_file, fourcc, 30, size)

        if not video_writer.isOpened():
            print(Fore.RED + "=> Error: " + "Failed to initialize video writer")
            exit(1)

        print("=> Processing frames...")
        for i, img_path in enumerate(self.all_images, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(Fore.YELLOW + f"=> Warning: Skipped corrupted/invalid file {img_path}")
                continue

            # Ensure consistent frame size
            if img.shape != self.video_shape:
                img = cv2.resize(img, size)
                
            video_writer.write(img)
            if i % 100 == 0:
                print(f"=> Processed {i}/{len(self.all_images)} frames")

        video_writer.release()
        print(Fore.GREEN + f"=> Success: Video saved as {os.path.abspath(target_file)}")

# Example usage
if __name__ == "__main__":
    combiner = VideoCombiner("path/to/your/images")
    combiner.combine("output_video.mp4")