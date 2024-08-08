try:
    import face_recognition
    import os
except ImportError as e:
    pass
import glob
from PIL import Image


def extract_and_save_face(image_path):
    if os.path.isdir(image_path):
        # Handle multiple file types
        images = glob.glob(os.path.join(image_path, "*"))
        images = [img for img in images if img.endswith((".png", ".jpg", ".jpeg"))]
    else:
        images = [image_path]

    for image_path in images:
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Find face locations
        face_locations = face_recognition.face_locations(image)

        ratio = 0.35

        for i, face_location in enumerate(face_locations):
            print(f"{i} {face_location}")
            top, right, bottom, left = face_location
            # left, top, right, bottom = face_location

            # Expand the box by 1.5 ratio
            height, width = bottom - top, right - left
            top = max(0, int(top - ratio * height))
            left = max(0, int(left - ratio * width))
            bottom = min(image.shape[0], int(bottom + ratio * height))
            right = min(image.shape[1], int(right + ratio * width))

            # Crop the face from the image
            face_image = image[top:bottom, left:right]

            # Save the image
            pil_image = Image.fromarray(face_image)
            pil_image.save(f"{image_path[:-4]}_{i}.jpg")

        if len(face_locations) == 0:
            print(f"No face detected. in: {image_path}")

    print("done!")


# Use the function
# extract_and_save_face("example.jpg")
