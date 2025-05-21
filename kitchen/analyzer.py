


# Expect image input to be HEIC, convert to JPEG
from PIL import Image
import pillow_heif



import numpy as np
import cv2


# Example
def convert_heic_to_jpeg(input_path, output_path):
    heif_file = pillow_heif.read_heif(input_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data
    )
    image.save(output_path, format="JPEG")

convert_heic_to_jpeg("/Users/melodywang/Downloads/IMG_7279.HEIC", "/Users/melodywang/Desktop/Over-AI-nalyzer/output.jpg")



# Feature Extraction

# image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)








