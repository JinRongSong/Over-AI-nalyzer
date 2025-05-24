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

from ultralytics import YOLO
import cv2
import numpy as np



CONF_THRESH = 0.3

# Load a pretrained YOLOv5 model
model = YOLO('yolov8n.pt')  # or 'yolov5s.pt' depending on model availability

# Read image
image = cv2.imread("/Users/jinrongs/Desktop/Over-AI-nalyzer/output.jpg")

# Run inference
results = model(image)

# Results: bounding boxes, class IDs, confidences
boxes = results[0].boxes.xyxy.cpu().numpy()  # xmin, ymin, xmax, ymax
class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
confidences = results[0].boxes.conf.cpu().numpy()

# Get class names (coco dataset)
class_names = model.names



objects = []

for box, cls_id, conf in zip(boxes, class_ids, confidences):
    xmin, ymin, xmax, ymax = map(int, box)
    cropped_obj = image[ymin:ymax, xmin:xmax]
    if conf > CONF_THRESH:
        objects.append((cropped_obj, class_names[cls_id], conf))



def average_color(image):
    # image is a cropped numpy array in BGR format (OpenCV)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color  # BGR format

for obj_img, label, conf in objects:
    avg_color = average_color(obj_img)
    print(f"Object: {label}, Confidence: {conf:.2f}, Avg BGR color: {avg_color} ")









