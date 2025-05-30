# Expect image input to be HEIC, convert to JPEG
from PIL import Image
# import pillow_heif



import numpy as np
import cv2
import prompt


# # Example
# def convert_heic_to_jpeg(input_path, output_path):
#     heif_file = pillow_heif.read_heif(input_path)
#     image = Image.frombytes(
#         heif_file.mode, heif_file.size, heif_file.data
#     )
#     image.save(output_path, format="JPEG")

# convert_heic_to_jpeg("/Users/melodywang/Downloads/IMG_7279.HEIC", "/Users/melodywang/Desktop/Over-AI-nalyzer/output.jpg")



# Feature Extraction

from ultralytics import YOLO
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv



CONF_THRESH = 0.3

# # Load a pretrained YOLOv5 model
# model = YOLO('yolov8n.pt')  # or 'yolov5s.pt' depending on model availability

# # Read image
image = cv2.imread("/Users/jinrongs/Desktop/Over-AI-nalyzer/output.jpg")

# # Run inference
# results = model(image)

# # Results: bounding boxes, class IDs, confidences
# boxes = results[0].boxes.xyxy.cpu().numpy()  # xmin, ymin, xmax, ymax
# class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
# confidences = results[0].boxes.conf.cpu().numpy()

# # Get class names (coco dataset)
# class_names = model.names

# objects = []

# for box, cls_id, conf in zip(boxes, class_ids, confidences):
#     xmin, ymin, xmax, ymax = map(int, box)
#     cropped_obj = image[ymin:ymax, xmin:xmax]
#     if conf > CONF_THRESH:
#         objects.append((cropped_obj, class_names[cls_id], conf))



# def average_color(image):
#     # image is a cropped numpy array in BGR format (OpenCV)
#     avg_color_per_row = np.average(image, axis=0)
#     avg_color = np.average(avg_color_per_row, axis=0)
#     return avg_color  # BGR format

# for obj_img, label, conf in objects:
#     avg_color = average_color(obj_img)
#     print(f"Object: {label}, Confidence: {conf:.2f}, Avg BGR color: {avg_color} ")

url = "https://predict.ultralytics.com"

load_dotenv()

X_API_KEY = os.getenv("X_API_KEY")

headers = {"x-api-key": X_API_KEY}

data = {"model": "https://hub.ultralytics.com/models/y0z7VFqOXtdtcoyrDWzZ", "imgsz": 640, "conf": 0.25, "iou": 0.45}

with open("/Users/jinrongs/Desktop/Over-AI-nalyzer/output.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"file": f})


result = response.json()

print(json.dumps(response.json(), indent=2))

# predictions = result["data"]["predictions"]
# class_names = result["data"].get("names", {}) 


# objects = []

# for pred in predictions:
#     conf = pred["confidence"]
#     if conf >= CONF_THRESH:
#         # Extract bounding box
#         xmin = int(pred["x"] - pred["width"] / 2)
#         ymin = int(pred["y"] - pred["height"] / 2)
#         xmax = int(pred["x"] + pred["width"] / 2)
#         ymax = int(pred["y"] + pred["height"] / 2)

#         # Crop object from image
#         cropped_obj = image[ymin:ymax, xmin:xmax]

#         # Get label
#         class_id = int(pred["class"])
#         label = class_names.get(str(class_id), f"class_{class_id}")

#         objects.append((cropped_obj, label, conf))

# # === Print detected objects ===
# for i, (cropped, label, conf) in enumerate(objects):
#     print(f"[{i}] {label} ({conf:.2f})")




