from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
model.predict(
    source = 'https://media.roboflow.com/notebooks/examples/dog.jpeg',
    conf = 0.25
)

