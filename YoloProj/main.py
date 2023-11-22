from ultralytics import YOLO
from videoio import * # -> includes resize_image and resize_video
import torch

ultralytics.checks()

model = YOLO('yolov8n.yaml')






