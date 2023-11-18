from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np

def resize_image(input_path, output_size):
    """
    Resize an image to a square shape with the specified size for PyTorch, 
    maintaining the aspect ratio and adding padding if necessary.
    
    Parameters:
    input_path (str): Path to the input image.
    output_size (int): The size of the output image (both width and height).

    Returns:
    Tensor: Image tensor suitable for PyTorch models.
    """
    # 이미지를 불러옴
    image = Image.open(input_path)
    # 변환을 위한 Transform 정의
    transform = transforms.Compose([
        transforms.Resize(output_size),  # 먼저, 최대 크기를 output_size로 조정
        transforms.CenterCrop(output_size),  # 중앙에서 output_size 크기로 자름
        transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])

    # 이미지에 Transform 적용
    image_tensor = transform(image)

    return image_tensor

def resize_video(input_path, output_size):
    """
    Resize a video for PyTorch, converting each frame to a square shape 
    of the specified size while maintaining the aspect ratio.

    Parameters:
    input_path (str): Path to the input video.
    output_size (int): The size of the output video frames (both width and height).

    Returns:
    List of Tensors: A list containing each frame as a PyTorch tensor.
    """
    cap = cv2.VideoCapture(input_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 이미지를 PIL 이미지로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # 이미지 리사이즈 함수 사용
        frame_tensor = resize_image(image, output_size)

        # Tensor를 리스트에 추가
        frames.append(frame_tensor)

    cap.release()

    return frames

# 사용 예시
# image_tensor = resize_image('path/to/image.jpg', 416)
# video_frames = resize_video('path/to/video.mp4', 416)
