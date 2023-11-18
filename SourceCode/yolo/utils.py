from PIL import Image
import cv2
import numpy as np

def resize_image(input_path, output_size):
    """
    Resize an image to a square shape with the specified size while maintaining the aspect ratio.
    
    Parameters:
    input_path (str): Path to the input image.
    output_size (int): The size of the output image (both width and height).

    Returns:
    Image: The resized image.
    """
    # 이미지를 불러옴
    image = Image.open(input_path)

    # 원본 이미지의 크기를 얻음
    original_size = max(image.size)

    # 원본 비율을 유지하면서 출력 크기에 맞춰 조정
    scale = output_size / original_size
    new_size = (int(scale * image.size[0]), int(scale * image.size[1]))
    image = image.resize(new_size, Image.ANTIALIAS)

    # 새로운 이미지를 생성하고, 원본 이미지를 중앙에 배치
    new_image = Image.new("RGB", (output_size, output_size))
    new_image.paste(image, ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2))

    return new_image

def resize_video(input_path, output_path, output_size):
    """
    Resize a video to square frames of the specified size while maintaining the aspect ratio.
    
    Parameters:
    input_path (str): Path to the input video.
    output_path (str): Path for the output resized video.
    output_size (int): The size of the output video frames (both width and height).
    """
    # 비디오를 불러옴
    cap = cv2.VideoCapture(input_path)

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (output_size, output_size))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 PIL 이미지로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # 이미지 리사이즈 함수 사용
        resized_image = resize_image(image, output_size)

        # 리사이즈된 이미지를 다시 OpenCV 형식으로 변환
        frame = np.array(resized_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 출력 비디오에 프레임 추가
        out.write(frame)

    # 자원 해제
    cap.release()
    out.release()
