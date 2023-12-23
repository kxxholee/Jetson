import cv2
from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Export code
# Load the YOLOv8 model
model = YOLO('/workspace/Ultra/backup-latest.engine')

# 동영상 파일 사용시
# video_path = "path/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

# 웹캠 프레임 크기 조정 (640x480)
# webcam에서 받아오는 프레임이 너무 큰 문제 존재 -> 화면 크기 조절
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 웹캠 프레임 속도 조정 (30 fps)
# 화면 초당 프레임을 적절히 조절함
cap.set(cv2.CAP_PROP_FPS, 30)

# 오버레이 좌표
OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2 = 220, 160, 420, 430

# OpenCV 이미지를 PIL 이미지로 변환하는 함수
# def cv2_to_pil(cv2_image):
#     cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(cv2_image)

# # PIL 이미지를 OpenCV 이미지로 변환하는 함수
# def pil_to_cv2(pil_image):
#     return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def key_event_listener(key):
    global OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2
    # >>> 이동 >>>
    if key == ord('w'):
        OVERLAY_Y1 = OVERLAY_Y1 - 10
        OVERLAY_Y2 = OVERLAY_Y2 - 10
        return 0
    elif key == ord('s'):
        OVERLAY_Y1 += 10
        OVERLAY_Y2 += 10
        return 0
    elif key == ord('a'):
        OVERLAY_X1 = OVERLAY_X1 - 10
        OVERLAY_X2 = OVERLAY_X2 - 10
        return 0
    elif key == ord('d'):
        OVERLAY_X1 += 10
        OVERLAY_X2 += 10
        return 0
    # <<< 이동 <<<
    # >>> zoom >>>
    elif key == ord('i'): # 세로 작게
        OVERLAY_Y1 = OVERLAY_Y1 - 10 if OVERLAY_Y2 - OVERLAY_Y1 > 10 else OVERLAY_Y1
        return 0
    elif key == ord('k'): # 세로 크게
        OVERLAY_Y1 += 10
        return 0
    elif key == ord('l'): # 가로 크게
        OVERLAY_X1 += 10
        return 0
    elif key == ord('j'):
        OVERLAY_X1 = OVERLAY_X1 - 10 if OVERLAY_X2 - OVERLAY_X1 > 10 else OVERLAY_X1
        return 0
    # <<< zoom <<<
    # >>> exit >>>
    elif key == ord('q'):
        return 1

def is_overlap(rect1, rect2):
    # rect1과 rect2는 (x1, y1, x2, y2) tuple
    x1_rect1, y1_rect1, x2_rect1, y2_rect1 = rect1
    x1_rect2, y1_rect2, x2_rect2, y2_rect2 = rect2

    return not ((x1_rect1 > x2_rect2) or    # if x1_rect1 > x2_rect2 or x1_rect2 > x2_rect1: False
                (x1_rect2 > x2_rect1) or    # if x1_rect1 > x2_rect2 or x1_rect2 > x2_rect1: False
                (y1_rect1 > y2_rect2) or    # else true
                (y1_rect2 > y2_rect1))

def do_warning_action():
    # 여기에 특정 작업 정의
    print("Object overlapped with shape!")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        # >>> overlay되는 부분 표시 >>>
        if key_event_listener(cv2.waitKey(1) & 0xFF) == 1: # 여기서 loop 벗어나는 조건 : 'q'
            break
        overlay = frame.copy()
        cv2.rectangle(overlay, (OVERLAY_X1, OVERLAY_Y1), (OVERLAY_X2, OVERLAY_Y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)
        # <<< overlay 되는 부분 표시 <<<

        rect_color = (0, 255, 0)
        for detection in results[0]:
            frame = detection.plot()
            boxs = detection.boxes              # detect한 box를 꺼내고
            x1, y1, x2, y2 = boxs.xyxy[0]       # xy좌표를 꺼냄

            object_bbox = (int(x1), int(y1), int(x2), int(y2))  # 탐지한 객체의 좌표

            rect_color = (0, 255, 0)
            if is_overlap(object_bbox, (OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2)):
                rect_color = (0, 0, 225) # red
                do_warning_action()
            else:
                rect_color = (0, 255, 0) # green
            # 바운딩 박스 그리기
            
        cv2.rectangle(frame, (5, 5), (635, 635), rect_color, 2)
        cv2.imshow("YOLOv8 Inference", frame)

    else:
        break


# video capture release && 디스플레이 끄기
cap.release()
cv2.destroyAllWindows()
