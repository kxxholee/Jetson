import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2 = 220, 160, 420, 430

def key_event_listener(key):
    global OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2
    # >>> 이동 >>>
    if key == ord('w'):
        
        OVERLAY_Y1 = OVERLAY_Y1 - 10 if OVERLAY_Y1 > 10 else OVERLAY_Y1 
        OVERLAY_Y2 = OVERLAY_Y2 - 10 if OVERLAY_Y2 > 10 else OVERLAY_Y2 
        return 0
    elif key == ord('s'):
        OVERLAY_Y1 += 10
        OVERLAY_Y2 += 10
        return 0
    elif key == ord('a'):
        OVERLAY_X1 = OVERLAY_X1 = OVERLAY_X1 - 10 if OVERLAY_X1 > 10 else OVERLAY_X1
        OVERLAY_X2 = OVERLAY_X2 = OVERLAY_X2 - 10 if OVERLAY_X2 > 10 else OVERLAY_X2
        return 0
    elif key == ord('d'):
        OVERLAY_X1 += 10
        OVERLAY_X2 += 10
        return 0
    # <<< 이동 <<<
    # >>> zoom >>>
    elif key == ord('r'):
        OVERLAY_X1 = OVERLAY_X1 - 10 if OVERLAY_X1 > 10 else OVERLAY_X1
        OVERLAY_Y1 = OVERLAY_Y1 - 10 if OVERLAY_Y1 > 10 else OVERLAY_Y1 
        OVERLAY_X2 += 10
        OVERLAY_Y2 += 10
        return 0
    elif key == ord('f'):
        OVERLAY_X1 += 10
        OVERLAY_Y1 += 10
        OVERLAY_X2 = OVERLAY_X2 - 10 if OVERLAY_X2 > 10 else OVERLAY_X2
        OVERLAY_Y2 = OVERLAY_Y2 - 10 if OVERLAY_Y2 > 10 else OVERLAY_Y2
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
        if key_event_listener(cv2.waitKey(1) & 0xFF) == 1:
            break;
        overlay = frame.copy()
        cv2.rectangle(overlay, (OVERLAY_X1, OVERLAY_Y1), (OVERLAY_X2, OVERLAY_Y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)
        # <<< overlay 되는 부분 표시 <<<

        for detection in results[0]:            # result[0] -> detection정보로부터
            boxs = detection.boxes              # detect한 box를 꺼내고
            x1, y1, x2, y2 = boxs.xyxy[0]       # xy좌표를 꺼냄

            object_bbox = (int(x1), int(y1), int(x2), int(y2))  # 탐지한 객체의 좌표

            if is_overlap(object_bbox, (OVERLAY_X1, OVERLAY_Y1, OVERLAY_X2, OVERLAY_Y2)):
                rect_color = (0, 0, 225) # red
                do_warning_action()
            else:
                rect_color = (0, 255, 0) # green
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3]), rect_color, 2)
            textInfo = {
                "img" : frame,
                "text" : detection.names[0],
                "org" : (object_bbox[0], object_bbox[1]),
                "fontFace" : cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale" : 1,
                "color" : (0, 0, 0),
                "thickness" : 2,
                "lineType" : cv2.LINE_AA
            }

            text_size, _ = cv2.getTextSize(textInfo["text"], textInfo["fontFace"], 
                                           textInfo["fontScale"], textInfo["thickness"])

            # 텍스트의 배경의 될 사각형의 각 점 좌표 계산
            start_x, start_y = textInfo["org"]
            bg_width, bg_height = text_size
            bg_start_x, bg_start_y = start_x, start_y - bg_height

            # 배경 사각형 그리기
            cv2.rectangle(frame, (bg_start_x, bg_start_y), (bg_start_x + bg_width, bg_start_y + bg_height), rect_color, -1)

            # 텍스트 집어넣기
            cv2.putText(img=textInfo["img"], text=textInfo["text"], 
                        org=textInfo["org"], fontFace=textInfo["fontFace"], 
                        fontScale=textInfo["fontScale"], color=textInfo["color"], 
                        thickness=textInfo["thickness"], lineType=textInfo["lineType"]) 

        cv2.imshow("YOLOv8 Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# video capture release && 디스플레이 끄기
cap.release()
cv2.destroyAllWindows()
