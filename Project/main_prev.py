import cv2
from ultralytics import YOLO

'''
    yolo export model=backup-latest.pt format=engine
    main.py
    훈련된 모델을 바탕으로 openCV를 사용해 객체 탐지를 수행하는 코드

    동영상 파일 사용시
    video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(video_path)
'''


# 훈련된 YOLO 모델 불러오기
model = YOLO('/workspace/Ultra/backup-latest.engine')
# model = YOLO('/workspace/Ultra/runs/detect/yolomodeln.pt5/weights/best.pt')

# webcam 사용시
cap = cv2.VideoCapture(0)

# 웹캠 프레임 크기 조정 (640x480)
# webcam에서 받아오는 프레임이 너무 큰 문제 존재 -> 화면 크기 조절
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 웹캠 프레임 속도 조정 (30 fps)
# 화면 초당 프레임을 적절히 조절함
cap.set(cv2.CAP_PROP_FPS, 30)

# 'q' 누르면 끝나는 loop
while cap.isOpened():
    # cap에서 읽기 성공여부, 현재의 frame 받기
    success, frame = cap.read()

    if success:
        # 훈련된 YOLO 모델에 프레임 입력, 결과 받기
        results = model(frame)

        # 현재 frame에 올릴 label 정보 받아오기
        annotated_frame = results[0].plot()

        # label정보가 annotation된 frame 출력
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 반복문 종료 조건 : 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 비디오 못읽어왔으면 끄기
        break

# capture obj release 및 활성 윈도우 끄기
cap.release()
cv2.destroyAllWindows()