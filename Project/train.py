from ultralytics import YOLO
import os
import argparse

'''
    모델 훈련시키는 코드
    YOLOv8모델 사용

    model configuration : scratch.yaml 에 적힌 구조에 따라 모델을 디자인하고 훈련함
                          이미 train한적이 있다면 (path가 존재한다면) 그 모델을 불러와서 훈련 재개
                          ./run/detect/yolomodel.pt{숫자}/weights/best.pt에 존재 (혹은 latest)
    
    model dataset       : dataset.yaml 에 모델이 사용할 데이터셋의 경로와, 구별해야 하는 class, 
                          그리고 각 class가 무엇을 의미하는지 정의되어 있음

'''

MODEL_LOAD = "./savepoint.pt"
MODEL_START = "./scratch.yaml"

# Load the model.
model:YOLO
modelExists:bool
if modelExists := os.path.isfile(MODEL_LOAD):
    model = YOLO(model=MODEL_LOAD)
    print("Resume training from checkpoint!")
else:
    model = YOLO(model=MODEL_START)
    print("Building Model from scratch!")

# 훈련
results = model.train(
   data='datasets.yaml',
   # imgsz=1280, # image size는 그냥 기본값을 사용함
   epochs=3,
   batch=8,
   name='yolomodeln.pt', # runs/detect/yolomodeln.pt(숫자) 에 저장됨
   resume=modelExists    # 훈련 이력이 있다면 훈련 재개
)
