# 전기전자 심화 설계 09조

> [!NOTE]
> 202110965 이관호 작성

### Repository Structure
| 파일명 | 역할 | 준비물 |
|--------|------|--------|
| [train.py](./train.py) | YOLOv8 모델을 훈련 | `datasets.yaml`, `scratch.yaml` |
| [json2txt.py](./json2txt.py) | json 이미지 데이터를 txt로 변환 | `jpg`, `json`으로 구성된 데이터셋 |
| [main.py](./main.py) | 훈련된 yolo모델을 기반으로 camera를 통해 detection | 훈련된 YOLO 모델 |
| [classcount.py](./classcount.py) | 주어진 데이터셋에서 class의 개수를 센다 | `json`으로부터 변환된 `txt`파일 |
| [filefilter.sh](./filefilter.sh) | 데이터셋에서부터 학습에 필요없는 class를 가진 파일들 삭제 | `json`으로부터 변환된 `txt`파일 |
| [datasets.yaml](./datasets.yaml) | 이 레포지토리에서 dataset들에 대한 정리 | 없음 |
| [scratch.yaml](./scratch.yaml) | 이 모델에서 사용할 아키텍처가 정의되어 있음 | 없음 |

* [train.py](./train.py) : yolov8 모델을 학습시키는 코드이다. 웬만하면 주석에 써져 있다. dataset에 대한 정보로 datasets.yaml을 입력받고, model의 구조에 대한 정보로 scratch.yaml을 입력받는다.
* [json2txt.py](./json2txt.py) : AIhub에서 구한 데이터셋은 json의 형태로 이미지에 대한 annotation이 되어 있다. json파일의 각 태그를 읽어서 이를 YOLO학습에 알맞게 txt파일로 변환해 준다.
* [main.py](./main.py) : 이미 학습된 모델을 통해서, 웹캠에서의 비디오 입력으로부터 detect해주는 코드이다.
* [classcount.py](./classcount.py) : 데이터셋의 경로를 받아서, 이 경로에 있는 txt파일들에 대해 몇 개의 class가 존재하고, class별로 몇 개의 파일들이 이 class를 가지는지 출력해 준다.
* [filefilter.sh](./filefilter.py) : 어떤 데이터셋에서, 모든 txt파일들을 검사해서, 학습 및 검증과 관련없는 class를 가진 txt파일과, 동명의 json, jpg파일들을 삭제한다.
* [datasets.yaml](./datasets.yaml) : 데이터셋이 어떤 구조를 가지는지 정의한 yaml파일로, 데이터셋의 루트 경로 `path`와. 이 경로로부터 상대적인 학습 및 검증 데이터 `train`, `val`을 가진다. 추가로, 분류할 class의 수 `nc`와 각 클래스와 대응되는 이름이 정의된 리스트 `names`를 가진다.
* [scratch.yaml](./scratch.yaml) : class의 수 `nc`와 `scales` 등의 값들을 가지고 있다. (저도 아주 자세히는 모르니까 조사해서 쓰세요)

### 작업 순서
1. aihub에서 데이터 다운로드를 용이하게 해 주는 실행파일 다운로드
2. 실행파일로 PC(노트북)에 데이터셋 설치 후 모두 압축 해제
3. 이때, 1.Train과 2.Validation에 들어 있는 `원천데이터`와 `라벨링데이터` 내부의 압축 파일들을 모두 1.Train, 2.Validation에 압축 해제하고, 이 디렉토리들의 이름을 train, val로 수정
4. train과 val에 있는 json파일들에 대해서 json2txt.py를 사용해 txt파일들 생성(모두)
5. `rsync`의 `ssh`연결을 통해서 PC(노트북)에서의 데이터를 모두 Jetson으로 이동
7. Jetson에서 Ultralytics Docker Image 로부터 만들어진 Container에서, 공유 폴더에 있는 코드와 데이터셋으로 학습 진행
8. Jetson의 ultralytics관련 configuration을 수정해 줌 : 데이터셋과 모델 저장 위치에 대해서
9. 여러 차례에 거친 학습 진행. Epoch도 굉장히 많음

### 데이터셋 출처
1. 사용한 데이터셋의 링크는 [여기](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=645) 클릭  
2. Linux환경에서 aihubshell을 사용해서 데이터셋을 가져올 수 있음. 자세한 내용은 aihubshell 사용 관련 [공식문서](https://aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100) 확인하기

### YOLO 에서 지원하는 데이터셋 직렬화 (TensorRT)
```bash
$ yolo export model=backup-latest.pt format=engine
```

### 추가자료

1. main.py작성을 위한 YOLO 리턴값 분석 : [https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.to](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.to)


