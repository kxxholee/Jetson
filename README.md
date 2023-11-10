# Deep Learning

### YOLOv8
* YOLOv8 을 다운로드하려면 다음과 같은 절차를 따를 수 있다.
* pip과 conda, git을 이용한 다운로드 방법만을 다뤘다.
* Docker 로부터의 다운로드 방법 등 원문을 보려면 [여기](https://docs.ultralytics.com/quickstart/#conda-docker-image) 를 클릭하자.
1. **Python - pip**
  * PyPI를 통한 설치
    ```bash
    $ pip install ultralytics
    ```
  * git을 통한 설치
    ```bash
    $ pip install git+https://github.com/ultralytics/ultralytics.git@main
    ```
2. **Python - conda**
  * 현재 환경에 YOLOv8 설치
    ```bash
    $ conda install -c conda-forge ultralytics
    ```
  * (추천) Pytorch 설치와 함께 YOLOv8설치
    ```bash
    $ conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
    ```
3. **Github**
  * git Repository에서 클론해서 설치
    ```bash
    # Git Repository 가져오기
    $ git clone https://github.com/ultralytics/ultralytics
    ```
    ```bash
    # 클론한 Repository로 진입
    $ cd ultralytics
    ```
    ```bash
    $ pip install -e .
    ```
### 참고
* 팀원들의 역할 분담 및 프로젝트 구성은 [이 문서](./Roles/README.md)를 참고
* 대용량 파일들을 포함한 Git Repository 관리는 [이 문서](./SourceCode/README.md)를 참고