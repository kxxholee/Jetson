# 개발 환경 세팅 : Run Docker Image
## 1. Docker 권한 설정 (강의자료 참고)
1. 환경 변수 설정 (작업 간편하게)  
<code>t=ultralytics/ultralytics:latest-jetson</code><br>
2. 적절한 플래그와 함께 docker run  
<code>docker run -it -d -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0 --device=/dev/video1 --runtime nvidia --network host --name torch -v $HOME/projects:/workspace $t</code>

> - `docker run`: Docker 컨테이너를 생성하고 실행하는 명령어입니다.
> - `-it`: 대화형 터미널 모드로 실행합니다. `i`는 인터랙티브 모드, `t`는 터미널을 할당합니다.
> - `-d`: 분리 모드(detached mode)로 실행하여, 컨테이너가 백그라운드에서 실행됩니다.
> - `-e DISPLAY=$DISPLAY`: 호스트의 DISPLAY 환경변수를 컨테이너에 전달합니다. GUI 애플리케이션 실행에 필요합니다.
> - `-v /tmp/.X11-unix:/tmp/.X11-unix`: 호스트의 X11 유닉스 소켓 디렉토리를 컨테이너에 마운트합니다. GUI 출력을 위해 필요합니다.
> - `--device=/dev/video0 --device=/dev/video1`: 호스트의 두 웹캠 장치 파일을 컨테이너에 마운트합니다.
> - `--runtime nvidia`: NVIDIA 컨테이너 런타임을 사용하여 GPU 지원을 활성화합니다.
> - `--network host`: 호스트의 네트워크 설정을 컨테이너에서 사용하도록 설정합니다.
> - `--name torch`: 생성되는 컨테이너에 'torch'라는 이름을 부여합니다.
> - `-v $HOME/projects:/workspace`: 호스트의 `$HOME/projects` 디렉토리를 컨테이너의 `/workspace` 디렉토리에 마운트합니다.
> - `$t`: 사용할 Docker 이미지의 이름이나 태그를 지정합니다. (`$t`는 변수로, 실제 이미지 이름/태그로 대체해야 함)

## 2. xhost 명령어 - X서버 접근 권한 설정

<code>xhost +local:root</code>

- `xhost`: X 서버의 접근 제어 목록을 관리하는 프로그램입니다.
- `+local:root`: 로컬에서 실행되는 `root` 사용자에게 X 서버에 접근할 수 있는 권한을 부여합니다. 이는 Docker 컨테이너에서 GUI 애플리케이션을 실행할 때 필요합니다.
- 한번 실행했다고 영구적인 것이 아니므로 재부팅 시마다 실행해 주어야 합니다
