# Docker

> ***TODO list***
> - [ ] Windows에서의 Docker 사용에 대한 조사
> - [x] Docker 환경을 만들고 업로드
> - [x] Docker 환경에 기초적인 의존성 설치
> - [ ] Docker 환경에 NVIDIA Driver와 관련 라이브러리 설치
> - [ ] Docker Hub 접속하기 위한 준비물 안내  
> - [ ] README 작성

## 0. Docker란? 
Docker는 어떤 가상 환경을 만들어서 배포하기 위한 목적으로 사용된다. 원하는 os를 선택해 가상의 환경을 만들고, 원하는 의존성들을 설치해서 나만의 환경을 만들 수 있다.
> [!NOTE]
> * Conda와의 차이점
> Conda 환경은 내 로컬 환경에서, 주로 Python이나 R을 사용해 데이터를 분석하고 딥러닝 / 머신러닝 코드들을 실행시키는 데 사용되는 환경이다. 내가 필요한 의존성만을 Conda 가상 환경에 만들어 주고, 활성화시켜 줄 때마다 현재 프로젝트에 필요한 의존성들을 불러와 쓸 수 있는 식이다.  
> 반면, Docker는 Python에 대한 몇몇 의존성 뿐 아니라, OS수준에서 새로운 환경을 만들어 줄 수 있다는 점이 다르다. 그리고 Git을 사용하듯이 내가 만들거나 수정한 내용들을 Docker Hub에 업로드해두고 사용할 수 있다.
> * Git과의 차이점
> 이는 굳이 보지 않아도 잘 알고 있을 것이라고 생각한다. Git은 어떤 프로젝트에 대해서, 여러 개의 파일을 업로드 / 다운로드하기 편하게 하기 위해 존재한다. 그리고 파일의 크기나 Repository의 크기가 제한적이다.  
> 반면 Docker는 어떤 환경 전체를 이미지의 형식으로 만들어 업로드해 두기에, AI학습을 위한 데이터, CUDA와 PyTorch, Ultralytics의  버전 관리 및 팀원들과의 프로젝트 공유가 용이하다.  
>  

## 1. Docker 사용법
### 0. Docker 설치
먼저, 자신의 OS에 맞추어서 Docker를 설치해야 한다.
#### Linux
```bash
$ curl -fsSL https://get.docker.com/ | sudo sh
```
#### Window & MAC OS
이건 아직 모른다. 조사해서 알게되면 업로드하겠지만, 조사했다고 해서 내가 검증해봤다는 소리는 아니다. 난 리눅스 유저다.  
(+ Window 사용자라면 제발 WSL2를 사용하자. 왜 아직도 가상머신을 쓰는지 모르겠다.)

### 1. Dockerfile 작성
Docker로 어떤 환경을 만들면서 어떤 작업들을 수행할지 정해 주는 파일이다.  
내가 올려 둔 Docker환경은 다음과 같은 Dockerfile을 따라 만들어졌다. ~~필요해보이는건 다 때러박았다~~
```Dockerfile
# Ubuntu 22.04 이미지 사용
FROM ubuntu:22.04

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
	curl \
    wget \
    git \
    build-essential \
    vim \
    python3 \
    python3-pip \
    net-tools \
    iputils-ping \
    unzip \
    tar \
    htop \
    gnupg \
    software-properties-common

WORKDIR /app
```
이렇게 해서 Dockerfile을 모두 작성했다면, 다음의 명령어로 Dockerfile의 의존성을 가지는 환경을 만들 수 있다.  
참고로 나는 Ubuntu-22.04버전의 OS로 가상 환경을 만들었다. 앞으로 PyTorch, CUDA, cudnn, 그리고 ultralytics의 설치를 조사할 예정이다.
```bash
$ docker build -t [Repository]:[tagname] .
$ docker build -t vanillapenguin/yolo-model:v1.0 . # 내가 만든 docker 환경의 예시
```
### 2. Docker 환경을 사용하기
어떤 환경을 사용하고 있으면, 그 환경에 대해 `run`커맨드를 사용해서 그 환경을 활성화해줄 수 있다.  
이때, `-it`은 이 환경에 대해서 지금 터미널 세션을 사용하겠다는 소리이고, 마지막의 `/bin/bash`는 bash shell 로 실행하겠다는 소리이다.  
따라서 아래와 같이 실행하면 내가 만든 환경에 대한 터미널을 사용할 수 있다.
```bash
$ docker run -it [Repository]:[tagname] /bin/bash
$ docker build -t vanillapenguin/yolo-model:v1.0 . # 내가 만든 docker 환경의 예시
```
이미 run으로 활성화된 환경이라면 (이미 돌아가고 있는 환경이라면) 다음과 같은 명령어로 터미널을 켤 수 있다.
```bash
$ docker exec -it my_container /bin/bash
```
### 3. sudo mode에서의 docker 수행 (user 추가)
> 1. 먼저, Docker Group이 있는지 확인해야 한다. 이는 Docker를 설치할 때 자동으로 설치되어야 한다.
>   ```bash
>   $ getent group docker
>   ```
> 2. 현재 사용자를 Docker Group에 넣어 주려면 다음과 같이 써 줄 수 있다.
>   ```bash
>   $ sudo usermod -aG docker $USER
>   ```
> 3. Docker Group의 멤버가 되었는지 확인하려면 다음과 같이 써 주면 된다.
>   ```bash
>   $ groups $USER
>   ```
> 4. Docker 명령어들이 sudo 를 써 주지 않아도 실행되는 것을 확인한다 *(종종 재부팅해야 하는 경우가 있다)*

### 4. docker login및 docker push
Docker login을 해 줘야 Docker Hub에 어떤 이미지를 푸시할 수 있다.  
다음과 같은 명령어로 Docker Login을 해 보자.
```bash
$ docker login
```
명령어를 입력하고 나면 Username과 Password를 입력하라는 창이 나온다. Docker Hub에 로그인했던 이름과 비밀번호를 넣어 주면 된다. 

```bash
$ docker push [Repository]:[tagname]
$ docker push vanillapenguin/yolo-model:v1.0 # 내가 만든 docker 환경의 예시
```

## 2. Docker *(그 외 명령어들)*
* Docker 에서 마운트 사용하기 *( -v flag 사용 )*
```bash
$ docker run -v /path/to/your/repository:/path/in/container -it image_name
```
* Docker Hub 에서 이미지 가져오기 
```bash
$ docker pull username/repository:tag
```
* 현재 Local에서 프로세스가 진행 중인 Docker환경 살펴보기
```bash
$ docker ps -a
```
* 현재 Local에 가지고 있는 Docker Image목록 살펴보기
```bash
$ docker images
```
* 현재 프로세스가 진행 중이 Docker 환경 종료하기
```bash
$ docker stop username/repository:tag
$ docker rm username/repository:tag
```
* 로컬에 가지고 있는 Docker Image 삭제하기
```bash
$ docker rmi username/repository:tag
```