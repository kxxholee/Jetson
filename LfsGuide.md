# :octocat:**Github** *for* **Large Files**
Git LFS는 대용량의 파일을 별도의 서버에 올리고, 원래 위치에는 포인터를 남긴다.  
즉, 대용량 파일은 다른 서버에서 받아오고 있지만, 포인터가 설정되어 있으므로, 사용자는 git push, git pull등을 그대로 사용할 수 있다

### 0. Git LFS 사용 동기
Raspberry Pi 를 위한 Custom Buildroot를 Github에 업로드하려는 시도를 했었다.
대용량 파일들을 commit 및 push하기 위해 필요했지만, 전체 Repository의 용량 제한으로 실패했다.

### 1. Git LFS 설치하기
```bash
sudo apt-get install git-lfs
```
### 2. Git LFS 사용 선언
어떤 Repository에서 Git LFS를 사용하고 싶다면, 아래의 명령어로 선언할 수 있다
```bash
git lfs install
```
### 3. Git Track 해제
LFS에 올리고 싶은 파일은 Tracking에서 제외해야 한다. 아래의 명령어로 Unstaging을 수행할 수 있다.
```bash
git rm --cached [file-path]
git rm --cached ./workdir/.../something # example
```
이때, LFS에 올려야 할 파일 (100MB보다 큰 파일)은 다음과 같은 명령어로 찾을 수 있다.
```bash
find [/path/to/directory] -type f -size +100M
```
### 4. Git LFS Track 설정
Git에서 Unstaging을 수행했다면, LFS에서 사용할 수 있도록 Tracking을 설정해야 한다.
```bash
git lfs track [file-path]
git lfs track ./workdir/.../something # example
```
### 5. .gitattributes 파일 생성 확인
Git lfs를 설정했다면, `.gitattributes`라는 파일이 설정되고, 이 안에 Git LFS로 관리되는 파일 정보가 저장된다.  
이런 변경 사항을 추가해 주기 위해 다음과 같은 명령을 실행해 준다.
```bash
git add .
git add .gitattributes # repository에서
git commit-m "message"
```
### 6. git push
이렇게 Commit까지 완료했다면, Remote에 push해 줄 수 있다
```bash
git push [remote] [branch]
git push origin master # example
```
