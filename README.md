# Deep Learning
> [!IMPORTANT]
> * 👩‍💻 [역할분담](./Project/RolesGuide.md)
> * 📝 [대용량 파일 관리법](./Memo/LfsGuide.md)
> * 🐳 [Docker 사용법](./Memo/DockerGuide.md)
> * 📝 [최종 레포트 참고자료!!](./Project/REPORT.md)

### YOLOv8
* YOLOv8 을 다운로드하려면 다음과 같은 절차를 따를 수 있다.
* pip과 conda, git을 이용한 다운로드 방법만을 다뤘다.
* Docker 로부터의 다운로드 방법 등 원문을 보려면 [여기](https://docs.ultralytics.com/quickstart/#conda-docker-image) 를 클릭하자.  
<details>
  <summary>
    <strong>🔍 Install Guides</strong>
  </summary>
  <ol>
    <li>
      <details>
        <summary>
          <strong>Python - pip</strong>
        </summary>
          <ol>
            <li>
              PyPI를 통한 설치<br>
              <pre><code>pip install ultralytics</code></pre>
            </li>
            <li>
              Git을 통한 설치
              <pre><code>pip install git+https://github.com/ultralytics/ultralytics.git@main</code></pre>
            </li>
          </ol>
      </details>
    </li>
    <li>
      <details>
        <summary>
          <strong>Python - conda</strong>
        </summary>
          <ol>
            <li>
              현재 환경에 YOLOv8설치<br>
              <pre><code>conda install -c conda-forge ultralytics</code></pre>
            </li>
            <li>
              (추천) Pytorch 설치와 함께 YOLOv8설치
              <pre><code>conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics</code></pre>
            </li>
          </ol>
      </details>
    </li>
    <li>
      <details>
        <summary>
          <strong>Github</strong>
        </summary>
        <ul>
          <li>git Repository에서 클론해서 설치<br>
            <pre><code># Git Repository 가져오기<br>git clone https://github.com/ultralytics/ultralytics</code></pre>
            <pre><code># 클론한 Repository로 진입<br>cd ultralytics</code></pre>
            <pre><code>pip install -e .</code></pre>
          </li>
        </ul>
      </details>
    </li>
  </ol>
</details>
