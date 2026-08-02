# 093 — NVDEC/cudacodec 검증 + GPU 상주 스티칭 프로토타입

2026-08-02. TODOs 의 "NVDEC 은 보류 — 이 OpenCV 는 CUDA 없이 빌드돼 프로세스
내 경로가 없다"를 실제로 뚫었다. **보류 해제 근거가 나왔다.**

## 요약

| 항목 | 결과 |
|---|---|
| WSL2 NVDEC | **된다** (`/usr/lib/wsl/lib/libnvcuvid.so`) |
| 9p 가 병목인가 | **아니다** (120MB/s 실측 vs 4K h264 필요량 7.5MB/s) |
| CUDA OpenCV | cudawarped 4.13.0.90, CUDA Toolkit 설치 **불필요** |
| cudacodec 디코드 | CPU 디코더와 **비트정확 일치** |
| cudacodec 시크 | `firstFrameIdx` **프레임 정확**, 231~318ms |
| GPU 상주 파이프라인 | **wall 2.29배, CPU 7.1배 절감** (NVENC 포함하고도) |

**모든 벤치는 `pano_0002` 스티칭이 동시에 도는 상태**에서 쟀다 — 경합이 낀
보수적 값이다.

## 9p 는 병목이 아니었다

WSL 검증을 꺼릴 이유가 9p 라고 봤는데 실측은 반대였다. 같은 30초 4K 구간:

| 디코드 | ext4 wall | 9p wall | ext4 CPU | 9p CPU |
|---|---|---|---|---|
| 소프트웨어 | 11.6s | 13.4s | 47.7s | 46.2s |
| NVDEC + CPU 다운로드 | 6.7s | 6.3s | 6.4s | 4.4s |
| NVDEC GPU 유지 | 4.9s | 5.2s | 2.1s | 1.7s |

9p 와 ext4 가 오차 범위 안이다. 231MB 를 1.9초에 읽었으니 **약 120MB/s**,
4K h264 는 **7.5MB/s** 면 된다. CLAUDE.md 의 9p 경고는 100GB 급 쓰기·랜덤
시크 얘기지 순차 디코드 읽기에는 해당하지 않는다.

## 설치 — CUDA Toolkit 없이

[cudawarped/opencv-python-cuda-wheels](https://github.com/cudawarped/opencv-python-cuda-wheels)
4.13.0.90. 릴리스 노트는 "GPU Computing Toolkit v13.1 필요"라고 하지만
**apt 로 3GB 를 깔 필요가 없었다**:

- torch(2.13.0+cu130)가 이미 pip 로 CUDA 13.0 런타임을 끌고 와 있고, CUDA 는
  마이너 버전 호환이라 13.1 빌드가 그대로 붙는다
- 빠진 건 **NPP** 뿐 → `pip install nvidia-npp`
- 이 wheel 은 **Ubuntu 22.04 빌드**라 24.04 에선 ffmpeg 4.x 5종이 없다
  (`libavcodec.so.58`, `libavdevice.so.58`, `libavformat.so.58`,
  `libavutil.so.56`, `libswscale.so.5`) → **micromamba 로 별도 prefix** 에
  `ffmpeg=4.4` 만 받아 `LD_LIBRARY_PATH` 에 붙였다. 시스템 무변경

> wheel 파일명을 바꾸면 pip 가 거부한다 (`not a valid wheel filename`).

결과: `cv2 4.13.0`, CUDA device 1, `sm_75`, Driver/Runtime **13.10**,
`cudacodec`·`cuda.remap`·`cudacodec.VideoWriter` 모두 사용 가능.

### 4.13 다운그레이드는 안전하다 (5.0.0 → 4.13)

CUDA wheel 은 4.x 뿐이라 확인이 필요했다. PyPI 의 같은 4.13.0.90 으로 cv2 만
그림자 설치해 검증:

- **테스트 182건 전부 통과**
- 실제 렌더 비교 PSNR **48.7dB**, diff>32 인 130픽셀이 **전부 마스크/쐐기
  경계 7×7 이웃 안** (경계 밖 0픽셀)

## cudacodec 은 비트정확하다 — 함정 하나 주의

처음엔 GPU/CPU 프레임이 PSNR 30dB 로 어긋나 색공간 문제로 의심했다.
그런데 **H.264 디코드는 규격상 비트정확**이므로 그럴 리가 없다. 원시 YUV 로
직접 비교하니 원인이 나왔다:

```
GPU[0] == CPU[8]  (비트정확)
GPU[1] == CPU[9]  (비트정확)
```

**8프레임 오프셋.** 그런데 이건 내가 `ffmpeg -ss 300 -t 30 -c copy` 로 만든
테스트 클립의 컨테이너 아티팩트였다. **원본 GoPro 파일에서는 완벽하다**:

```
GPU[0..4] == CPU[0..4]        (순차, 비트정확)
firstFrameIdx=5  -> CPU[5]     (시크, 비트정확)
firstFrameIdx=20 -> CPU[20]
```

교훈: `-ss ... -c copy` 로 자른 클립은 디코더 간 시작 프레임 기준이 달라져
**비교 실험의 기준으로 쓰면 안 된다.**

### 남은 차이는 색변환뿐

`ColorFormat_BGR` 로 받으면 CPU 대비 PSNR 40.5dB(평균차 1.86). 소스는
`bt709` + `color_range=pc`(풀레인지)인데 이 빌드의
`VideoReader.set(colorFormat, bitDepth, planar)` 에는 **색공간 인자가 없다**
(기본 BT.601 추정). YUV 자체는 비트정확하므로 **`NV_YUV_SURFACE_FORMAT` 으로
받아 우리가 직접 변환하면 해소된다.** devlog 088 에서 인코더 입력을 이미
I420 으로 바꿨으니, YUV 를 끝까지 들고 가면 변환 자체가 사라지는 그림도 된다.

### 시크 비용

`firstFrameIdx` 는 디코더 재생성이라 **231~318ms**. 순차 소비(스티칭·분석)엔
무관하지만 **GUI 스크러빙엔 느리다** — 랜덤 접근은 `ChapteredVideo` 를
유지하고 순차 경로만 GPU 로 가는 분리가 맞다.

## GPU 상주 프로토타입 (`scripts/gpu_pipeline_probe.py`)

`render.py` 를 안 건드리고 별도 경로로. 정합·맵·심 가중치는 기존 `Renderer`
에서 그대로 꺼내 써 기하는 동일하다. 채널 게인은 블렌딩 가중치에 미리
접어넣어 프레임당 곱셈 2회를 없앴다.

NVDEC → `cv2.cuda.remap` ×2 → GPU 블렌딩 → NVENC. 5976×2306, 60프레임:

| 경로 | wall | fps | CPU |
|---|---|---|---|
| CPU (현행, **인코드 제외**) | 7.83s | 7.66 | 18.15s |
| GPU (**NVENC 인코드 포함**) | 3.43s | 17.52 | **2.54s** |

**wall 2.29배 빠르고 CPU 7.1배 적은데, GPU 쪽만 인코딩까지 한다.** 출력은
PSNR 40.5dB 로 일치하고(차이는 위의 색변환), hevc 5976×2306 60프레임이
정상 디코드된다. 심(하프라인)도 육안으로 깨끗하다.

산출물: `F:\Pictures\20260801_GoPro\gpu_vs_cpu_crop.jpg`,
`gpu_pipeline_sample.mp4`

## 의미

devlog 088 이 스티칭을 "싱글스레드 CPU + 파이프 복사에 묶여 있다"고 진단했는데,
이 경로는 **그 둘을 통째로 없앤다** — 프레임이 GPU 를 안 떠나므로 708MB/s
파이프도, 소프트 디코드도, CPU remap 도 사라진다. 088 의 I420 파이프 최적화가
겨냥한 병목 자체가 없어지는 셈이다.

8K 검토(devlog 미작성, 대화상)와도 직결된다. 8K 두 스트림 소프트 디코드는
현재 CPU 로 불가능하지만 NVDEC 이면 가능 범위다.

## 다음

- 색변환을 `NV_YUV_SURFACE_FORMAT` + 자체 BT.709 풀레인지 변환으로 (또는
  YUV 상주). 지금 40.5dB 차이가 팀 분류 같은 색 기반 로직에 영향을 줄 수 있다
- 세그먼트 전환·챕터 경계 처리 (`ChapteredVideo` 상당물이 cudacodec 엔 없다).
  챕터마다 리더를 새로 여는 비용은 231~318ms 라 무시 가능
- Windows 적용은 별개 작업 — win_amd64 wheel + CUDA Toolkit 13.1 설치 필요.
  스티칭만 WSL 에서 돌리는 선택지도 있다 (9p 가 병목이 아님이 확인됐으므로)
- 정식 채택하려면 `render.py` 에 GPU 경로를 넣고 CUDA 미설치 환경 폴백을
  유지해야 한다 — `pitchstitch.py` 가 torch 없이도 기동하는 것과 같은 원칙
