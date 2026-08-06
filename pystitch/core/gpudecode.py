"""분석 입력 디코드를 NVDEC 으로 — 없으면 조용히 CPU 로 폴백 (devlog 105).

분석은 파노라마(6K hevc)의 **모든 프레임**을 grab 하고 detect_every 마다
retrieve 한다. 6K hevc 소프트 디코드는 프레임당 74ms(3.8코어)로 재분석
배치 100시간+ 의 큰 몫이다. NVDEC 은 같은 일을 0.99코어로 한다.

실측 (pano_0001, 5976×2306 hevc, 3000프레임, detect_every=3):

    CPU (현행)                 51.5fps   CPU 221.8s (3.81코어)
    NVDEC + 3프레임마다 다운로드  109.2fps   CPU  27.3s (0.99코어)
    NVDEC GPU 상주(참고)        229.1fps   CPU   7.4s (0.57코어)

→ **wall 2.1배, CPU 8.1배 절감.** 스티칭에서는 NVDEC 이득이 원시 프레임
전송에 먹혔지만(devlog 101), 분석은 3프레임 중 1장만 내리므로 남는다.

**색차 주의:** cudacodec 의 BGR 변환은 CPU 디코더와 다르다 (파노라마 실측
PSNR 42.6dB, HSV 로 H 1.2 / S 3.6 / V 2.6 차이). 팀 분류는 트랙릿 간
*상대* 색으로 군집하므로 전역 이동은 사실상 무해하지만, `.analysis.json`
에 기록되는 값이 달라진다 — **한 파일 안에서 디코더를 섞지 말 것**
(`--far-augment` 는 원본과 같은 백엔드를 써야 한다).
"""
from __future__ import annotations

import os

# GoPro/파노라마 컨테이너의 부가 스트림 때문에 FFmpeg 디먹서가 비디오
# 패킷을 찾다 기본 4096회에서 포기하고 **EOF 와 구분되지 않는 False** 를
# 낸다 (devlog 094 에서 GP010001 이 30프레임만에 조용히 끊겼다).
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")

import cv2  # noqa: E402 — 위 환경변수가 먼저 잡혀야 한다


# NVDEC 하드웨어 디코드 폭 한계 — 인코더(devlog 090)와 같은 구조다.
# 넘으면 createVideoReader 가 assertion 으로 죽으면서 OpenCV ERROR 를
# 시끄럽게 뱉으므로 **미리 거른다**. 구 파노라마는 h264 6348px 라
# 여기 걸린다 (pano_0374 실측) — hevc 로 재인코드된 것만 NVDEC 을 탄다.
NVDEC_H264_MAX_W = 4096
NVDEC_HEVC_MAX_W = 8192


def _is_hevc_tag(tag: str) -> bool:
    """FOURCC 문자열이 HEVC 인가.

    빌드마다 다르게 나온다 — 컨테이너 태그(`hvc1`/`hev1`)일 수도, **코덱
    이름 그대로**(`hevc`)일 수도 있다. 실측(opencv 4.13)은 후자였고,
    이걸 놓쳐 hevc 파노라마가 h264 한계(4096)에 걸려 NVDEC 을 건너뛰었다.
    """
    t = (tag or "").strip().strip("\x00").lower()
    return t.startswith(("hev", "hvc")) or t in ("h265", "x265")


def _too_wide_for_nvdec(path) -> str | None:
    """NVDEC 로 못 여는 크기/코덱이면 사유 문자열, 가능하면 None."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None                   # 열기 실패는 아래에서 처리
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cc4 = int(cap.get(cv2.CAP_PROP_FOURCC))
        tag = "".join(chr((cc4 >> (8 * i)) & 0xFF) for i in range(4))
    finally:
        cap.release()
    is_hevc = _is_hevc_tag(tag)
    lim = NVDEC_HEVC_MAX_W if is_hevc else NVDEC_H264_MAX_W
    if w > lim:
        return f"폭 {w}px > {'hevc' if is_hevc else 'h264'} NVDEC 한계 {lim}"
    return None


def nvdec_available() -> bool:
    """cudacodec 으로 디코드할 수 있는가. `PYSTITCH_NVDEC=0` 이면 끈다."""
    if os.environ.get("PYSTITCH_NVDEC") == "0":
        return False
    try:
        return (hasattr(cv2, "cudacodec")
                and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:                     # noqa: BLE001 — 없으면 그냥 CPU
        return False


class FrameSource:
    """`cv2.VideoCapture` 의 grab/retrieve/seek 를 그대로 흉내내는 리더.

    NVDEC 이 가능하면 그쪽을, 아니면 VideoCapture 를 쓴다. 호출부는
    백엔드를 몰라도 된다 — 분석 루프를 그대로 두기 위한 얇은 껍데기다.

    NVDEC 은 `grab()` 에서 GpuMat 을 들고만 있다가 `retrieve()` 에서
    내린다. 즉 **버리는 프레임은 GPU 를 떠나지 않는다** — 이득의 원천.
    """

    def __init__(self, path, prefer_gpu=True, log=None):
        self.path = str(path)
        self._log = log
        self._pending = None              # NVDEC: 직전 grab 의 GpuMat
        self.backend = "cpu"
        self._r = None
        why = _too_wide_for_nvdec(self.path) if prefer_gpu else None
        if why and log:
            log(f"[decode] NVDEC 건너뜀 ({why}) — CPU")
        if prefer_gpu and not why and nvdec_available():
            try:
                self._open_gpu(0)
                self.backend = "nvdec"
            except Exception as e:        # noqa: BLE001 — 폴백이 정상 경로
                if log:
                    log(f"[decode] NVDEC 사용 불가 ({type(e).__name__}) — CPU")
                self._r = None
        if self._r is None:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise IOError(f"열 수 없음: {self.path}")
        if log:
            log(f"[decode] 백엔드: {self.backend}")

    # ---------------------------------------------------------- 내부
    def _open_gpu(self, first_frame: int):
        cc = cv2.cudacodec
        p = cc.VideoReaderInitParams()
        p.firstFrameIdx = int(first_frame)
        self._r = cc.createVideoReader(self.path, params=p)
        self._r.set(cc.BGR)
        self._pending = None

    # ---------------------------------------------------------- 공개 API
    def grab(self) -> bool:
        if self._r is not None:
            ok, g = self._r.nextFrame()
            self._pending = g if ok else None
            return bool(ok)
        return bool(self._cap.grab())

    def retrieve(self):
        """(ok, BGR ndarray). grab() 직후에만 유효."""
        if self._r is not None:
            if self._pending is None:
                return False, None
            arr = self._pending.download()
            if arr is not None and arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return arr is not None, arr
        return self._cap.retrieve()

    def seek_frame(self, frame: int):
        """절대 프레임으로 이동 (체크포인트 재개용)."""
        if self._r is not None:
            self._open_gpu(frame)        # cudacodec 은 재오픈이 곧 시크
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))

    def get(self, prop):
        """CAP_PROP_* 조회 — NVDEC 경로도 메타는 VideoCapture 로 읽는다."""
        if self._r is None:
            return self._cap.get(prop)
        cap = cv2.VideoCapture(self.path)
        try:
            return cap.get(prop)
        finally:
            cap.release()

    def release(self):
        self._pending = None
        self._r = None
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
