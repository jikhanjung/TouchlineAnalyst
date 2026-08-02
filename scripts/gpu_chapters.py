"""cudacodec 용 챕터 체인 리더 — ChapteredVideo 의 GPU 대응물.

devlog 093 의 GPU 상주 경로용. 정식 채택되면 pystitch/core 로 옮긴다.
CUDA 빌드 OpenCV 필요 (설치는 gpu_pipeline_probe.py 헤더 참고).
"""
import os

# GoPro 챕터는 스트림이 5개다 (video, aac, data×3 — GPMF 포함). cudacodec 의
# FFmpeg 디먹서는 비디오 패킷을 찾다 기본 4096회에서 포기하는데, 그때
# nextFrame() 이 **EOF 와 구분되지 않는 False** 를 낸다 — GP010001.MP4 는
# 기본값에서 30프레임만에 조용히 끊겼다(devlog 094). 잘린 파노라마가
# 오류 없이 나오는 종류의 사고라 한도를 크게 올린다.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")

import cv2  # noqa: E402 — 위 환경변수가 먼저 잡혀야 한다

cc = cv2.cudacodec


class ChapteredGpuReader:
    """cudacodec 용 챕터 체인 리더 — ChapteredVideo 의 GPU 대응물.

    cudacodec 은 파일 하나만 열 수 있어 GoPro 챕터 체인을 못 잇는다.
    프레임 수는 CPU 로 1회 조사하고(디코드 없음), 경계에서 다음 챕터
    리더를 새로 연다. 재오픈 비용은 실측 231~318ms 로 챕터당 1회면
    무시 가능 (devlog 093).
    """

    def __init__(self, files, color=None):
        self.files = [str(f) for f in files]
        self.color = cc.BGR if color is None else color
        self.chapter_frames = []
        for f in self.files:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                raise IOError(f"열 수 없음: {f}")
            self.chapter_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        self.cum = [0]
        for n in self.chapter_frames:
            self.cum.append(self.cum[-1] + n)
        self.total_frames = self.cum[-1]
        self._r = None
        self._chapter = -1
        self._pos = 0
        self.reopens = 0

    def _chapter_of(self, frame):
        for i in range(len(self.files)):
            if frame < self.cum[i + 1]:
                return i
        return len(self.files) - 1

    def _open(self, ci, local):
        p = cc.VideoReaderInitParams()
        p.firstFrameIdx = int(local)
        self._r = cc.createVideoReader(self.files[ci], params=p)
        self._r.set(self.color)
        self._chapter = ci
        self.reopens += 1

    def seek_frame(self, frame):
        frame = max(0, min(int(frame), self.total_frames - 1))
        ci = self._chapter_of(frame)
        self._open(ci, frame - self.cum[ci])
        self._pos = frame

    def read(self):
        """(ok, GpuMat). 챕터 경계를 자동으로 넘어간다."""
        if self._r is None:
            self.seek_frame(self._pos)
        ok, g = self._r.nextFrame()
        if not ok and self._chapter + 1 < len(self.files):
            self._open(self._chapter + 1, 0)
            ok, g = self._r.nextFrame()
        if ok:
            self._pos += 1
        return ok, g

    @property
    def position(self):
        return self._pos
