"""FrameSource — NVDEC/CPU 공통 인터페이스 (devlog 105).

NVDEC 이 없는 환경(CI 포함)에서도 **CPU 폴백이 정확히 같은 프레임**을
내야 한다. 분석 루프가 이 껍데기 위에서 돌기 때문이다.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.encoders import ffmpeg_bin  # noqa: E402
from pystitch.core.gpudecode import FrameSource, nvdec_available  # noqa: E402

W, H, N = 320, 240, 30


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """프레임마다 다른 무늬 — 프레임이 밀리면 바로 드러난다."""
    d = tmp_path_factory.mktemp("gpudec")
    out = d / "clip.mp4"
    src = d / "raw.bin"
    with src.open("wb") as f:
        for i in range(N):
            img = np.full((H, W, 3), i * 8 % 256, np.uint8)
            img[: H // 2, : W // 2] = (i * 3 % 256, 255 - i * 5 % 256, 40)
            f.write(img.tobytes())
    r = subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", str(src),
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(out)],
        capture_output=True)
    if r.returncode != 0 or not out.exists():
        pytest.skip("ffmpeg 로 테스트 클립을 만들 수 없음")
    return out


def _read_all(fs, every=1):
    got = []
    i = 0
    while True:
        if not fs.grab():
            break
        if i % every == 0:
            ok, f = fs.retrieve()
            if not ok:
                break
            got.append(f)
        i += 1
    return got


def test_cpu_backend_reads_all_frames(clip):
    fs = FrameSource(clip, prefer_gpu=False)
    try:
        assert fs.backend == "cpu"
        frames = _read_all(fs)
    finally:
        fs.release()
    assert len(frames) == N
    assert frames[0].shape == (H, W, 3)


def test_retrieve_every_nth_matches_videocapture(clip):
    """분석과 같은 패턴 — 모든 프레임 grab, N 번째만 retrieve."""
    fs = FrameSource(clip, prefer_gpu=False)
    try:
        got = _read_all(fs, every=3)
    finally:
        fs.release()
    cap = cv2.VideoCapture(str(clip))
    ref, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % 3 == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            ref.append(f)
        i += 1
    cap.release()
    assert len(got) == len(ref) > 0
    for a, b in zip(got, ref):
        assert np.array_equal(a, b), "grab/retrieve 정렬이 어긋남"


def test_seek_frame_lands_on_the_same_frame(clip):
    """체크포인트 재개 경로 — 시크 후 첫 프레임이 정확해야 한다."""
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, ref = cap.read()
    cap.release()
    assert ok
    fs = FrameSource(clip, prefer_gpu=False)
    try:
        fs.seek_frame(10)
        assert fs.grab()
        ok, got = fs.retrieve()
    finally:
        fs.release()
    assert ok and np.array_equal(got, ref)


def test_get_returns_metadata(clip):
    fs = FrameSource(clip, prefer_gpu=False)
    try:
        assert int(fs.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
        assert int(fs.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H
    finally:
        fs.release()


def test_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        FrameSource(tmp_path / "없는파일.mp4", prefer_gpu=False)


def test_nvdec_disabled_by_env(monkeypatch):
    monkeypatch.setenv("PYSTITCH_NVDEC", "0")
    assert nvdec_available() is False


@pytest.mark.skipif(not nvdec_available(), reason="NVDEC 없음 (CPU 전용 환경)")
def test_nvdec_matches_cpu_frame_alignment(clip):
    """NVDEC 이 있으면 **같은 순서의 같은 프레임**이어야 한다.

    색변환은 CPU 디코더와 미세하게 다르므로(devlog 105) 비트 비교 대신
    프레임 정렬을 본다 — 밀리면 무늬가 통째로 어긋나 크게 벌어진다.
    """
    fs = FrameSource(clip, prefer_gpu=True)
    try:
        if fs.backend != "nvdec":
            pytest.skip("이 파일은 NVDEC 로 못 연다")
        got = _read_all(fs, every=3)
    finally:
        fs.release()
    fs2 = FrameSource(clip, prefer_gpu=False)
    try:
        ref = _read_all(fs2, every=3)
    finally:
        fs2.release()
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        d = np.abs(a.astype(int) - b.astype(int)).mean()
        assert d < 12, f"프레임이 어긋난 것으로 보임 (평균차 {d:.1f})"


@pytest.mark.parametrize("tag,expect", [
    ("hevc", True),    # opencv 4.13 실측 — 코덱 이름 그대로 나온다
    ("hvc1", True), ("hev1", True), ("h265", True), ("HEVC", True),
    ("h264", False), ("avc1", False), ("mp4v", False), ("", False),
])
def test_hevc_tag_detection(tag, expect):
    """빌드마다 FOURCC 표기가 달라 hevc 를 h264 로 오인하면 NVDEC 을 놓친다."""
    from pystitch.core.gpudecode import _is_hevc_tag
    assert _is_hevc_tag(tag) is expect


def test_wide_hevc_is_allowed_but_wide_h264_is_not():
    """폭 한계는 코덱마다 다르다 (h264 4096 / hevc 8192)."""
    from pystitch.core.gpudecode import NVDEC_H264_MAX_W, NVDEC_HEVC_MAX_W
    assert NVDEC_H264_MAX_W < 5976 <= NVDEC_HEVC_MAX_W
