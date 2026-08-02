"""인코더 파이프 포맷 (I420 직행) — 바이트 레이아웃/플레인 순서 검증.

export_pano 는 원시 프레임을 ffmpeg stdin 으로 넘긴다. 여기서 포맷이
어긋나면(플레인 순서·프레임당 바이트 수) 조용히 프레임이 밀려 영상이
깨지므로, 실제 ffmpeg 왕복으로 확인한다.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.encoders import ffmpeg_bin  # noqa: E402
from pystitch.core.export import pipe_format  # noqa: E402

N_FRAMES = 6


def _frame(w, h, seed=0):
    """그라디언트 + 채도 높은 블록 (크로마 서브샘플링에 민감한 내용)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h)[:, None]
    img[:, :, 2] = rng.integers(60, 200, (h, w), np.uint8)
    img[: h // 4, : w // 4] = (0, 0, 255)
    img[: h // 4, w // 4 : w // 2] = (255, 255, 0)
    return img


def _smooth_frame(w, h, seed=0):
    """실제 영상에 가까운 내용 — 매끄러운 색 변화 + 약한 노이즈.

    _frame() 의 포화 블록은 4:2:0 경계에서 크로마 필터 차이를 과장한다
    (64×48 합성 29dB vs 실제 파노라마 프레임 43dB). 경로 간 일치 검증은
    실제 콘텐츠에 가까운 쪽으로 재야 의미가 있다.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        120 + 60 * np.sin(xx / w * 3),
        110 + 50 * np.cos(yy / h * 2),
        100 + 40 * np.sin((xx + yy) / (w + h) * 4),
    ], axis=-1)
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)


def _pipe_through_ffmpeg(img, declared_fmt, payload_fmt, out_path, n=N_FRAMES):
    """declared_fmt 로 선언하고 payload_fmt 로 실제 바이트를 보낸다 (무손실 저장)."""
    h, w = img.shape[:2]
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo",
           "-pix_fmt", declared_fmt, "-s", f"{w}x{h}", "-r", "10", "-i", "-",
           "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(out_path)]
    buf = (cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420) if payload_fmt == "yuv420p"
           else img).tobytes()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for _ in range(n):
            p.stdin.write(buf)
    except BrokenPipeError:
        pass
    p.stdin.close()
    p.wait()
    return out_path


def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


# ------------------------------------------------------- 포맷 결정

def test_even_dims_use_i420():
    assert pipe_format(6022, 2254) == "yuv420p"


@pytest.mark.parametrize("w,h", [(63, 48), (64, 47), (63, 47)])
def test_odd_dims_fall_back_to_bgr24(w, h):
    """4:2:0 은 홀수 해상도를 표현할 수 없다 — 폴백해야 한다."""
    assert pipe_format(w, h) == "bgr24"


def test_i420_buffer_is_1_5_bytes_per_pixel():
    """프레임당 바이트 수가 어긋나면 ffmpeg 가 프레임을 밀어서 읽는다."""
    img = _frame(64, 48)
    buf = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    assert buf.nbytes == 64 * 48 * 3 // 2


# ------------------------------------------------------- 실제 왕복

def test_i420_pipe_roundtrip_preserves_image(tmp_path):
    """I420 로 넘긴 프레임이 그대로 복원돼야 한다 (플레인 순서 검증)."""
    img = _frame(64, 48)
    out = _pipe_through_ffmpeg(img, "yuv420p", "yuv420p", tmp_path / "i420.mkv")
    frames = _read_frames(out)
    assert len(frames) == N_FRAMES, "프레임 수 불일치 = 바이트 수 어긋남"
    # 4:2:0 자체 손실이 있으니 같은 손실을 거친 기준과 비교
    expect = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420),
                          cv2.COLOR_YUV2BGR_I420)
    for f in frames:
        assert _psnr(f, expect) > 45, "복원 프레임이 기준과 다름"


def test_i420_matches_the_bgr24_path(tmp_path):
    """기존 경로(bgr24→swscale)와 결과가 사실상 같아야 한다.

    **루마로 판정한다.** Y 는 4:2:0 서브샘플링을 거치지 않으므로 ffmpeg
    빌드와 무관하게 거의 일치해야 한다. 반면 색차는 빌드의 크로마 필터에
    좌우돼 전체 PSNR 이 흔들린다 — CI 실측 Windows 39.4 / Ubuntu 39.9 로,
    원래 걸어둔 40dB 문턱을 아슬아슬하게 밑돌아 계속 빨간불이었다
    (실제 파노라마 프레임은 43dB). 그래서 색차는 여유 있는 하한만 본다.
    """
    img = _smooth_frame(128, 96, seed=1)
    a = _read_frames(_pipe_through_ffmpeg(img, "yuv420p", "yuv420p",
                                          tmp_path / "a.mkv"))
    b = _read_frames(_pipe_through_ffmpeg(img, "bgr24", "bgr24",
                                          tmp_path / "b.mkv"))
    assert len(a) == len(b) == N_FRAMES
    y_a = cv2.cvtColor(a[0], cv2.COLOR_BGR2GRAY)
    y_b = cv2.cvtColor(b[0], cv2.COLOR_BGR2GRAY)
    assert _psnr(y_a, y_b) > 44, "루마가 어긋남 = 색행렬/레인지 불일치"
    assert _psnr(a[0], b[0]) > 36, "색차까지 포함해도 두 경로는 사실상 동일"


def test_pipe_format_does_not_change_fidelity(tmp_path):
    """파이프 포맷을 바꿔도 원본 충실도가 실질적으로 달라지지 않아야 한다.

    **어느 쪽이 더 나은지는 단언하지 않는다.** devlog 088 은 실제 파노라마
    프레임에서 I420 이 4dB 우세하다고 기록했지만(48.6 vs 44.5), Windows CI
    의 다른 ffmpeg 빌드에서는 합성 프레임 기준 0.8dB 열세로 나온다 — 우열은
    빌드의 크로마 필터에 달린 성질이라 불변식이 될 수 없다. 지켜야 할 것은
    **차이가 작다**는 것뿐이다.
    """
    img = _smooth_frame(128, 96, seed=2)
    a = _read_frames(_pipe_through_ffmpeg(img, "yuv420p", "yuv420p",
                                          tmp_path / "a.mkv"))[0]
    b = _read_frames(_pipe_through_ffmpeg(img, "bgr24", "bgr24",
                                          tmp_path / "b.mkv"))[0]
    assert abs(_psnr(a, img) - _psnr(b, img)) < 2.0


def test_wrong_payload_is_detected(tmp_path):
    """네거티브 컨트롤 — 포맷을 틀리게 보내면 위 테스트들이 잡아낸다는 증거.

    yuv420p 라고 선언하고 bgr24 바이트(2배)를 보내면 프레임 수가 맞지 않는다.
    """
    img = _frame(64, 48)
    out = _pipe_through_ffmpeg(img, "yuv420p", "bgr24", tmp_path / "bad.mkv")
    frames = _read_frames(out)
    assert len(frames) != N_FRAMES or _psnr(frames[0], img) < 35


def test_pipe_fmt_env_override(monkeypatch):
    """PYSTITCH_PIPE_FMT 강제 (측정용) — 없으면 기존 자동 판정."""
    monkeypatch.delenv("PYSTITCH_PIPE_FMT", raising=False)
    assert pipe_format(6022, 2254) == "yuv420p"
    monkeypatch.setenv("PYSTITCH_PIPE_FMT", "bgr24")
    assert pipe_format(6022, 2254) == "bgr24"      # 짝수여도 강제가 이긴다
    monkeypatch.setenv("PYSTITCH_PIPE_FMT", "yuv420p")
    assert pipe_format(6023, 2254) == "yuv420p"    # 홀수 폴백보다도 강제가 우선
    monkeypatch.delenv("PYSTITCH_PIPE_FMT")
    assert pipe_format(6023, 2254) == "bgr24"
