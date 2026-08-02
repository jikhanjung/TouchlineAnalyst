"""키스톤 캔버스 확장 (src_w / persp_widen) — 상단 보존과 하단 쐐기.

행마다 가로 배율이 다르므로(상단 1/m, 하단 1) 직사각형 출력에서 상단 보존과
하단 채움을 동시에 만족하는 폭은 없다. widen=m 이면 상단이 온전히 남고
하단 좌우가 비며, widen=1 이면 그 반대다 (devlog 091).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.perspective import build_perspective_maps  # noqa: E402

W, H = 400, 200
HORIZON = 60.0
M = 1.3


def test_src_w_default_matches_legacy():
    """src_w 생략 = 기존 동작 (출력 폭 = 소스 폭)."""
    a = build_perspective_maps(W, H, HORIZON, k=0.3, m=M)
    b = build_perspective_maps(W, H, HORIZON, k=0.3, m=M, src_w=W)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_widen_preserves_top_row():
    """widen=m 이면 상단 행이 소스 전폭 [0, W-1] 을 담는다."""
    W2 = int(W * M)
    map_x, _ = build_perspective_maps(W2, H, HORIZON, k=0.0, m=M, src_w=W)
    top = map_x[0]
    assert top[0] == pytest.approx(0.0, abs=0.5)
    assert top[-1] == pytest.approx(W - 1, abs=0.5)


def test_no_widen_crops_top_row():
    """widen=1 이면 상단 행이 소스 중앙 1/m 만 담는다 — 좌우가 잘린다."""
    map_x, _ = build_perspective_maps(W, H, HORIZON, k=0.0, m=M)
    top = map_x[0]
    kept = (top[-1] - top[0]) / (W - 1)
    assert kept == pytest.approx(1.0 / M, rel=0.02)


def test_widen_bottom_row_falls_outside_source():
    """하단 행은 소스 밖을 가리켜야 한다 — 그 자리가 빈 쐐기가 된다."""
    W2 = int(W * M)
    map_x, _ = build_perspective_maps(W2, H, HORIZON, k=0.0, m=M, src_w=W)
    bottom = map_x[-1]
    assert bottom.min() < 0                      # 좌측 쐐기
    assert bottom.max() > W - 1                  # 우측 쐐기
    inside = ((bottom >= 0) & (bottom <= W - 1)).mean()
    assert inside == pytest.approx(1.0 / M, rel=0.05)


def test_center_column_is_fixed_point():
    """심(중앙)은 확장해도 중앙에 남아야 한다 — 아니면 블렌딩이 어긋난다."""
    W2 = int(W * M) & ~1
    map_x, _ = build_perspective_maps(W2, H, HORIZON, k=0.3, m=M, src_w=W)
    cx_out, cx_src = (W2 - 1) / 2.0, (W - 1) / 2.0
    for row in (0, H // 2, H - 1):
        assert np.interp(cx_out, np.arange(W2), map_x[row]) == \
            pytest.approx(cx_src, abs=0.5)


def test_renderer_widen_requires_keystone():
    """m=1 인데 widen>1 이면 검은 띠만 생긴다 — 설정 실수로 보고 막는다."""
    from pystitch.core.lens import LensProfile, builtin_profiles
    from pystitch.core.render import Renderer
    lens = LensProfile.load(builtin_profiles()["GoPro_HERO5_Black_Wide_4K_16x9"])
    R = np.eye(3)
    kw = dict(scale=0.02, feather_px=4)
    with pytest.raises(ValueError, match="persp_widen"):
        Renderer(lens, R, R, -0.4, 0.4, -0.6, 0.15,
                 persp_m=1.0, persp_widen=1.2, **kw)
    with pytest.raises(ValueError, match="persp_widen"):
        Renderer(lens, R, R, -0.4, 0.4, -0.6, 0.15,
                 persp_m=1.3, persp_widen=0.9, **kw)


def test_renderer_widen_changes_out_w_only():
    """widen 은 출력 폭만 바꾼다 — 높이는 그대로."""
    from pystitch.core.lens import LensProfile, builtin_profiles
    from pystitch.core.render import Renderer
    lens = LensProfile.load(builtin_profiles()["GoPro_HERO5_Black_Wide_4K_16x9"])
    R = np.eye(3)
    kw = dict(scale=0.05, feather_px=4, persp_k=0.3, persp_m=M)
    base = Renderer(lens, R, R, -0.4, 0.4, -0.6, 0.15, **kw)
    wide = Renderer(lens, R, R, -0.4, 0.4, -0.6, 0.15, persp_widen=M, **kw)
    assert wide.out_h == base.out_h
    assert wide.out_w == pytest.approx(base.out_w * M, rel=0.02)


def test_map_type_paths_agree():
    """CV_16SC2 맵과 float32 맵이 사실상 같은 출력을 내야 한다.

    Windows 빌드에서 float32 가 1.54배 빨라 플랫폼별로 다른 경로를 쓴다
    (devlog 097). 두 경로가 갈라지면 OS 마다 다른 파노라마가 나온다.
    고정소수점은 좌표를 1/32 px 로 양자화하므로 완전 일치는 아니다.
    """
    import cv2
    from pystitch.core import render as R
    from pystitch.core.lens import LensProfile, builtin_profiles
    lens = LensProfile.load(builtin_profiles()["GoPro_HERO5_Black_Wide_4K_16x9"])
    rot = np.eye(3)
    kw = dict(scale=0.06, feather_px=8)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (lens.height, lens.width, 3), dtype=np.uint8)

    outs = {}
    for fixed in (True, False):
        old = R.USE_FIXED_MAPS
        R.USE_FIXED_MAPS = fixed
        try:
            r = R.Renderer(lens, rot, rot, -0.4, 0.4, -0.6, 0.15, **kw)
            outs[fixed] = r.render(img, img)
        finally:
            R.USE_FIXED_MAPS = old

    a, b = outs[True].astype(np.float64), outs[False].astype(np.float64)
    assert a.shape == b.shape
    mse = ((a - b) ** 2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    assert psnr > 35, f"두 맵 경로가 갈라짐 (PSNR {psnr:.1f}dB)"
