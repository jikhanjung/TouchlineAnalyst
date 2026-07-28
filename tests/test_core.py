"""pystitch.core 유닛 테스트 (영상 파일 불필요 — 합성 데이터/실제 렌즈 프로파일 사용)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.align import Alignment  # noqa: E402
from pystitch.core.chapters import find_chapters, group_directory  # noqa: E402
from pystitch.core.geometry import (  # noqa: E402
    estimate_relative_rotation, kabsch, pixel_to_ray, ray_to_pixel, rot_xz,
)
from pystitch.core.lens import LensProfile, builtin_profiles  # noqa: E402
from pystitch.core.project import (  # noqa: E402
    alignment_from_dict, alignment_to_dict, load_project, save_project,
)
from pystitch.core.render import seam_weights  # noqa: E402


@pytest.fixture(scope="module")
def lens():
    profiles = builtin_profiles()
    assert profiles, "내장 렌즈 프로파일 없음"
    return LensProfile.load(next(iter(profiles.values())))


# ---------------------------------------------------------------- chapters

def test_find_chapters_orders_gopro_chain(tmp_path):
    for name in ["GP020395.MP4", "GOPR0395.MP4", "GP010395.MP4",
                 "GOPR0001.MP4", "GP010001.MP4", "GP100395.MP4"]:
        (tmp_path / name).touch()
    chain = [p.name for p in find_chapters(tmp_path / "GOPR0395.MP4")]
    assert chain == ["GOPR0395.MP4", "GP010395.MP4", "GP020395.MP4", "GP100395.MP4"]


def test_find_chapters_non_gopro_is_single(tmp_path):
    f = tmp_path / "match.mp4"
    f.touch()
    assert find_chapters(f) == [f]


def test_group_directory(tmp_path):
    for name in ["GOPR0395.MP4", "GP010395.MP4", "GOPR0001.MP4"]:
        (tmp_path / name).touch()
    groups = group_directory(tmp_path)
    assert [[p.name for p in g] for g in groups] == [
        ["GOPR0001.MP4"], ["GOPR0395.MP4", "GP010395.MP4"]]


# ---------------------------------------------------------------- geometry

def test_ray_pixel_roundtrip(lens):
    pts = np.array([[1920.0, 1080.0], [800.0, 500.0], [3000.0, 1700.0]])
    rays = pixel_to_ray(pts, lens)
    back, valid = ray_to_pixel(rays, lens)
    assert valid.all()
    assert np.allclose(back, pts, atol=0.05)


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R_true = rot_xz(0.3, -0.1)
    b = a @ R_true.T
    R = kabsch(a, b)
    assert np.allclose(R, R_true, atol=1e-10)


def test_ransac_rotation_with_outliers():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(200, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R_true = rot_xz(0.25, 0.05)
    b = a @ R_true.T
    b[:40] = rng.normal(size=(40, 3))          # 20% 아웃라이어
    b[:40] /= np.linalg.norm(b[:40], axis=1, keepdims=True)
    R, inliers, errs = estimate_relative_rotation(b, a)  # rays_l ≈ R @ rays_r
    assert inliers.sum() >= 150
    assert np.allclose(R, R_true, atol=1e-3)


# ---------------------------------------------------------------- render

def test_seam_weights_binary_away_from_seam():
    h, w = 40, 400
    mask = np.full((h, w), 255, np.uint8)
    w_l = seam_weights(mask, mask, -1.0, 1.0, 0.0, feather_px=20)
    assert (w_l[:, :150] == 1.0).all()
    assert (w_l[:, 250:] == 0.0).all()
    mid = w_l[:, 150:250]
    assert ((mid >= 0) & (mid <= 1)).all()


def test_seam_weights_mask_fallback():
    h, w = 40, 400
    mask_l = np.full((h, w), 255, np.uint8)
    mask_r = np.full((h, w), 255, np.uint8)
    mask_l[:10, :] = 0        # L 이 못 덮는 영역은 심 좌측이라도 R 로 채움
    w_l = seam_weights(mask_l, mask_r, -1.0, 1.0, 0.0, feather_px=20)
    assert (w_l[:10, :] == 0.0).all()
    assert (w_l[10:, :150] == 1.0).all()


# ---------------------------------------------------------------- project

def _fake_alignment():
    return Alignment(Rh=rot_xz(0.3, 0.02), yaw_split_deg=68.8,
                     pitch_auto=0.5, roll_auto=-0.1, yaw_auto=-0.07,
                     n_matches=100, n_inliers=80, residual_deg=0.13)


def test_alignment_dict_roundtrip():
    a = _fake_alignment()
    b = alignment_from_dict(alignment_to_dict(a))
    assert np.allclose(a.Rh, b.Rh)
    assert a.pitch_auto == b.pitch_auto
    assert a.el0 == b.el0


def test_project_roundtrip(tmp_path):
    (tmp_path / "GOPR0001.MP4").touch()
    data = {
        "left_files": [str(tmp_path / "GOPR0001.MP4")],
        "right_files": [],
        "offset_sec": 0.068,
        "lens_profile": "test",
        "segments": [{"start_sec": 0.0, "alignment": _fake_alignment()},
                     {"start_sec": 100.0, "alignment": _fake_alignment()}],
        "user": {"pitch": 1.5, "roll": 0.0, "yaw": -0.3, "feather_px": 40},
        "export": {"start": 0, "end": 60},
    }
    p = tmp_path / "proj.json"
    save_project(p, data)
    d = load_project(p)
    assert d["offset_sec"] == 0.068
    assert len(d["segments"]) == 2
    assert d["segments"][1]["start_sec"] == 100.0
    assert np.allclose(d["segments"][0]["alignment"].Rh, data["segments"][0]["alignment"].Rh)


def test_project_relative_path_recovery(tmp_path):
    (tmp_path / "GOPR0001.MP4").touch()
    data = {"left_files": [str(tmp_path / "GOPR0001.MP4")], "right_files": [],
            "segments": [], "offset_sec": 0.0}
    p = tmp_path / "proj.json"
    save_project(p, data)
    # 절대경로가 깨진 상황을 흉내: JSON 의 절대경로를 존재하지 않는 경로로 교체
    txt = p.read_text().replace(str(tmp_path / "GOPR0001.MP4"),
                                "/nonexistent/GOPR0001.MP4")
    p.write_text(txt)
    d = load_project(p)
    assert d["left_files"] == [str(tmp_path / "GOPR0001.MP4")]


# ---------------------------------------------------------------- sync (ffmpeg 필요)

def test_audio_sync_synthetic(tmp_path):
    from pystitch.core.sync import estimate_offset
    base = tmp_path / "L.wav"
    delayed = tmp_path / "R.wav"
    # 같은 핑크 노이즈, R 은 1.5초 늦게 시작 (같은 사건이 R 에서 1.5초 뒤)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "anoisesrc=d=20:c=pink:r=8000:a=0.5:seed=7",
                    str(base)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(base),
                    "-af", "adelay=1500,apad=pad_dur=0", "-t", "20",
                    str(delayed)], check=True)
    offset, conf = estimate_offset(str(base), str(delayed), duration=20)
    assert abs(offset - 1.5) < 0.02, f"offset={offset}"
    assert conf > 4


def test_alignment_rejects_degenerate_frames(lens):
    """무관한 두 이미지(가짜 매칭/저인라이어)는 예외 — 검은 화면 방지 게이트."""
    import cv2

    from pystitch.core.align import estimate_alignment

    rng = np.random.default_rng(11)
    imgs = []
    for _ in range(2):
        im = rng.integers(0, 255, (270, 480, 3), np.uint8)
        imgs.append(cv2.resize(im, (lens.width, lens.height)))
    with pytest.raises(RuntimeError):
        estimate_alignment(imgs[0], imgs[1], lens, log=lambda s: None)


def test_worker_attrs_do_not_shadow_qthread_api():
    """QThread 메서드(start 등)를 속성이 가리면 GUI가 abort 로 죽는다."""
    pytest.importorskip("PyQt6")          # CI 테스트 환경엔 GUI 스택 없음
    from pystitch.gui.workers import ExportWorker, PlaybackWorker

    w = ExportWorker(None, [], [], [], 0.0, 0.0, 60.0, "out.mp4")
    assert callable(w.start) and callable(w.run)
    p = PlaybackWorker(None, None, [], [], 0.0, 0)
    assert callable(p.start) and callable(p.run)


def test_project_resolves_cross_platform_paths(tmp_path):
    """Windows 식 경로(D:\\...)로 저장된 프로젝트를 WSL 에서 열면 /mnt/d 로 복구."""
    from pystitch.core.align import Alignment
    from pystitch.core.project import load_project, save_project

    # WSL 에 존재하는 파일 — 저장소 이름에 독립적으로 저장소 자신을 쓴다
    real = (Path(__file__).resolve().parents[1] / "README.md")
    if not str(real).startswith("/mnt/"):
        pytest.skip("WSL 전용 테스트 (경로 변환 대상이 /mnt/*)")
    drive = real.parts[2]                                  # /mnt/<drive>/...
    win_style = f"{drive.upper()}:\\" + "\\".join(real.parts[3:])
    a = Alignment(Rh=np.eye(3), yaw_split_deg=40.0,
                  pitch_auto=0.0, roll_auto=0.0, yaw_auto=0.0)
    proj = tmp_path / "p.json"
    save_project(proj, {"left_files": [win_style], "right_files": [win_style],
                        "segments": [{"start_sec": 0.0, "alignment": a}]})
    d = load_project(proj)
    assert d["left_files"][0] == str(real)
