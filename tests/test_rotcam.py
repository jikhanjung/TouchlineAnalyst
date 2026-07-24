"""core/rotcam.py — 회전 카메라 필드 정합 (P06-3) 합성 검증.

GT 카메라: 반대편 사이드라인 (0, 45, 5.5m), f=1400px/1920 폭.
검증: 기준 캘리브레이션(f·위치 복원), 팬+줌 시퀀스 체인 추적의
누적 오차, 지면 투영 정확도.
"""
import numpy as np

from pystitch.core.rotcam import (
    calibrate_reference, decompose_H, field_to_pixel, make_K,
    pixel_to_field, track_step,
)

W, H = 1920, 1080
F_TRUE = 1400.0
CAM = np.array([3.0, 45.0, 5.5])          # 반대편 사이드라인 위


def look_at(pos, target):
    """OpenCV 관례 (x우, y하, z전방) world→cam 회전 (행 = 카메라 축)."""
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, fwd])


def gt_state(yaw_deg=0.0, f=F_TRUE):
    """센터 응시 기준에서 yaw 만큼 팬한 GT (R, t, K)."""
    R0 = look_at(CAM, np.array([0.0, 0.0, 0.0]))
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = R0 @ Rz                              # world Z 축(연직) 팬
    t = -R @ CAM
    return R, t, make_K(f, W, H)


def test_reference_calibration():
    """랜드마크 → f·설치 위치 복원 (f 1%, 위치 0.5m)."""
    rng = np.random.default_rng(2)
    R, t, K = gt_state()
    marks = np.array([[-52.5, 34], [0, 34], [52.5, 34], [-52.5, -34],
                      [0, -34], [52.5, -34], [0, 0], [-41, 20.16],
                      [41, 20.16], [-52.5, 0], [52.5, 0],
                      [0, 9.15], [0, -9.15], [-9.15, 0], [9.15, 0],
                      [-36, -20.16], [36, -20.16], [-11, 0], [11, 0],
                      [-20, -10], [20, -10], [-30, 5], [30, 5]])
    px = field_to_pixel(K, R, t, marks) + rng.normal(0, 1.0, (len(marks), 2))
    vis = np.isfinite(px[:, 0]) & (px[:, 0] > -200) & (px[:, 0] < W + 200)
    assert vis.sum() >= 6
    c = calibrate_reference(px[vis], marks[vis], (W, H))
    assert c is not None and c["rms_px"] < 3.0
    assert abs(c["f"] - F_TRUE) / F_TRUE < 0.01
    assert np.linalg.norm(c["cam_pos"] - CAM) < 0.5


def _project_scene(R, t, K, scene):
    xc = (R @ scene.T).T + t[None]
    uv = (K @ xc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    ok = (xc[:, 2] > 1) & (uv[:, 0] > -50) & (uv[:, 0] < W + 50) \
        & (uv[:, 1] > -50) & (uv[:, 1] < H + 50)
    return uv, ok


def test_decompose_pan_and_zoom():
    """단일 스텝: 순수 팬과 팬+줌의 (R, f비율) 분해 정확도."""
    rng = np.random.default_rng(4)
    scene = np.column_stack([rng.uniform(-55, 55, 400),
                             rng.uniform(-36, 36, 400),
                             rng.uniform(0, 2.0, 400)])
    for dyaw, ratio in ((1.5, 1.0), (0.8, 1.05), (-2.0, 0.95)):
        R1, t1, K1 = gt_state(0.0)
        R2, t2, _ = gt_state(dyaw)
        K2 = make_K(F_TRUE * ratio, W, H)
        uv1, ok1 = _project_scene(R1, t1, K1, scene)
        uv2, ok2 = _project_scene(R2, t2, K2, scene)
        m = ok1 & ok2
        p1 = uv1[m] + rng.normal(0, 0.5, (m.sum(), 2))
        p2 = uv2[m] + rng.normal(0, 0.5, (m.sum(), 2))
        import cv2
        Hm, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        R_rel, r_est, res = decompose_H(Hm, K1)
        ang = np.rad2deg(np.arccos(
            np.clip((np.trace(R_rel @ (R2 @ R1.T).T) - 1) / 2, -1, 1)))
        assert ang < 0.25, f"회전 오차 {ang:.3f}°"   # 0.5px 노이즈 기준
        assert abs(r_est - ratio) < 0.005, f"줌비 {r_est} vs {ratio}"


MARKS = np.array([[-52.5, 34], [0, 34], [52.5, 34], [-52.5, -34],
                  [0, -34], [52.5, -34], [0, 0], [-41, 20.16],
                  [41, 20.16], [-52.5, 0], [52.5, 0],
                  [0, 9.15], [0, -9.15], [-9.15, 0], [9.15, 0],
                  [-36, -20.16], [36, -20.16], [-30, 5], [30, 5]])


def test_chain_tracking_with_anchor():
    """80스텝 팬(±20°)+중간 줌: 체인만으론 드리프트 누적(랜덤워크) →
    20스텝마다 랜드마크 재정렬(anchor_rotation)로 지면 오차 < 1.5m."""
    from pystitch.core.rotcam import anchor_rotation
    rng = np.random.default_rng(9)
    scene = np.column_stack([rng.uniform(-55, 55, 500),
                             rng.uniform(-36, 36, 500),
                             rng.uniform(0, 2.0, 500)])
    yaws = 20 * np.sin(np.linspace(0, 2 * np.pi, 81))
    fs = np.where((np.arange(81) > 30) & (np.arange(81) < 50),
                  F_TRUE * 1.2, F_TRUE)
    R0, t0, K0 = gt_state(yaws[0], fs[0])
    state = {"f": float(fs[0]), "K": make_K(fs[0], W, H), "R": R0}
    for k in range(1, 81):
        Ra, ta, Ka = gt_state(yaws[k - 1], fs[k - 1])
        Rb, tb, Kb = gt_state(yaws[k], fs[k])
        uva, oka = _project_scene(Ra, ta, Ka, scene)
        uvb, okb = _project_scene(Rb, tb, Kb, scene)
        m = oka & okb
        pa = uva[m] + rng.normal(0, 0.5, (m.sum(), 2))
        pb = uvb[m] + rng.normal(0, 0.5, (m.sum(), 2))
        state = track_step(state, pa, pb)
        assert state is not None, f"스텝 {k} 추적 실패"
        if k % 20 == 0:                     # 주기 재정렬 (흰 라인/랜드마크)
            mk_px = field_to_pixel(Kb, Rb, tb, MARKS)
            vis = (np.isfinite(mk_px[:, 0]) & (mk_px[:, 0] > 0)
                   & (mk_px[:, 0] < W) & (mk_px[:, 1] > 0)
                   & (mk_px[:, 1] < H))
            assert vis.sum() >= 3
            a = anchor_rotation(CAM, mk_px[vis]
                                + rng.normal(0, 1.0, (vis.sum(), 2)),
                                MARKS[vis], state["f"], (W, H))
            assert a is not None and a["res_deg"] < 0.2
            state.update(R=a["R"], f=a["f"], K=a["K"])
    # 마지막 프레임에서 알려진 필드 점의 지면 투영 오차
    Rg, tg, Kg = gt_state(yaws[-1], fs[-1])
    probe = np.array([[0.0, 0.0], [-30.0, 10.0], [20.0, -20.0]])
    px = field_to_pixel(Kg, Rg, tg, probe)
    est = pixel_to_field(state["K"], state["R"], CAM, px)
    err = np.hypot(*(est - probe).T)
    assert np.nanmax(err) < 1.5, f"지면 오차 {err}"
    assert abs(state["f"] - fs[-1]) / fs[-1] < 0.02


def test_track_step_rejects_garbage():
    state = {"f": F_TRUE, "K": make_K(F_TRUE, W, H), "R": np.eye(3)}
    rng = np.random.default_rng(1)
    assert track_step(state, rng.uniform(0, W, (10, 2)),
                      rng.uniform(0, W, (10, 2))) is None   # 매칭 부족
    p = rng.uniform(0, W, (60, 2))
    q = rng.uniform(0, W, (60, 2))                          # 무상관 → 기각
    assert track_step(state, p, q) is None


def test_auto_anchor_recovers_perturbed_pose():
    """필드 마킹 자동 앵커 (P06-3 자동화): 합성 흰 라인 렌더 →
    자세를 흔든 뒤 auto_anchor 가 GT 로 되돌리는지."""
    import cv2

    from pystitch.core.rotcam import auto_anchor, template_polylines

    R_gt, t_gt, K_gt = gt_state(yaw_deg=6.0)
    # GT 자세로 템플릿을 투영해 흰 라인 이미지를 렌더 (초록 바탕)
    img = np.full((H, W, 3), (40, 90, 40), np.uint8)
    fld, _tans, _fams = template_polylines(step=0.25)
    proj = field_to_pixel(K_gt, R_gt, t_gt, fld)
    for u, v in proj[np.isfinite(proj).all(1)]:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (int(u), int(v)), 2, (235, 235, 235), -1)
    # 섭동: 팬 0.35° + 틸트 0.2° + f +1% — 주기 앵커의 실제 임무
    # 규모 (앵커 간 드리프트, 043 실측 랜덤워크 스케일)
    c, s = np.cos(np.deg2rad(0.35)), np.sin(np.deg2rad(0.35))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    c2, s2 = np.cos(np.deg2rad(0.2)), np.sin(np.deg2rad(0.2))
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]])
    f_bad = F_TRUE * 1.01
    state = {"R": Rx @ (R_gt @ Rz), "f": f_bad, "K": make_K(f_bad, W, H)}
    got = auto_anchor(img, state, CAM)
    assert got is not None, "auto_anchor 가 대응을 못 찾음"
    assert got["n_pts"] >= 100
    # 회전 오차 (deg)
    dR = got["R"] @ R_gt.T
    ang = np.rad2deg(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    assert ang < 0.1, f"회전 오차 {ang:.3f}°"
    assert abs(got["f"] - F_TRUE) / F_TRUE < 0.01, got["f"]
    # 지면 투영 정확도: 원측 페널티 마크 부근 점
    p = field_to_pixel(got["K"], got["R"], -got["R"] @ CAM,
                       [(41.5, 0.0)])
    p_gt = field_to_pixel(K_gt, R_gt, t_gt, [(41.5, 0.0)])
    assert np.linalg.norm(p - p_gt) < 4.0, np.linalg.norm(p - p_gt)


def test_auto_anchor_fails_without_lines():
    """라인 없는 화면(관중석 팬 등) → None (품질 플래그 경로)."""
    from pystitch.core.rotcam import auto_anchor

    R_gt, _t, _K = gt_state()
    img = np.full((H, W, 3), (40, 90, 40), np.uint8)
    state = {"R": R_gt, "f": F_TRUE, "K": make_K(F_TRUE, W, H)}
    assert auto_anchor(img, state, CAM) is None


def test_chain_homography_static_and_transfer(tmp_path):
    """정지 합성 영상 → 체인 H ≈ 항등, transfer_points 왕복."""
    import cv2

    from pystitch.core.rotcam import chain_homography, transfer_points

    rng = np.random.default_rng(3)
    base = (rng.random((240, 420, 3)) * 255).astype(np.uint8)
    vp = tmp_path / "static.mp4"
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"),
                         30.0, (420, 240))
    for _ in range(40):
        vw.write(base)
    vw.release()
    H = chain_homography(vp, 0, 35, det_w=420, every_s=0.5)
    assert H is not None
    pts = np.array([[100.0, 60.0], [300.0, 200.0]])
    moved = transfer_points(H, pts)
    assert np.abs(moved - pts).max() < 2.0, moved


def test_bundle_calibrate_pools_frames():
    """다프레임 랜드마크를 상대회전으로 모아 f·설치 복원 (팬 퇴화 회피)."""
    from pystitch.core.rotcam import bundle_calibrate
    from pystitch.core.field import landmark_positions
    pos = landmark_positions(105.0, 68.0)
    keys = ["corner_far_l", "corner_far_r", "half_far", "half_near",
            "circle_l", "circle_r", "pen_l_far", "pen_r_far"]
    # 여러 팬 각도(프레임)에서 랜드마크 관측 — 상대회전은 GT 로 제공
    R0, _t0, _K = gt_state(yaw_deg=0.0)
    rel_rots, landmarks = {}, []
    for fr, yaw in enumerate((-10, -4, 0, 6, 12)):
        Rf, tf, Kf = gt_state(yaw_deg=yaw)
        rel = Rf @ R0.T                          # 기준(yaw0)→이 프레임
        rel_rots[fr] = rel
        for k in keys:
            uv = field_to_pixel(Kf, Rf, tf, [pos[k]])[0]
            if np.all(np.isfinite(uv)) and 0 <= uv[0] < W and 0 <= uv[1] < H:
                landmarks.append((uv, pos[k], fr))
    cal = bundle_calibrate(rel_rots, landmarks, (W, H),
                           np.array([0.0, 45.0, 6.0]))
    assert cal is not None
    assert abs(cal["f"] - F_TRUE) / F_TRUE < 0.03, cal["f"]
    assert np.linalg.norm(cal["cam_pos"] - CAM) < 2.0, cal["cam_pos"]
    assert cal["rms_px"] < 2.0
