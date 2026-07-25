"""회전 카메라 캘리브레이션 — 찍힌 랜드마크 기반 (팬 자가보정 대체).

팬만 있는 영상은 호모그래피 자가보정으로 초점거리 f 를 못 푼다(Sturm
임계 운동 = 팬 퇴화 → f 발산). 대신 경기장은 **평면**이므로, 알려진
필드 좌표를 가진 랜드마크를 한 기준 프레임으로 모아 평면 PnP(f 탐색)
로 f·설치위치를 직접 푼다 (core.rotcam.calibrate_reference). 이것이
GUI 의 _refit_field_rotcam 과 동일한 경로.

전송은 이미 만들어진 궤적 캐시(<video>.rotcam_traj.npz)의 Hcum 을 쓴다
(재디코드·SIFT 없음, 수 초). 랜드마크가 찍힌 프레임 범위까지만 사용.

  python scripts/rotcam_from_landmarks.py <video>.MP4 [--write]

--write 없으면 f/FOV/rms 만 보고(검증), 있으면 <video>.rotcam.json 갱신.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402
from pystitch.core.field import (  # noqa: E402
    LINE_LANDMARKS, VLINE_LANDMARKS, landmark_positions,
)
from pystitch.core.rotcam import (  # noqa: E402
    calibrate_reference, calibrate_reference_lines, decompose_H, make_K,
    rotation_average, transfer_points,
)

# 라인키 → _marking_lines 패밀리 인덱스 (gui.ptz_tab.LINE_FAMILIES 와 동일)
LINE_FAM = {"touch_near": 0, "touch_far": 1, "goal_l": 2, "goal_r": 3,
            "halfway": 4}


def _H(Hcum, gi, ri):
    """그리드 gi → ri 전송 호모그래피 (Hcum[k]: 프레임0→프레임k)."""
    return Hcum[ri] @ np.linalg.inv(Hcum[gi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--write", action="store_true",
                    help="검증만 아니라 <video>.rotcam.json 을 갱신")
    args = ap.parse_args()

    vp = Path(args.video)
    sp = vp.with_suffix(".ptz.json")
    tp = vp.with_suffix(".rotcam_traj.npz")
    if not sp.exists():
        sys.exit(f"랜드마크 사이드카 없음: {sp}")
    if not tp.exists():
        sys.exit(f"궤적 캐시 없음: {tp} — 먼저 rotcam_calibrate.py --traj-only")

    doc = json.loads(sp.read_text(encoding="utf-8"))
    fpts = doc.get("field_points") or {}
    frms = doc.get("field_point_frames") or {}
    invalid = doc.get("field_point_invalid") or {}
    size = doc.get("field_size") or [105.0, 68.0]
    pos = landmark_positions(size[0], size[1])
    lms = [(k, fpts[k], int(frms[k])) for k in fpts
           if k in frms and k not in LINE_LANDMARKS
           and k not in VLINE_LANDMARKS and k in pos]
    if len(lms) < 4:
        sys.exit(f"위치 랜드마크 {len(lms)}개 (<4) — 더 찍어야 함")

    z = np.load(tp, allow_pickle=True)
    step, det_w = int(z["step"]), int(z["det_w"])
    Hcum = [None if np.isnan(a).any() else a for a in z["Hcum"]]
    loop_H = [(int(i), int(j), Hd) for i, j, Hd in z["loop"]]
    ncov = len(Hcum)
    valid = [i for i in range(ncov) if Hcum[i] is not None]

    cap = cv2.VideoCapture(str(vp))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    def grid_of(fr):
        """프레임 fr 에 가장 가까운 유효 그리드 인덱스."""
        gi = int(round(fr / step))
        if gi in valid:
            return gi
        return min(valid, key=lambda i: abs(i - gi)) if valid else None

    # 기준 프레임 = 랜드마크가 가장 많이 찍힌 프레임 (전송 최소)
    ref_frame = Counter(fr for _k, _p, fr in lms).most_common(1)[0][0]
    ri = grid_of(ref_frame)
    if ri is None:
        sys.exit("유효 그리드 없음 — 궤적 캐시 확인")
    hi_frame = max(fr for _k, _p, fr in lms)
    print(f"[lm] {W}x{H} @ {fps:.2f}fps, 랜드마크 {len(lms)}개, "
          f"범위 0..{hi_frame} ({hi_frame/fps/60:.1f}분), "
          f"기준 프레임 {ref_frame} (그리드 {ri}), 궤적 커버 {ncov}그리드",
          flush=True)

    def invalid_here(k, fr):
        return any(a <= fr <= b for a, b in invalid.get(k, ()))

    px, fld, used, dropped = [], [], [], []
    for k, p, fr in lms:
        gi = grid_of(fr)
        if gi is None or Hcum[gi] is None or fr > (ncov - 1) * step:
            dropped.append((k, "궤적 밖"))
            continue
        if invalid_here(k, ref_frame):
            dropped.append((k, "기준 프레임 무효"))
            continue
        q = p if gi == ri else transfer_points(_H(Hcum, gi, ri), [p])[0]
        px.append([float(q[0]), float(q[1])])
        fld.append([float(pos[k][0]), float(pos[k][1])])
        used.append(k)
    print(f"[lm] 기준으로 전송된 랜드마크 {len(px)}개: {', '.join(used)}",
          flush=True)
    for k, why in dropped:
        print(f"[lm]   제외 {k}: {why}", flush=True)
    if len(px) < 4:
        sys.exit(f"기준 프레임 유효 랜드마크 {len(px)}개 (<4)")

    # 로버스트: 재투영 오차가 튀는(엉뚱하게 이송된) 랜드마크를 하나씩
    # 제거하며 재피팅 — 최소 5개는 남긴다.
    def fit_resid(px_, fld_):
        c = calibrate_reference(px_, fld_, (W, H))
        if c is None:
            return None, None
        rv, _ = cv2.Rodrigues(c["R"])
        obj = np.array([[x, y, 0.0] for x, y in fld_])
        proj, _ = cv2.projectPoints(obj, rv, c["t"].reshape(3, 1),
                                    c["K"], None)
        e = np.linalg.norm(proj.reshape(-1, 2) - np.array(px_), axis=1)
        return c, e

    # 라인 점(터치라인 등) → 기준 프레임 이송 + 패밀리
    field_lines = doc.get("field_lines") or {}
    line_px, line_fam = [], []
    for key, pts in field_lines.items():
        if key not in LINE_FAM:
            continue
        for lx, ly, lfr in pts:
            gi = grid_of(int(lfr))
            if gi is None or Hcum[gi] is None:
                continue
            q = ([lx, ly] if gi == ri
                 else transfer_points(_H(Hcum, gi, ri), [[lx, ly]])[0])
            line_px.append([float(q[0]), float(q[1])])
            line_fam.append(LINE_FAM[key])
    if line_px:
        print(f"[lm] 라인 점 {len(line_px)}개 (터치라인 등) point-to-line "
              "구속 사용", flush=True)

    cal, err = fit_resid(px, fld)
    if cal is None:
        sys.exit("calibrate_reference 실패")
    while len(px) > 5:
        i = int(np.argmax(err))
        if err[i] < 30 or err[i] < 2.5 * np.median(err):
            break
        print(f"[lm]   로버스트 제외: {used[i]} (재투영 {err[i]:.0f}px)",
              flush=True)
        px.pop(i); fld.pop(i); used.pop(i)
        cal2, err2 = fit_resid(px, fld)
        if cal2 is None:
            break
        cal, err = cal2, err2
    # 최종: 살아남은 점 + 라인 점으로 point-to-line 정밀화 (라인 있으면)
    if line_px:
        cal_pt = cal
        cal = calibrate_reference_lines(px, fld, line_px, line_fam, (W, H),
                                        length=size[0], width=size[1])
        if cal is None:
            cal = cal_pt
        else:
            print(f"[lm] 라인 정밀화: f {cal_pt['f']:.0f}→{cal['f']:.0f}px, "
                  f"점 rms {cal_pt['rms_px']:.1f}→{cal['rms_px']:.1f}px, "
                  f"라인 잔차 {cal.get('res_line')}", flush=True)
    f = cal["f"]
    cp = cal["cam_pos"]
    fov = 2 * np.degrees(np.arctan(W / 2 / f))
    print(f"[lm] === 결과 ({len(px)}개): f {f:.0f}px (f/W {f/W:.2f}, "
          f"FOV {fov:.1f}°), 설치 ({cp[0]:+.1f},{cp[1]:+.1f},{cp[2]:.1f})m, "
          f"기준 재투영 rms {cal['rms_px']:.1f}px ===", flush=True)
    # 임계값은 줌·후방 설치 카메라(AX700)까지 허용 — 진짜 발산만 차단
    warn = []
    if not (0.3 <= f / W <= 3.0):
        warn.append(f"f/W {f/W:.2f} 비정상(발산 의심)")
    if cal["rms_px"] > 60:
        warn.append(f"rms {cal['rms_px']:.0f}px 큼 (랜드마크 정밀도)")
    info = []
    if abs(cp[1]) > size[1]:
        info.append(f"카메라 폭방향 {cp[1]:+.0f}m (경기장 밖 = 후방 설치)")
    print("[lm] " + ("경고: " + " / ".join(warn) if warn
                     else "정상 범위 — 랜드마크 기반 f 성공"), flush=True)
    if info:
        print("[lm] 참고: " + " / ".join(info), flush=True)

    if not args.write:
        print("[lm] (검증만 — --write 로 rotcam.json 갱신)", flush=True)
        return
    if warn:
        print("[lm] 경고가 있어 저장하지 않음 (랜드마크 보강 후 재시도)",
              flush=True)
        return

    # 프레임별 자세: 기준 R 을 궤적 상대회전으로 전파 (회전평균)
    Kf = make_K(f, W, H)
    Rch = [None] * ncov
    Hri_inv = np.linalg.inv(Hcum[ri])
    for i in valid:
        Rch[i] = decompose_H(Hcum[i] @ Hri_inv, Kf)[0]   # ref → i
    edges = []
    for a, b in zip(valid, valid[1:]):
        edges.append((b, a, decompose_H(Hcum[b] @ np.linalg.inv(Hcum[a]),
                                        Kf)[0]))
    for i, j, Hd in loop_H:
        edges.append((i, j, decompose_H(Hd, Kf)[0]))
    Ravg = rotation_average(ncov, edges, Rch)
    R_ref = cal["R"]
    frames, rvecs, fs = [], [], []
    for i in range(ncov):
        if Ravg[i] is None:
            continue
        R_i = Ravg[i] @ R_ref
        ratio = decompose_H(Hcum[i] @ Hri_inv, Kf)[1]
        rv, _ = cv2.Rodrigues(R_i)
        frames.append(int(i * step))
        rvecs.append([round(float(v), 6) for v in rv.ravel()])
        fs.append(round(float(f * ratio), 2))
    out = {"version": 1, "video": vp.name, "img_size": [W, H],
           "fps": round(fps, 4), "field_size": size,
           "cam_pos": [round(float(v), 3) for v in cp],
           "f": round(float(f), 2), "rms_px": round(cal["rms_px"], 2),
           "grid_sec": round(step / fps, 3), "source": "landmarks",
           "frames": frames, "rvec": rvecs, "f_frame": fs}
    op = vp.with_suffix(".rotcam.json")
    op.write_text(json.dumps(out), encoding="utf-8")
    print(f"[lm] 저장: {op.name} — 자세 {len(frames)}개 (source=landmarks)",
          flush=True)


if __name__ == "__main__":
    main()
