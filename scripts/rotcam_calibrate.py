"""회전 카메라 전역 캘리브레이션 (P06-3, devlog 060 아키텍처).

조밀 호모그래피 궤적(끊긴 링크 복구) + 다프레임 랜드마크 번들로
카메라 설치 위치·초점거리·프레임별 방향/줌을 풀어 <video>.rotcam.json
사이드카에 기록한다. 이후 GUI/융합은 사이드카만 읽어 임의 프레임의
자세와 A→B 회전을 즉시 얻는다.

전제: <video>.ptz.json 에 랜드마크(field_points + field_point_frames)가
찍혀 있어야 한다 (PitchWatch 에서, 여러 프레임에 나눠 찍어도 됨).

사용법:
  python scripts/rotcam_calibrate.py C0011.MP4 [--grid-sec 2] [--det-w 1600]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.field import (  # noqa: E402
    LINE_LANDMARKS, VLINE_LANDMARKS, landmark_positions,
)
from pystitch.core.rotcam import (  # noqa: E402
    bundle_calibrate, decompose_H, make_K, match_frames,
)


def _homography(ga, gb, min_inl=15):
    """그레이 (img, scale) 두 개 → 원본 좌표 호모그래피 (실패 None)."""
    if ga is None or gb is None:
        return None
    pa, pb = match_frames(ga[0], gb[0])
    if len(pa) < 20:
        return None
    Hs, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if Hs is None or mask is None or mask.sum() < min_inl:
        return None
    S = np.diag([ga[1], ga[1], 1.0])
    T = np.diag([gb[1], gb[1], 1.0])
    return np.linalg.inv(T) @ Hs @ S


def build_trajectory(cap, grid, det_w, log):
    """그리드 프레임 인접 호모그래피 → grid[0] 기준 누적 (Hcum).

    끊긴 링크는 이웃 건너뛰기(gap+1)로 복구 — 한 프레임 실패가 이후
    전체를 무효화하던 문제(devlog 060) 방지. 반환: (Hcum, n_ok, n_repair).
    """
    def gray(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            return None
        sc = det_w / fr.shape[1] if fr.shape[1] > det_w else 1.0
        if sc < 1.0:
            fr = cv2.resize(fr, (det_w, int(fr.shape[0] * sc)),
                            interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), sc

    grays = [None, None]                       # 직전 2개만 유지 (건너뛰기용)
    grays[1] = gray(grid[0])
    Hcum = [np.eye(3)]
    n_ok = n_rep = 0
    t0 = time.perf_counter()
    for i in range(1, len(grid)):
        cur = gray(grid[i])
        H = _homography(grays[1], cur)         # grid[i-1] → grid[i]
        if H is not None:
            n_ok += 1
        elif grays[0] is not None:             # 복구: grid[i-2] → grid[i]
            H2 = _homography(grays[0], cur)
            if H2 is not None and Hcum[i - 2] is not None:
                Hcum.append(H2 @ Hcum[i - 2])  # i-2 기준으로 누적
                grays = [grays[1], cur]
                n_rep += 1
                if len(Hcum) % 60 == 0:
                    log(f"  궤적 {i}/{len(grid)-1} "
                        f"({(time.perf_counter()-t0):.0f}s)")
                continue
        Hcum.append(None if (Hcum[-1] is None or H is None) else H @ Hcum[-1])
        grays = [grays[1], cur]
        if i % 60 == 0:
            log(f"  궤적 {i}/{len(grid)-1} ({(time.perf_counter()-t0):.0f}s)")
    return Hcum, n_ok, n_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--grid-sec", type=float, default=2.0,
                    help="궤적 샘플 간격(초) — 촘촘할수록 링크 확실")
    ap.add_argument("--det-w", type=int, default=1600)
    ap.add_argument("--record-sec", type=float, default=1.0,
                    help="사이드카에 자세 기록 간격(초)")
    args = ap.parse_args()

    vp = Path(args.video)
    sp = vp.with_suffix(".ptz.json")
    if not sp.exists():
        sys.exit(f"{sp.name} 없음 — PitchWatch 에서 랜드마크 먼저")
    doc = json.loads(sp.read_text(encoding="utf-8"))
    fpts = doc.get("field_points") or {}
    frms = doc.get("field_point_frames") or {}
    size = doc.get("field_size") or [105.0, 68.0]
    pos = landmark_positions(size[0], size[1])
    lms = [(k, fpts[k], int(frms[k])) for k in fpts
           if k in frms and k not in LINE_LANDMARKS
           and k not in VLINE_LANDMARKS and k in pos]
    if len(lms) < 4:
        sys.exit(f"위치 랜드마크 {len(lms)}개 (<4) — 프레임 기록과 함께 "
                 "더 찍어야 함")

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        sys.exit(f"열기 실패: {vp}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lm_max = max(f for _k, _p, f in lms)
    step = max(1, int(args.grid_sec * fps))
    # 궤적은 랜드마크 범위 + 여유를 커버 (사이드카는 전체)
    grid = list(range(0, min(lm_max + step, total), step))
    print(f"[rotcam] {W}x{H} @ {fps:.1f}fps, 랜드마크 {len(lms)}개, "
          f"궤적 그리드 {len(grid)}개 ({args.grid_sec}s)", flush=True)

    Hcum, n_ok, n_rep = build_trajectory(cap, grid, args.det_w, print)
    n_valid = sum(1 for h in Hcum if h is not None)
    print(f"[rotcam] 궤적: 링크 성공 {n_ok}, 복구 {n_rep}, "
          f"유효 자세 {n_valid}/{len(grid)}", flush=True)

    ref_i = len(grid) // 2                      # 기준 = 중앙 그리드
    while ref_i < len(Hcum) and Hcum[ref_i] is None:
        ref_i += 1
    if ref_i >= len(Hcum):
        sys.exit("유효한 기준 프레임 없음 — 궤적 전부 끊김")
    Href_inv = np.linalg.inv(Hcum[ref_i])

    def rel_rots(f):
        """기준 그리드 대비 각 랜드마크 프레임 상대회전 {frame: R}."""
        K = make_K(f, W, H)
        out = {}
        for k, _p, fr in lms:
            gi = min(range(len(grid)), key=lambda i: abs(grid[i] - fr))
            if Hcum[gi] is None:
                out[fr] = None
                continue
            Hrel = Hcum[gi] @ Href_inv          # ref → gi
            out[fr] = decompose_H(Hrel, K)[0]
        return out

    # f 초기값마다 상대회전 재계산하며 번들 (팬 퇴화 회피)
    cam_init = np.array([0.0, size[1] / 2 + 12, 7.0])
    best = None
    for frac in (0.5, 0.7, 0.9, 1.1):
        rr = rel_rots(frac * W)
        landmarks = [(p, pos[k], fr) for k, p, fr in lms]
        cal = bundle_calibrate(rr, landmarks, (W, H), cam_init,
                               f_init_fracs=(frac,))
        if cal is not None and (best is None or cal["rms_px"] < best["rms_px"]):
            best = cal
    if best is None:
        sys.exit("번들 실패 — 랜드마크/궤적 확인")
    cp = best["cam_pos"]
    fov = 2 * np.degrees(np.arctan(W / 2 / best["f"]))
    print(f"[rotcam] 번들: f {best['f']:.0f}px (f/W {best['f']/W:.2f}, "
          f"FOV {fov:.0f}°), 설치 ({cp[0]:+.1f},{cp[1]:+.1f},{cp[2]:.1f})m, "
          f"재투영 rms {best['rms_px']:.1f}px, 랜드마크 {best['n']}개",
          flush=True)
    warn = []
    if not (0.4 <= best["f"] / W <= 1.6):
        warn.append(f"f/W {best['f']/W:.2f} 비정상(팬 퇴화 의심)")
    if abs(cp[1]) > size[1]:
        warn.append(f"설치 폭방향 {cp[1]:+.0f}m 경기장 밖")
    if warn:
        print("[rotcam] 경고: " + " / ".join(warn), flush=True)

    # 사이드카: record-sec 간격 프레임별 자세 (rvec) + f
    K = best["K"]
    rstep = max(1, int(args.record_sec * fps))
    rec_frames, rvecs, fs = [], [], []
    for i, gf in enumerate(grid):
        if Hcum[i] is None or gf % rstep >= step:
            pass
    # 그리드 자세를 record 간격으로 (그리드가 곧 자세 격자)
    for i, gf in enumerate(grid):
        if Hcum[i] is None:
            continue
        Hrel = Hcum[i] @ Href_inv
        R_rel = decompose_H(Hrel, K)[0]
        ratio = decompose_H(Hrel, K)[1]
        R_i = R_rel @ best["R0"]
        rv, _ = cv2.Rodrigues(R_i)
        rec_frames.append(int(gf))
        rvecs.append([round(float(v), 6) for v in rv.ravel()])
        fs.append(round(float(best["f"] * ratio), 2))
    out = {"version": 1, "video": vp.name,
           "img_size": [W, H], "fps": round(fps, 4),
           "field_size": size,
           "cam_pos": [round(float(v), 3) for v in cp],
           "f": round(float(best["f"]), 2),
           "rms_px": round(best["rms_px"], 2),
           "grid_sec": args.grid_sec,
           "frames": rec_frames, "rvec": rvecs, "f_frame": fs}
    op = vp.with_suffix(".rotcam.json")
    op.write_text(json.dumps(out), encoding="utf-8")
    print(f"[rotcam] 저장: {op.name} — 자세 {len(rec_frames)}개", flush=True)
    cap.release()


if __name__ == "__main__":
    main()
