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
    bundle_calibrate, decompose_H, make_K, match_frames, rotation_average,
)  # noqa: E501


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


def build_trajectory(cap, grid, det_w, img_size, log):
    W_HINT = img_size
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

    BR = 8                                      # 브리지 최대 뒤로 스텝 수
    all_gray = {}                               # {grid idx: (gray, scale)}
    all_gray[0] = gray(grid[0])
    recent = [all_gray[0]]                       # 최근 그레이 (뒤로 BR 개)
    Hcum = [np.eye(3)]
    n_ok = n_rep = 0
    t0 = time.perf_counter()
    for i in range(1, len(grid)):
        cur = gray(grid[i]); all_gray[i] = cur
        H = _homography(recent[-1], cur)        # grid[i-1] → grid[i]
        if H is not None and Hcum[i - 1] is not None:
            n_ok += 1
            Hcum.append(H @ Hcum[i - 1])
        else:
            # 끊김: 직전 유효 그리드로 직접 브리지 (고정 카메라 = 배경
            # 매칭. 하프타임·급전환도 배경 공유하면 재연결)
            done = False
            for back in range(2, min(BR, len(recent)) + 1):
                j = i - back
                if j < 0 or Hcum[j] is None or recent[-back] is None:
                    continue
                Hb = _homography(recent[-back], cur)
                if Hb is not None:
                    Hcum.append(Hb @ Hcum[j]); n_rep += 1; done = True
                    break
            if not done:
                Hcum.append(None)
        recent.append(cur)
        if len(recent) > BR:
            recent.pop(0)
        if i % 60 == 0:
            log(f"  궤적 {i}/{len(grid)-1} ({(time.perf_counter()-t0):.0f}s)")
    # 루프 클로저: 그레이가 메모리에 있는 지금 검출 (재로딩 회피).
    # f0 로 체인 절대회전 근사 → 시야(yaw) 유사한 먼 쌍만 직접 SIFT.
    valid = [i for i in range(len(grid)) if Hcum[i] is not None]
    loop = []
    if len(valid) >= 3:
        ref = valid[len(valid) // 2]
        Hri = np.linalg.inv(Hcum[ref])
        K0 = make_K(0.9 * W_HINT[0], W_HINT[0], W_HINT[1])
        R0 = {i: decompose_H(Hcum[i] @ Hri, K0)[0] for i in valid}
        yaw = {i: np.arctan2(R0[i][0, 2], R0[i][2, 2]) for i in valid}
        t1 = time.perf_counter()
        # 후보(유사 시야 먼 쌍)를 먼저 다 모으고, 상한(LOOP_MAX)까지
        # 균등 샘플링해 SIFT — 팬 왕복이 많으면 후보가 O(n²)로 폭발하므로
        # (SIFT 무제한 호출 방지). 회전 평균엔 골고루 퍼진 루프면 충분.
        LOOP_MAX = 400
        cand = []
        for a in range(len(valid)):
            i = valid[a]
            for b in range(a + 8, len(valid)):
                j = valid[b]
                if abs(yaw[i] - yaw[j]) > np.deg2rad(5):
                    continue
                ang = (np.trace(R0[i] @ R0[j].T) - 1) / 2
                if ang > np.cos(np.deg2rad(5)):     # <5° 시야차
                    cand.append((b - a, i, j))      # 스팬 클수록 유용
        cand.sort(reverse=True)                     # 긴 루프 우선
        if len(cand) > LOOP_MAX:
            idx = np.linspace(0, len(cand) - 1, LOOP_MAX).astype(int)
            cand = [cand[k] for k in idx]
        for _sp, i, j in cand:
            Hd = _homography(all_gray.get(j), all_gray.get(i))
            if Hd is not None:
                loop.append((i, j, Hd))
        log(f"  루프 클로저 {len(loop)}/{len(cand)}후보 "
            f"({time.perf_counter()-t1:.0f}s)")
    return Hcum, n_ok, n_rep, loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--grid-sec", type=float, default=2.0,
                    help="궤적 샘플 간격(초) — 촘촘할수록 링크 확실")
    ap.add_argument("--det-w", type=int, default=1600)
    ap.add_argument("--record-sec", type=float, default=1.0,
                    help="사이드카에 자세 기록 간격(초)")
    ap.add_argument("--traj-only", action="store_true",
                    help="궤적 캐시(.rotcam_traj.npz)만 생성 (랜드마크 불요)")
    ap.add_argument("--rebuild-traj", action="store_true",
                    help="궤적 캐시 무시하고 재계산")
    args = ap.parse_args()

    vp = Path(args.video)
    sp = vp.with_suffix(".ptz.json")
    size = [105.0, 68.0]
    lms = []
    if sp.exists():
        doc = json.loads(sp.read_text(encoding="utf-8"))
        fpts = doc.get("field_points") or {}
        frms = doc.get("field_point_frames") or {}
        size = doc.get("field_size") or size
        pos = landmark_positions(size[0], size[1])
        lms = [(k, fpts[k], int(frms[k])) for k in fpts
               if k in frms and k not in LINE_LANDMARKS
               and k not in VLINE_LANDMARKS and k in pos]
    if not args.traj_only and len(lms) < 4:
        sys.exit(f"위치 랜드마크 {len(lms)}개 (<4) — PitchWatch 에서 프레임 "
                 "기록과 함께 더 찍어야 함 (또는 --traj-only 로 궤적만)")
    pos = landmark_positions(size[0], size[1])

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        sys.exit(f"열기 실패: {vp}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, int(args.grid_sec * fps))
    # 랜드마크 있으면 그 범위+여유, 없으면(--traj-only) 전체 영상
    hi = (min(max(f for _k, _p, f in lms) + step, total) if lms else total)
    grid = list(range(0, hi, step))
    print(f"[rotcam] {W}x{H} @ {fps:.1f}fps, 랜드마크 {len(lms)}개, "
          f"궤적 그리드 {len(grid)}개 ({args.grid_sec}s)", flush=True)

    tcache = vp.with_suffix(".rotcam_traj.npz")
    Hcum = loop_H = None
    if tcache.exists() and not args.rebuild_traj:
        try:
            z = np.load(tcache, allow_pickle=True)
            if int(z["step"]) == step and int(z["det_w"]) == args.det_w \
                    and int(z["n"]) == len(grid):
                Hcum = [None if np.isnan(a).any() else a for a in z["Hcum"]]
                loop_H = [(int(i), int(j), Hd)
                          for i, j, Hd in z["loop"]]
                print(f"[rotcam] 궤적 캐시 재사용: {tcache.name} "
                      f"(루프 {len(loop_H)}개)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[rotcam] 캐시 무시: {e}", flush=True)
            Hcum = None
    if Hcum is None:
        Hcum, n_ok, n_rep, loop_H = build_trajectory(
            cap, grid, args.det_w, (W, H), print)
        arr = np.array([np.full((3, 3), np.nan) if h is None else h
                        for h in Hcum], np.float64)
        loop_arr = np.array([(i, j, Hd) for i, j, Hd in loop_H],
                            dtype=object)
        np.savez(tcache, Hcum=arr, loop=loop_arr, step=step,
                 det_w=args.det_w, n=len(grid))
        print(f"[rotcam] 궤적: 링크 {n_ok}, 복구 {n_rep}, 루프 "
              f"{len(loop_H)} → 캐시 {tcache.name}", flush=True)
    n_valid = sum(1 for h in Hcum if h is not None)
    print(f"[rotcam] 유효 자세 {n_valid}/{len(grid)}", flush=True)
    if args.traj_only:
        print("[rotcam] --traj-only: 궤적 캐시만 생성하고 종료", flush=True)
        cap.release(); return

    valid = [i for i in range(len(grid)) if Hcum[i] is not None]
    ref_i = valid[len(valid) // 2]              # 기준 = 중앙 유효 그리드
    Hri = np.linalg.inv(Hcum[ref_i])
    Hchain_rel = {i: (Hcum[i] @ Hri) for i in valid}   # ref → i

    def averaged_rots(f):
        K = make_K(f, W, H)
        Rch = [None] * len(grid)
        for i in valid:
            Rch[i] = decompose_H(Hchain_rel[i], K)[0]
        edges = []
        for a, b in zip(valid, valid[1:]):      # 체인 엣지 (b 기준 a)
            edges.append((b, a, decompose_H(Hcum[b] @ np.linalg.inv(Hcum[a]),
                                            K)[0]))
        for i, j, Hd in loop_H:                 # 루프 엣지 (i 기준 j)
            edges.append((i, j, decompose_H(Hd, K)[0]))
        return rotation_average(len(grid), edges, Rch)

    cam_init = np.array([0.0, size[1] / 2 + 12, 7.0])
    best = None
    for frac in (0.5, 0.7, 0.9, 1.1):
        Ravg = averaged_rots(frac * W)
        rr = {}
        for k, _p, fr in lms:
            gi = min(valid, key=lambda i: abs(grid[i] - fr))
            rr[fr] = Ravg[gi]
        landmarks = [(p, pos[k], fr) for k, p, fr in lms]
        cal = bundle_calibrate(rr, landmarks, (W, H), cam_init,
                               f_init_fracs=(frac,))
        if cal is not None and (best is None or cal["rms_px"] < best["rms_px"]):
            best = cal
    if best is None:
        sys.exit("번들 실패 — 랜드마크/궤적 확인")
    print(f"[rotcam] 루프 클로저 {len(loop_H)}개로 회전 평균", flush=True)
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

    # 사이드카: 그리드별 자세 (rvec) + f — 회전평균 절대회전 사용
    K = best["K"]
    Ravg = averaged_rots(best["f"])             # 최적 f 로 평균 회전 재산출
    Href_inv = Hri
    rec_frames, rvecs, fs = [], [], []
    for i, gf in enumerate(grid):
        if Hcum[i] is None or Ravg[i] is None:
            continue
        R_i = Ravg[i] @ best["R0"]              # ref기준 → 절대(월드)
        ratio = decompose_H(Hcum[i] @ Href_inv, K)[1]   # 줌비만
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
