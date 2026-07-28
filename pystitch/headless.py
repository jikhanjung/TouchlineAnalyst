"""헤드리스 파이프라인: 좌/우 디렉터리 → 짝 맞춤 → 오디오 동기화 →
자동 정합 → 파노라마 인코딩 → 선수/공 분석 → 등번호 OCR (devlog 041).

사용법:
  python main.py --headless <left_dir> <right_dir> [--out DIR] [옵션]

출력은 두 디렉터리 이름의 공통부분으로 만든 새 디렉터리(예:
20260712_GoPro5_L/_R → <부모>/20260712_GoPro5)에 프로젝트 파일·파노라마
영상·분석·이벤트 파일을 모아 넣는다 (--out 으로 재지정 가능).
좌/우 짝은 챕터 체인 총 파일 크기 유사도로 맞춘다 (core/pairing.py).
긴 영상은 check_every(기본 5분) 간격으로 겹침 매칭 잔차를 확인해 정합이
틀어진 지점을 이진 탐색(자이로 GPMF 스파이크가 있으면 힌트)으로 찾고,
그 지점부터 새 세그먼트로 재정합한다.

단계별 산출물(<out>/pano_XXXX.mp4, .analysis.json, .events.json 의
"ocr_numbers")이 이미 있으면 건너뛴다 — 중단 후 재실행 안전, --force 로
다시 실행. 등번호 OCR 은 필드 캘리브레이션 없이 박스 높이 게이트만으로
동작하며 결과는 제안으로만 저장된다 (적용은 GUI 번호 메뉴).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .core.align import estimate_alignment, match_overlap
from .core.chapters import ChapteredVideo
from .core.events import load_events_doc, save_events
from .core.export import export_pano
from .core.geometry import pixel_to_ray
from .core.lens import LensProfile, builtin_profiles
from .core.ocr import collect_ocr_candidates, run_jersey_ocr
from .core.pairing import chain_size, pair_directories
from .core.project import save_project
from .core.ptz import analyze_video, classify_teams
from .core.sync import estimate_offset

# 정합 프레임 후보 (구간 내 비율) — 앞쪽 우선: 드리프트 스캔이 여기서
# 시작해 앞으로만 진행하므로, 초반 정합이 서야 전체 구간이 커버된다.
# auto-level 은 프레임 민감도가 커서 후보를 촘촘히 두고 최선 잔차를 고른다.
_ALIGN_FRACS = (0.05, 0.12, 0.2, 0.28, 0.36, 0.5, 0.65, 0.8)
_LEVEL_GOOD_DEG = 0.4     # 이 잔차면 즉시 채택
_LEVEL_MAX_DEG = 2.0      # 전 후보 중 최선이 이보다 나쁘면 실패

# 설치(settle) 탐지: 이 시간 안의 자이로 방향 변경 이벤트는 녹화 시작 후
# 각도 조절·삼각대 세우기로 본다 — 마지막 이벤트 + 여유 이후에서 정합.
_SETTLE_WINDOW = 900.0
_SETTLE_MARGIN = 30.0

# --auto-el 탐색 상한: 원통 투영 세로는 tan(el)·f 라 극단 고도는 픽셀
# 폭발 + 심한 스트레칭뿐 — 이 밖은 어차피 쓸모없는 화질이다.
_AUTO_EL_CAP = (-60.0, 25.0)


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pair_name(left_chain) -> str:
    """출력 이름: 좌측 첫 챕터의 영상 번호 (GOPR0395 → pano_0395)."""
    stem = Path(left_chain[0]).stem
    return f"pano_{stem[-4:]}"


def _default_out_dir(left_dir: Path, right_dir: Path) -> Path:
    """좌/우 디렉터리 이름의 공통 부분을 출력 디렉터리 이름으로 쓴다.

    예: 20260712_GoPro5_L / 20260712_GoPro5_R → <부모>/20260712_GoPro5.
    구분자(_-. ) 토큰 단위로 앞뒤 공통 토큰을 취한다 (Left_cam/Right_cam
    → cam). 의미 있는 공통 부분이 없으면 PyStitch360_headless.
    """
    ta = re.split(r"[_\-. ]+", left_dir.name)
    tb = re.split(r"[_\-. ]+", right_dir.name)
    pre = os.path.commonprefix([ta, tb])
    rest_a, rest_b = ta[len(pre):], tb[len(pre):]
    suf = os.path.commonprefix([rest_a[::-1], rest_b[::-1]])[::-1]
    name = "_".join(pre + suf)
    if len(name) < 3:
        name = "PyStitch360_headless"
    out = left_dir.resolve().parent / name
    if out in (left_dir.resolve(), right_dir.resolve()):
        out = out.with_name(name + "_pano")   # 입력과 동명 — 원본 보호
    return out


def _estimate_sync(left0, right0, first_dur: float, prefer_start=60.0):
    """첫 챕터 쌍에서 오디오 오프셋 추정 — 신뢰도 낮으면 다른 구간 재시도."""
    dur = min(120.0, max(30.0, first_dur - 5.0))
    starts = [t for t in (prefer_start, 60.0, first_dur * 0.5, 0.0)
              if t + dur <= first_dur + 1e-6] or [0.0]
    starts = list(dict.fromkeys(starts))
    best = (0.0, 0.0)
    for start in starts:
        offset, conf = estimate_offset(str(left0), str(right0), start, dur)
        _log(f"[sync] t={start:.0f}s: 오프셋 {offset:+.3f}s, 신뢰도 {conf:.1f}")
        if conf > best[1]:
            best = (offset, conf)
        if conf >= 4:
            break
    if best[1] < 4:
        _log(f"[sync] 경고: 신뢰도 {best[1]:.1f} < 4 — 결과 확인 필요")
    return best


def _read_pair(vid_l, vid_r, offset, t):
    ok_l, img_l = vid_l.read_at(int(round(t * vid_l.fps)))
    ok_r, img_r = vid_r.read_at(int(round((t + offset) * vid_r.fps)))
    return (img_l, img_r) if ok_l and ok_r else None


def _estimate_align(vid_l, vid_r, offset, lens, t0, t1):
    """구간 내 후보 시각들에서 정합 시도.

    auto-level 잔차(level_resid_deg)가 _LEVEL_GOOD_DEG 미만이면 즉시
    채택, 아니면 전 후보 중 최선을 쓴다. 앞 프레임의 최선 해를 다음
    auto-level 의 시작점으로 시드해 수렴을 돕는다.
    """
    best = None                # (resid_deg, alignment, t)
    last_err = None
    for frac in _ALIGN_FRACS:
        t = t0 + frac * (t1 - t0)
        imgs = _read_pair(vid_l, vid_r, offset, t)
        if imgs is None:
            continue
        seed = (best[1].pitch_auto, best[1].roll_auto) if best else None
        extras = []
        for dt in (15.0, 30.0):        # auto-level 점 풀링용 추가 프레임
            p = _read_pair(vid_l, vid_r, offset, min(t + dt, t1 - 1.0))
            if p is not None:
                extras.append(p)
        try:
            a = estimate_alignment(imgs[0], imgs[1], lens, log=_log,
                                   require_level=True, level_init=seed,
                                   level_frames=extras)
        except RuntimeError as e:
            last_err = e
            _log(f"[align] t={t:.0f}s 실패: {e}")
            continue
        if best is None or a.level_resid_deg < best[0]:
            best = (a.level_resid_deg, a, t)
        if a.level_resid_deg < _LEVEL_GOOD_DEG:
            return a, t
    if best is not None and best[0] < _LEVEL_MAX_DEG:
        _log(f"[align] 완전 수렴 실패 — 최선 해 채택 "
             f"(잔차 {best[0]:.2f}°, t={best[2]:.0f}s)")
        return best[1], best[2]
    raise RuntimeError(
        f"모든 후보 프레임에서 정합 실패 (최선 잔차 "
        f"{best[0]:.2f}°): {last_err}" if best else
        f"모든 후보 프레임에서 정합 실패: {last_err}")


# ------------------------------------------------------------ 드리프트 스캔


def _drift_deg(vid_l, vid_r, offset, lens, alignment, t):
    """시각 t 에서 현재 정합의 겹침 매칭 잔차 중앙값 (도). 측정 불가 None.

    잔차 = 매칭 광선쌍에 현재 상대 회전(Rh²)을 적용했을 때의 각오차.
    중앙값이라 겹침을 지나는 선수 등 이동체 아웃라이어에 강건하다.
    """
    imgs = _read_pair(vid_l, vid_r, offset, t)
    if imgs is None:
        return None
    pts_l, pts_r = match_overlap(imgs[0], imgs[1])
    if len(pts_l) < 20:
        return None
    rays_l = pixel_to_ray(pts_l, lens)
    rays_r = pixel_to_ray(pts_r, lens)
    R_lr = alignment.Rh @ alignment.Rh
    err = np.arccos(np.clip(np.sum(rays_l * (rays_r @ R_lr.T), axis=1), -1, 1))
    return float(np.rad2deg(np.median(err)))


def _locate_drift(vid_l, vid_r, offset, lens, a, lo, hi, thr, bumps,
                  min_gap=10.0):
    """(lo=정상, hi=틀어짐) 사이에서 틀어진 시각을 찾는다.

    자이로 스파이크(방향 변경) 힌트가 구간 안에 있으면 그 직전/직후를
    확인해 바로 확정하고, 없으면 이진 탐색으로 min_gap 까지 좁힌다.
    """
    for ev in bumps:
        if not (lo < ev.time_sec < hi):
            continue
        d_after = _drift_deg(vid_l, vid_r, offset, lens, a,
                             min(ev.time_sec + 2.0, hi))
        d_before = _drift_deg(vid_l, vid_r, offset, lens, a,
                              max(ev.time_sec - 2.0, lo))
        if (d_after is not None and d_after > thr
                and (d_before is None or d_before <= thr)):
            _log(f"[drift] 자이로 이벤트 {ev.time_sec:.1f}s 와 일치")
            return min(ev.time_sec + 2.0, hi)
    while hi - lo > min_gap:
        mid = (lo + hi) / 2
        d = _drift_deg(vid_l, vid_r, offset, lens, a, mid)
        if d is None or d <= thr:
            lo = mid
        else:
            hi = mid
    return hi


def _realign_after(vid_l, vid_r, offset, lens, boundary, t1, a0):
    """경계 직후 프레임에서 재정합 (수평/센터링은 초기 정합 재사용 —
    세그먼트 간 뷰·출력 크기가 유지된다)."""
    last_err = None
    for dt in (5.0, 35.0, 65.0, 125.0):
        t = boundary + dt
        if t >= t1:
            break
        imgs = _read_pair(vid_l, vid_r, offset, t)
        if imgs is None:
            continue
        try:
            a = estimate_alignment(imgs[0], imgs[1], lens, log=_log,
                                   reuse_level=a0)
            return a, t
        except RuntimeError as e:
            last_err = e
            _log(f"[drift] t={t:.0f}s 재정합 실패: {e}")
    raise RuntimeError(f"재정합 실패 ({boundary:.0f}s 이후): {last_err}")


def _scan_segments(vid_l, vid_r, offset, lens, a0, t0, t1, align_sec,
                   bumps, args):
    """check_every 간격으로 정합 잔차를 확인, 틀어진 지점마다 세그먼트 추가."""
    segments = [{"start_sec": t0, "align_sec": align_sec, "alignment": a0}]
    base = _drift_deg(vid_l, vid_r, offset, lens, a0, align_sec)
    thr = max(args.drift_thresh, 3.0 * (base or 0.0))
    _log(f"[drift] 기준 잔차 {base if base is not None else -1:.2f}° "
         f"→ 문턱 {thr:.2f}°, {args.check_every:.0f}s 간격 스캔")
    a_cur, t_good = a0, align_sec
    t = align_sec + args.check_every
    while t < t1 - 5.0:
        d = _drift_deg(vid_l, vid_r, offset, lens, a_cur, t)
        if d is not None and d > thr:
            boundary = _locate_drift(vid_l, vid_r, offset, lens, a_cur,
                                     t_good, t, thr, bumps)
            _log(f"[drift] t={t:.0f}s 잔차 {d:.2f}° > {thr:.2f}° — "
                 f"경계 {boundary:.1f}s, 재정합")
            a_cur, t_al = _realign_after(vid_l, vid_r, offset, lens,
                                         boundary, t1, a0)
            segments.append({"start_sec": boundary, "align_sec": t_al,
                             "alignment": a_cur})
            t_good = t_al
            t = max(t, t_al) + args.check_every
        else:
            if d is not None:
                t_good = t
            t += args.check_every
    if len(segments) > 1:
        _log(f"[drift] 세그먼트 {len(segments)}개: "
             + ", ".join(f"{s['start_sec']:.0f}s" for s in segments))
    return segments


def _auto_el_range(lens, alignment, scale=0.1, allow_frac=0.1):
    """커버리지 실측 기반 세로(el) 범위 자동 결정.

    _AUTO_EL_CAP 범위를 저해상도로 워핑해 두 카메라 유효 픽셀의 union 을
    만들고, 상/하 각각 열의 allow_frac 까지만 코너 블랙을 허용하는 행
    구간을 el 로 환산한다. (전 열 커버를 요구하면 파노라마 좌우 극단
    열 때문에 범위가 무의미하게 좁아진다 — 20260712 실측 −36..−12°.
    10% 허용이면 −48..+5° 로 기본값(−45..+10)과 거의 일치.)
    """
    from .core.geometry import build_cylindrical_maps
    R_wl, R_wr = alignment.rotations()
    yaw0, yaw1 = alignment.window()
    el0s, el1s = np.deg2rad(_AUTO_EL_CAP[0]), np.deg2rad(_AUTO_EL_CAP[1])
    f = lens.focal * scale
    out_w = max(64, int((yaw1 - yaw0) * f))
    out_h = max(64, int((np.tan(el1s) - np.tan(el0s)) * f))
    union = np.zeros((out_h, out_w), bool)
    src = np.full((lens.height, lens.width), 255, np.uint8)
    for R in (R_wl, R_wr):
        mx, my = build_cylindrical_maps(lens, R, out_w, out_h,
                                        yaw0, yaw1, el0s, el1s)
        union |= cv2.remap(src, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    if not union.any(axis=0).all():
        return None                       # 커버 안 되는 열 — 자동 결정 불가
    rows = np.arange(out_h)[:, None]
    top = float(np.quantile(np.where(union, rows, out_h).min(axis=0),
                            1.0 - allow_frac))
    bot = float(np.quantile(np.where(union, rows, -1).max(axis=0), allow_frac))
    if bot - top < out_h * 0.2:
        return None
    t1s, t0s = np.tan(el1s), np.tan(el0s)

    def el_of_row(r):
        return float(np.arctan(t1s + r / (out_h - 1) * (t0s - t1s)))

    return el_of_row(bot), el_of_row(top)


def _stitch(pair, pano: Path, lens, lens_name, args):
    left_chain, right_chain, _ = pair
    vid_l = ChapteredVideo(left_chain)
    vid_r = ChapteredVideo(right_chain)
    try:
        bumps = []
        if not args.no_gyro:
            try:
                from .core.gpmf import detect_bump_events
                durs = [n / vid_l.fps for n in vid_l.chapter_frames]
                bumps = [e for e in detect_bump_events(
                    [str(p) for p in left_chain], durs, log=_log)
                    if e.persistent]
                if bumps:
                    _log("[gpmf] 방향 변경 의심 이벤트: "
                         + ", ".join(f"{e.time_sec:.0f}s" for e in bumps))
            except Exception as e:  # noqa: BLE001 — 자이로는 힌트일 뿐
                _log(f"[gpmf] 자이로 읽기 실패 — 힌트 없이 진행: {e}")

        # 설치 완료(settle) 시점: 녹화 시작 후 각도 조절·삼각대 세우는
        # 몇 분은 자이로 방향 변경으로 나타난다 — 그 뒤에서 동기화·정합.
        settle = 0.0
        early = [e.time_sec for e in bumps if e.time_sec < _SETTLE_WINDOW]
        if early:
            settle = early[-1] + _SETTLE_MARGIN
            _log(f"[settle] 설치 완료 추정 {settle:.0f}s "
                 f"(초반 이벤트 {len(early)}개 이후)")

        first_dur = vid_l.chapter_frames[0] / vid_l.fps
        offset, _conf = _estimate_sync(left_chain[0], right_chain[0],
                                       first_dur, prefer_start=settle + 60.0)
        t0 = max(0.0, -offset)
        t1 = min(vid_l.duration, vid_r.duration - offset)
        if args.max_dur:
            t1 = min(t1, t0 + args.max_dur)
        if t1 - t0 < 10:
            raise RuntimeError(f"좌/우 겹치는 구간이 너무 짧음 ({t1 - t0:.0f}s)")
        if settle >= t1 - 30.0:
            _log(f"[settle] {settle:.0f}s 가 구간 끝({t1:.0f}s)에 너무 가까움 "
                 "— 무시")
            settle = 0.0
        a, align_sec = _estimate_align(vid_l, vid_r, offset, lens,
                                       max(t0, settle), t1)

        if args.no_drift_check or t1 - align_sec <= args.check_every:
            segments = [{"start_sec": t0, "align_sec": align_sec,
                         "alignment": a}]
        else:
            segments = _scan_segments(vid_l, vid_r, offset, lens, a,
                                      t0, t1, align_sec, bumps, args)
    finally:
        vid_l.release()
        vid_r.release()

    el0 = np.deg2rad(args.el_bottom)
    el1 = np.deg2rad(args.el_top)
    if args.auto_el:
        rng = _auto_el_range(lens, a)
        if rng is None:
            _log("[auto-el] 커버리지 실측 실패 — 기본 범위 사용")
        else:
            el0, el1 = rng
            _log(f"[auto-el] 검은 테두리 없는 최대 세로 범위 "
                 f"{np.rad2deg(el0):+.1f}°..{np.rad2deg(el1):+.1f}°")
    save_project(pano.with_suffix(".pystitch.json"), {
        "left_files": [str(p) for p in left_chain],
        "right_files": [str(p) for p in right_chain],
        "offset_sec": offset,
        "lens_profile": lens_name,
        "segments": segments,
        "user": {"pitch": 0.0, "roll": 0.0, "yaw": 0.0,
                 "feather_px": args.feather,
                 "el_top_deg": round(float(np.rad2deg(el1)), 2),
                 "el_bottom_deg": round(float(np.rad2deg(el0)), 2)},
    })

    _level_preview(pano, lens, segments, left_chain, right_chain,
                   offset, t0, el0, el1)
    export_pano(lens, segments, [str(p) for p in left_chain],
                [str(p) for p in right_chain], offset, t0, t1, str(pano),
                codec=args.codec, crf=args.crf, feather_px=args.feather,
                el0=el0, el1=el1, gop_sec=args.gop_sec, log=_log)
    if not args.no_proxy:
        _make_scrub_proxy(pano)


def _level_preview(pano: Path, lens, segments, left_chain, right_chain,
                   offset, t0, el0, el1):
    """스티칭 **시작 전** 수평 확인 프리뷰 1장 (<pano>.preview.jpg).

    auto-level 이 발산해 엉뚱한 pitch/roll 을 채택해도 몇 시간 인코드가
    끝나야 눈에 보였다 (효창 실측, devlog 081) — 정합 직후 저장해 인코드
    도중 확인·조기 중단이 가능하게 (사용자 방향: 스티칭 시작 때 확인).
    실패해도 파이프라인은 계속 (프리뷰는 보조 산출물)."""
    try:
        from .core.chapters import ChapteredVideo
        from .core.render import Renderer
        a = segments[0]["alignment"]
        t = max(t0, float(segments[0].get("align_sec", t0)))
        vl = ChapteredVideo([str(p) for p in left_chain])
        vr = ChapteredVideo([str(p) for p in right_chain])
        try:
            ok_l, img_l = vl.read_at(int(round(t * vl.fps)))
            ok_r, img_r = vr.read_at(int(round((t + offset) * vl.fps)))
        finally:
            vl.release()
            vr.release()
        if not (ok_l and ok_r):
            raise RuntimeError("프리뷰 프레임 읽기 실패")
        w0, w1 = a.window(0.0)
        half = (w1 - w0) / 2
        r = Renderer(lens, *a.rotations(0.0, 0.0),
                     a.yaw_auto - half, a.yaw_auto + half,
                     el0, el1, scale=0.3, feather_px=12)
        r.set_gains_from(img_l, img_r)
        prev = pano.with_suffix(".preview.jpg")
        cv2.imwrite(str(prev), r.render(img_l, img_r),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        _log(f"[level] 수평 확인 프리뷰 저장: {prev.name} "
             f"(pitch {np.rad2deg(a.pitch_auto):+.1f}°, "
             f"roll {np.rad2deg(a.roll_auto):+.1f}°) — 인코드 중 눈으로 "
             "확인, 틀어졌으면 중단 후 GUI/재수출로 보정")
    except Exception as e:  # noqa: BLE001 — 보조 산출물, 파이프라인 유지
        _log(f"[level] 프리뷰 생성 실패 (계속 진행): {e}")


def _make_scrub_proxy(pano: Path):
    """스크럽/재생용 1600px 프록시 (.scrub.mp4) — GUI 가 있으면 자동 사용.
    원본 5900px 디코드 부담 없이 재생·탐색이 가볍다."""
    out = pano.with_suffix(".scrub.mp4")
    if out.exists():
        return
    from .core.encoders import available_encoders, ffmpeg_bin
    # 예전엔 잘못된 키("NVENC (H.264)")로 조회해 NVENC 가 있어도 항상
    # libx264 로 폴백했다 — 실제 인코더 이름(h264_nvenc)으로 판정.
    # nvenc_works(): WSL 은 목록에 있어도 디바이스가 없어 런타임 판정 필수.
    from .core.encoders import nvenc_works
    has_nvenc = (any(e == "h264_nvenc"
                     for e in available_encoders().values())
                 and nvenc_works())
    venc = (["-c:v", "h264_nvenc", "-cq", "26"] if has_nvenc
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"])
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(pano),
           "-vf", "scale=1600:-2", "-g", "15", "-an"] + venc + [str(out)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    if r.returncode == 0:
        _log(f"[proxy] 스크럽 프록시 생성: {out.name} "
             f"({time.perf_counter()-t0:.0f}s)")
    else:
        _log(f"[proxy] 프록시 생성 실패 (건너뜀)")


def _analyze(pano: Path, args):
    out = pano.with_suffix(".analysis.json")
    ckpt = pano.with_suffix(".analysis.part.json")
    last = [0.0]

    def progress(i, total, fps):
        now = time.perf_counter()
        if now - last[0] >= 30:
            last[0] = now
            # fps 는 이미 '영상 프레임/초' (i 가 매 프레임 증가) —
            # detect_every 로 또 나누면 ETA 가 3배 과소 (0389/0390 실측)
            rem = (total - i) / max(fps, 1e-6) / 60
            _log(f"[analyze] {i}/{total} ({i/total*100:.1f}%) "
                 f"{fps:.1f}fps 남은 ~{rem:.0f}분")

    from .core.ocr import OnlineCropCache
    cap = cv2.VideoCapture(str(pano))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    cache = OnlineCropCache(fps, min_h=args.min_h, per_track=args.per_track)
    d = analyze_video(str(pano), weights=args.weights,
                      detect_every=args.detect_every,
                      checkpoint_path=str(ckpt), progress=progress,
                      crop_hook=cache.hook, log=_log)
    if d is None:
        raise RuntimeError("분석 취소/실패")
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)
    from .core.ptz import analysis_summary, link_ball_tracks_cached
    analysis_summary(out, d, log=_log)    # 요약 캐시 예열 (GUI 열기 가속)
    link_ball_tracks_cached(out, d, log=_log)   # 트랙 연결 캐시 예열
    n_ball = sum(1 for b in d["balls"] if b is not None)
    _log(f"[analyze] 샘플 {len(d['frames'])}개, 공 {n_ball}개, "
         f"선수 검출 {sum(len(p) for p in d['players'])} → {out.name} "
         f"(OCR 크롭 캐시 {cache.n}장)")
    return cache


def _build_match(pano: Path, args):
    """pano ↔ AX700 영상을 호각 동기로 연결 → <pano>.match.json.

    headless 가 이미 아는 연관(GoPro 체인→pano)과 호각 대역상관으로 찾은
    AX700 시계 모델을 경기 파일로 만들어 PitchWatch "경기 열기" 로 바로
    불러온다 (사용자 방향). 호각 매칭이 부족한 영상(옆 구장·다른 경기)은
    제외 — 이 게이트가 곧 '어느 AX700 이 이 경기인가' 판정이다."""
    from .core.match import MATCH_SUFFIX, save_match
    out = pano.with_suffix(MATCH_SUFFIX)
    if out.exists() and not args.force:
        _log(f"[match] {out.name} 있음 — 건너뜀")
        return
    from .core.audio import (extract_audio, load_whistle_track,
                             save_whistle_track, whistle_events,
                             whistle_track)
    from .core.sync_multi import sync_by_whistles

    def events_of(p: Path):
        _tr, ev = load_whistle_track(p)
        if _tr is not None:
            return ev
        _log(f"[match] 호각 트랙 계산: {p.name} (오디오 추출 — 수 분)")
        x = extract_audio(str(p))
        tr = whistle_track(x)
        ev = whistle_events(tr)
        save_whistle_track(p, tr, ev)
        return ev

    ev_p = events_of(pano)
    alts = []
    for d in args.ax700:
        vids = sorted(set(Path(d).glob("*.MP4")) | set(Path(d).glob("*.mp4")))
        for v in vids:
            try:
                r = sync_by_whistles(ev_p, events_of(v))
            except Exception as e:  # noqa: BLE001 — 개별 영상 실패는 건너뜀
                _log(f"[match] {v.name} 동기 실패 (무시): {e}")
                continue
            if not r or r["n"] < 4 or r["rms_s"] > 1.0:
                _log(f"[match] {v.name}: 호각 매칭 부족 — 제외"
                     + (f" (n={r['n']}, rms={r['rms_s']:.2f}s)" if r else ""))
                continue
            alts.append({"video": str(v),
                         "clock": {"offset": r["offset"],
                                   "drift": r["drift"]},
                         "stage": "whistle"})
            _log(f"[match] 연결: {v.name} offset {r['offset']:+.1f}s "
                 f"drift {r['drift']:.6f} (호각 {r['n']}쌍, "
                 f"rms {r['rms_s']:.2f}s)")
    save_match(out, {"title": pano.stem,
                     "halves": [{"label": "경기", "primary": str(pano),
                                 "alts": alts}]})
    _log(f"[match] 경기 파일 생성: {out.name} (alt {len(alts)}개)"
         + ("" if alts else " — AX700 매칭 없음"))


def _far_augment(pano: Path, args):
    """기존 분석에 원경 타일 사람 검출만 추가 (tid 800001+, 편집 보존)."""
    from .core.ptz import analysis_summary, augment_far_persons
    out = pano.with_suffix(".analysis.json")
    d = json.loads(out.read_text())
    if d.get("far_augment"):
        _log(f"[far] {out.name} 이미 보강됨 — 건너뜀 "
             f"(+{d['far_augment'].get('added', '?')}행)")
        return
    added = augment_far_persons(str(pano), d, weights=args.weights, log=_log)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)
    analysis_summary(out, d, log=_log)    # 요약 캐시 재생성 (스팬/세그)
    _log(f"[far] {out.name} 갱신 (+{added}행)")


def _whistle(pano: Path, args):
    """호각 트랙 추출 (.whistle.json) — 타임라인 호각 레인 + 멀티캠
    동기화(sync_cams)의 전제. 오디오만 읽으므로 파노라마당 수 분."""
    wp = pano.with_suffix(".whistle.json")
    if wp.exists() and not args.force:
        _log(f"[whistle] {wp.name} 있음 — 건너뜀")
        return
    from .core.audio import (
        extract_audio, save_whistle_track, whistle_events, whistle_track,
    )
    t0 = time.perf_counter()
    x = extract_audio(str(pano))
    track = whistle_track(x)
    ev = whistle_events(track)
    save_whistle_track(pano, track, ev)
    _log(f"[whistle] 이벤트 {len(ev)}개 → {wp.name} "
         f"({time.perf_counter() - t0:.0f}s)")


def _ocr(pano: Path, args, cache=None):
    ana = json.loads(pano.with_suffix(".analysis.json").read_text())
    teams = classify_teams(ana)
    picked = collect_ocr_candidates(
        ana, None, lambda t: teams.get(t, 2), lambda t: t,
        min_h=args.min_h, per_track=args.per_track)
    n_rep = len({r for _, _, r in picked})
    _log(f"[ocr] 후보: 트랙릿 {n_rep}개, 크롭 {len(picked)}장 "
         f"(min_h={args.min_h:.0f}px, 필드 게이트 없음)")
    gpu = False
    if not args.cpu:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except ImportError:
            pass
    if cache is not None and cache.n:
        # P09: 분석 패스가 모아둔 크롭 — 영상 재디코드 없이 OCR
        from .core.ocr import run_jersey_ocr_cached
        cp = cache.picked(lambda t: teams.get(t, 2), lambda t: t)
        _log(f"[ocr] 캐시 경로: 크롭 {len(cp)}장 (재디코드 없음)")
        out = run_jersey_ocr_cached(cp, min_conf=args.min_conf,
                                    min_votes=args.min_votes, gpu=gpu,
                                    log=_log)
    else:
        out = run_jersey_ocr(str(pano), ana, picked, min_conf=args.min_conf,
                             min_votes=args.min_votes, gpu=gpu, log=_log)
    for k in sorted(out, key=lambda k: -out[k]["score"]):
        v = out[k]
        _log(f"[ocr]   #{k:>7s} → {v['num']:>2s}번 "
             f"(점수 {v['score']}, 지분 {v['share']:.0%})")
    save_events(pano, ocr_numbers=out)
    _log(f"[ocr] 제안 {len(out)}건 → {pano.stem}.events.json \"ocr_numbers\"")


class _StageTimer:
    """단계별 소요를 <pano>.timing.json 에 기록 — 파이프라인 성능 비교용.

    스킵된 단계는 기록하지 않는다. 같은 pano 재실행 시 실행된 단계만
    갱신 (P09 전/후 비교는 stage 별 최신값으로).
    """

    def __init__(self, pano: Path):
        self.path = pano.with_suffix(".timing.json")
        try:
            self.doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.doc = {"version": 1, "runs": {}}

    def stage(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                return self

            def __exit__(self, exc_type, *_a):
                timer.doc["runs"][name] = {
                    "start": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
                    "sec": round(time.time() - self.t0, 1),
                    "ok": exc_type is None,
                }
                timer.path.write_text(
                    json.dumps(timer.doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                return False              # 예외는 그대로 전파

        return _Ctx()


def process_pair(pair, out_dir: Path, lens, lens_name, args) -> Path:
    left_chain, right_chain, cost = pair
    pano = out_dir / f"{_pair_name(left_chain)}.mp4"
    _log(f"=== {pano.stem}: L {left_chain[0].name} 외 {len(left_chain)-1}개 "
         f"↔ R {right_chain[0].name} 외 {len(right_chain)-1}개 "
         f"(크기차 {cost:.1%})")
    tm = _StageTimer(pano)
    # 최소 크기 가드: 실패한 인코드가 남긴 0바이트/파편 파일을 '있음'
    # 으로 보고 건너뛰면 분석이 moov 없음으로 죽는다 (실측 2회) —
    # 1MB 미만이면 깨진 잔재로 보고 재스티칭.
    if pano.exists() and pano.stat().st_size < (1 << 20):
        _log(f"[stitch] {pano.name} 이 {pano.stat().st_size}B — 깨진 잔재 "
             "삭제 후 재스티칭")
        pano.unlink()
    if pano.exists() and not args.force:
        _log(f"[stitch] {pano.name} 있음 — 건너뜀")
    else:
        with tm.stage("stitch"):
            _stitch(pair, pano, lens, lens_name, args)
    ana = pano.with_suffix(".analysis.json")
    cache = None
    if ana.exists() and not args.force and not args.reanalyze:
        if args.far_augment:
            with tm.stage("far_augment"):
                _far_augment(pano, args)
        else:
            _log(f"[analyze] {ana.name} 있음 — 건너뜀")
    else:
        with tm.stage("analyze"):
            cache = _analyze(pano, args)
    try:
        if not pano.with_suffix(".whistle.json").exists() or args.force:
            with tm.stage("whistle"):
                _whistle(pano, args)
        else:
            _whistle(pano, args)          # 내부 스킵 로그만
    except Exception as e:  # noqa: BLE001 — 호각은 부가 산출물, 파이프라인 유지
        _log(f"[whistle] 실패 (계속 진행): {e}")
    if args.ax700:
        try:
            with tm.stage("match"):
                _build_match(pano, args)
        except Exception as e:  # noqa: BLE001 — 경기 파일도 부가 산출물
            _log(f"[match] 실패 (계속 진행): {e}")
    if args.no_ocr:
        return pano
    if "ocr_numbers" in load_events_doc(pano) and not args.force:
        _log("[ocr] ocr_numbers 있음 — 건너뜀")
    else:
        with tm.stage("ocr_cached" if cache is not None and cache.n
                      else "ocr_seek"):
            _ocr(pano, args, cache=cache)
    return pano


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="main.py --headless",
        description="좌/우 GoPro 디렉터리에서 파노라마·분석·OCR 무인 실행")
    ap.add_argument("left_dir")
    ap.add_argument("right_dir")
    ap.add_argument("--out", default=None,
                    help="출력 디렉터리 (기본: 두 디렉터리 이름의 공통부분으로 "
                         "<left_dir> 옆에 생성, 예: ..._L/..._R → ...)")
    ap.add_argument("--lens", default="GoPro_HERO5_Black_Wide_4K_16x9")
    ap.add_argument("--codec", default="auto",
                    help="비디오 코덱 (기본 auto: NVENC 있으면 GPU, "
                         "없으면 libx264). 강제하려면 libx264/h264_nvenc")
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--gop-sec", type=float, default=2.0,
                    help="파노라마 키프레임 간격(초, 기본 2). 촘촘할수록 "
                         "원본 탐색이 빠르지만 파일이 커진다. 0=인코더 기본")
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("--el-top", type=float, default=10.0)
    ap.add_argument("--el-bottom", type=float, default=-45.0)
    ap.add_argument("--auto-el", action="store_true",
                    help="세로 범위를 커버리지 실측으로 자동 결정 "
                         "(코너 블랙 열 10%% 허용, 캡 -60..+25°)")
    ap.add_argument("--check-every", type=float, default=300.0,
                    help="드리프트 확인 간격 (초, 기본 300)")
    ap.add_argument("--drift-thresh", type=float, default=0.5,
                    help="드리프트 판정 잔차 하한 (도, 기준 잔차×3과 큰 쪽)")
    ap.add_argument("--no-drift-check", action="store_true")
    ap.add_argument("--no-gyro", action="store_true",
                    help="자이로(GPMF) 힌트 미사용")
    ap.add_argument("--max-dur", type=float, default=None,
                    help="스티칭 구간 상한 (초) — 빠른 확인용")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--min-h", type=float, default=90.0)
    ap.add_argument("--per-track", type=int, default=12)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="OCR 에서 GPU 미사용")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-proxy", action="store_true",
                    help="스크럽/재생 프록시(.scrub.mp4) 생성 안 함")
    ap.add_argument("--force", action="store_true", help="기존 산출물 무시하고 재실행")
    ap.add_argument("--reanalyze", action="store_true",
                    help="기존 .analysis.json 이 있어도 분석을 처음부터 다시 "
                         "(신 검출 코드 전체 반영). 주의: 트랙릿 tid 가 전부 "
                         "바뀌어 기존 수동 편집(역할·병합·번호) 연결이 끊김")
    ap.add_argument("--ax700", nargs="*", default=[], metavar="DIR",
                    help="AX700(회전캠) 영상 디렉터리 — 각 파노라마와 호각 "
                         "대역상관으로 시계를 맞춰 <pano>.match.json 자동 "
                         "생성 (PitchWatch '경기 열기'로 로드). 매칭 부족한 "
                         "영상(옆 구장 등)은 자동 제외")
    ap.add_argument("--far-augment", action="store_true",
                    help="기존 분석에 원경 네이티브 타일 '사람' 검출만 추가 "
                         "(tid 800001+). 기존 트랙릿·수동 편집 보존 — 편집한 "
                         "영상의 저렴한 대안 (재실행 안전: 이미 보강했으면 스킵)")
    args = ap.parse_args(argv)

    profiles = builtin_profiles()
    if args.lens not in profiles:
        print(f"렌즈 프로파일 없음: {args.lens} (사용 가능: {', '.join(profiles)})")
        return 1
    lens = LensProfile.load(profiles[args.lens])

    # 코덱 해석: auto → NVENC(GPU) 있으면 자동 사용
    from .core.encoders import resolve_codec
    resolved = resolve_codec(args.codec)
    if resolved != args.codec:
        _log(f"[encode] 코덱 auto → {resolved}"
             + (" (GPU)" if resolved.endswith("_nvenc") else " (CPU)"))
    args.codec = resolved

    left_dir, right_dir = Path(args.left_dir), Path(args.right_dir)
    out_dir = (Path(args.out) if args.out
               else _default_out_dir(left_dir, right_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"출력 디렉터리: {out_dir}")

    pairs, un_l, un_r = pair_directories(left_dir, right_dir)
    if not pairs:
        print("짝 맞는 영상 없음 — 파일 크기를 확인하세요")
        return 1
    _log(f"짝 {len(pairs)}건:")
    for cl, cr, cost in pairs:
        _log(f"  {cl[0].name} ({chain_size(cl)/2**30:.2f}GB, {len(cl)}챕터) ↔ "
             f"{cr[0].name} ({chain_size(cr)/2**30:.2f}GB, {len(cr)}챕터) "
             f"크기차 {cost:.1%}")
    for chain in un_l:
        _log(f"  짝 없음(L): {chain[0].name}")
    for chain in un_r:
        _log(f"  짝 없음(R): {chain[0].name}")

    failed = 0
    for pair in pairs:
        try:
            process_pair(pair, out_dir, lens, args.lens, args)
        except Exception as e:  # noqa: BLE001 — 한 쌍 실패해도 다음 쌍 계속
            failed += 1
            _log(f"[오류] {_pair_name(pair[0])}: {e}")
    _log(f"완료: {len(pairs) - failed}/{len(pairs)}건 성공 → {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
