"""파노라마 내보내기 코어 루프 — GUI ExportWorker 와 헤드리스 모드 공용.

3단 파이프라인 (읽기 → 렌더 → 인코더 쓰기), 세그먼트 경계에서 렌더러
재구성. GUI 의존 없음: 진행/로그/취소는 콜백으로 받는다.
"""
from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .align import Alignment
from .chapters import ChapteredVideo
from .encoders import encoder_args, ffmpeg_bin
from .render import Renderer


def export_pano(lens, segments, left_files, right_files, offset_sec,
                start_sec, end_sec, out_path,
                pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                codec="libx264", crf=19, scale=1.0, feather_px=40,
                ptz=False, persp_k=0.0, persp_m=1.0, el0=None, el1=None,
                progress=None, log=print, cancel=None, gop_sec=2.0) -> str:
    """좌/우 챕터 체인을 정합 세그먼트로 스티칭해 파노라마 영상 인코딩.

    progress(done, total, fps) 는 30프레임마다, cancel() -> bool 은 프레임마다
    확인. 취소되면 RuntimeError("취소됨"). 성공 시 out_path 반환.
    """
    if isinstance(segments, Alignment):
        segments = [{"start_sec": 0.0, "alignment": segments}]
    segments = sorted(segments, key=lambda s: s["start_sec"])
    user = (pitch_user, roll_user, yaw_user)
    out_path = str(out_path)
    abort = [False]

    def cancelled() -> bool:
        return abort[0] or (cancel is not None and cancel())

    vid_l = ChapteredVideo(left_files)
    vid_r = ChapteredVideo(right_files)
    fps = vid_l.fps
    f_start = int(round(start_sec * fps))
    f_end = min(int(round(end_sec * fps)), vid_l.total_frames)
    r_start = int(round((start_sec + offset_sec) * fps))
    total = max(0, f_end - f_start)
    if total == 0:
        raise RuntimeError("내보낼 구간이 비어 있음")

    def segment_index_at(t: float) -> int:
        idx = 0
        for i, s in enumerate(segments):
            if s["start_sec"] <= t + 1e-6:
                idx = i
        return idx

    # 모든 세그먼트가 같은 출력 크기를 갖도록 yaw 범위 폭은 첫 세그먼트 기준 고정
    first_a = segments[segment_index_at(start_sec)]["alignment"]
    w0, w1 = first_a.window(user[2])
    half_range = (w1 - w0) / 2

    # 게인·심 보정 측정용 프레임: 각 세그먼트의 정합 프레임(align_sec) +
    # 이후 120초 간격 후보 (해당 세그먼트 범위 안). 시작/경계 프레임은
    # 무늬 없는 잔디뿐일 수 있어 측정이 생략되곤 한다 — 심 밴드에 특징이
    # 잡히는 프레임이 나올 때까지 시도. 스트리밍 시작 전에 미리 읽어둔다
    # (경계 재구성 시 reader 스레드와 디코더 경합 방지).
    calib_imgs: dict[int, list] = {}
    for i, s in enumerate(segments):
        seg_end = (segments[i + 1]["start_sec"] if i + 1 < len(segments)
                   else end_sec)
        base_t = s.get("align_sec", s["start_sec"])
        frames_i = []
        for dt in (0.0, 120.0, 240.0):
            t = base_t + dt
            if frames_i and t >= seg_end:
                break
            ok_l, cl = vid_l.read_at(int(round(t * fps)))
            ok_r, cr = vid_r.read_at(int(round((t + offset_sec) * fps)))
            if ok_l and ok_r:
                frames_i.append((cl, cr))
        calib_imgs[i] = frames_i

    def make_renderer(seg_i, img_l, img_r) -> Renderer:
        alignment = segments[seg_i]["alignment"]
        cands = calib_imgs.get(seg_i) or [(img_l, img_r)]
        R_wl, R_wr = alignment.rotations(user[0], user[1])
        yaw_c = alignment.yaw_auto + np.deg2rad(user[2])
        e0 = el0 if el0 is not None else alignment.el0
        e1 = el1 if el1 is not None else alignment.el1
        r = Renderer(lens, R_wl, R_wr, yaw_c - half_range, yaw_c + half_range,
                     e0, e1, scale=scale, feather_px=feather_px,
                     persp_k=persp_k, persp_m=persp_m)
        r.set_gains_from(*cands[0])
        for cal_l, cal_r in cands:
            if r.refine_seam(cal_l, cal_r, log=log) > 0:
                break
        return r

    log("정합 렌더러 준비 중...")
    ok_l, img_l = vid_l.read_at(f_start)
    ok_r, img_r = vid_r.read_at(r_start)
    if not (ok_l and ok_r):
        raise RuntimeError("시작 프레임 읽기 실패")
    seg_idx = segment_index_at(start_sec)
    rend = make_renderer(seg_idx, img_l, img_r)
    pano_w, pano_h = rend.out_w, rend.out_h
    out_w, out_h = pano_w, pano_h

    vptz = None
    if ptz:
        from .ptz import VirtualPTZ
        log("가상 PTZ 초기화 (YOLO 로드)...")
        vptz = VirtualPTZ(pano_w, pano_h)
        out_w, out_h = vptz.out_w, vptz.out_h
    # 이번 내보내기 구간 안에 있는 이후 세그먼트 경계 (절대 프레임 번호)
    pending = [(int(round(s["start_sec"] * fps)), i)
               for i, s in enumerate(segments)
               if i > seg_idx and s["start_sec"] < end_sec]

    # 오디오: 좌측 챕터 체인을 concat demuxer 로 연결
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for f in left_files:
            tf.write(f"file '{Path(f).as_posix()}'\n")
        concat_list = tf.name

    duration = total / fps
    # 키프레임 간격(GOP): 기본 250(≈8초)이면 큰 원본 프레임 시크가 느리다
    # (키프레임부터 순차 디코드). gop_sec(기본 2초)로 촘촘히 넣어 원본
    # 탐색을 빠르게 — 대가는 파일 소폭 증가. keyint_min 까지 줘 고정 GOP.
    keyint = max(1, int(round(gop_sec * fps)))
    gop_args = (["-g", str(keyint), "-keyint_min", str(keyint)]
                if gop_sec and gop_sec > 0 else [])
    cmd = ([ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-",
            "-f", "concat", "-safe", "0", "-ss", f"{start_sec}",
            "-t", f"{duration}", "-i", concat_list,
            "-map", "0:v", "-map", "1:a?"]
           + encoder_args(codec, crf) + gop_args
           + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path])
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # 3단 파이프라인: 읽기 → 렌더(이 스레드) → 인코더 쓰기
    vid_l.seek_frame(f_start)
    vid_r.seek_frame(r_start)
    q_in: queue.Queue = queue.Queue(maxsize=4)
    q_out: queue.Queue = queue.Queue(maxsize=4)

    def reader():
        for _ in range(total):
            if cancelled():
                break
            ok_l, im_l = vid_l.read()
            ok_r, im_r = vid_r.read()
            if not (ok_l and ok_r):
                break
            q_in.put((im_l, im_r))
        q_in.put(None)

    def writer():
        while True:
            buf = q_out.get()
            if buf is None:
                break
            try:
                enc.stdin.write(buf)
            except BrokenPipeError:
                abort[0] = True
                break

    log(f"인코딩 시작: {total}프레임 ({total/fps/60:.1f}분 분량)")
    t_reader = threading.Thread(target=reader, daemon=True)
    t_writer = threading.Thread(target=writer, daemon=True)
    t_reader.start()
    t_writer.start()

    t0 = time.perf_counter()
    last_log = t0
    done = 0
    was_cancelled = False
    try:
        while True:
            if cancelled():
                was_cancelled = True
                log("사용자 취소")
                break
            item = q_in.get()
            if item is None:
                break
            abs_frame = f_start + done
            if pending and abs_frame >= pending[0][0]:
                _, si = pending.pop(0)
                t_seg = segments[si]["start_sec"]
                log(f"[segment] {t_seg:.1f}s 경계 — 렌더러 재구성")
                rend = make_renderer(si, *item)
                if (rend.out_w, rend.out_h) != (pano_w, pano_h):
                    raise RuntimeError("세그먼트 출력 크기 불일치")
            frame = rend.render(*item)
            if vptz is not None:
                frame = vptz.process(frame)
            q_out.put(frame.tobytes())
            done += 1
            if done % 30 == 0:
                now = time.perf_counter()
                fps_now = done / (now - t0)
                if progress is not None:
                    progress(done, total, fps_now)
                if now - last_log >= 15:   # 로그에도 주기적으로 생존 신고
                    last_log = now
                    remain = (total - done) / max(fps_now, 1e-9) / 60
                    log(f"[export] {done}/{total} ({done/total:.0%}) "
                        f"{fps_now:.1f}fps, 남은 시간 {remain:.0f}분")
    finally:
        abort[0] = abort[0] or was_cancelled
        # reader 가 가득 찬 큐에 막혀 있지 않도록 비운 뒤 종료 대기
        while not q_in.empty():
            try:
                q_in.get_nowait()
            except queue.Empty:
                break
        t_reader.join(timeout=10)
        q_out.put(None)
        t_writer.join(timeout=60)
        enc.stdin.close()
        enc.wait()
        vid_l.release()
        vid_r.release()
        Path(concat_list).unlink(missing_ok=True)

    if was_cancelled or (cancel is not None and cancel()):
        raise RuntimeError("취소됨")
    el = time.perf_counter() - t0
    log(f"완료: {done} 프레임 / {el:.0f}s = {done/max(el,1e-9):.2f} fps")
    return out_path
