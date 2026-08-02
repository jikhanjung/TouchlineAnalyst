"""GPU 상주 스티칭 프로토타입 — render.py 를 건드리지 않는 별도 경로.

NVDEC → cv2.cuda.remap → GPU 블렌딩 → NVENC. CPU 경로(현행)와 처리량·CPU
사용량·출력 일치도를 비교한다. 정합/맵/가중치는 기존 Renderer 에서 그대로
꺼내 쓰므로 기하는 동일하다.

**CUDA 빌드 OpenCV 가 필요하다** (PyPI opencv-python 엔 cv2.cuda 가 없다).
설치·검증 경위는 devlog 093 참고. WSL(Ubuntu 24.04) 요약:

    # cudawarped 의 CUDA wheel (Ubuntu 22.04 빌드) — 파일명 유지 필수
    pip install --target $DIR --no-deps opencv_contrib_python-4.13.0.90-cp37-abi3-linux_x86_64.whl
    pip install --target $DIR --no-deps nvidia-npp        # NPP 는 torch 가 안 끌고 온다
    # 22.04 빌드라 ffmpeg 4.x(.so.58/56/5) 5종이 필요 — 시스템 안 건드리고 별도 prefix 로
    micromamba create -p $DIR/ff4 -c conda-forge ffmpeg=4.4
    export LD_LIBRARY_PATH=<venv>/nvidia/*/lib:$DIR/nvidia/cu13/lib:/usr/lib/wsl/lib:$DIR/ff4/lib
    export PYTHONPATH=$DIR

CUDA Toolkit 전체 설치는 불필요했다 — torch 가 끌고 온 pip nvidia-* 런타임
(CUDA 13.0)이 13.1 빌드를 마이너 버전 호환으로 받아준다.

사용: python scripts/gpu_pipeline_probe.py [프레임수] [scale]
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/mnt/d/projects/TouchlineAnalyst")
from pystitch.core.chapters import ChapteredVideo
from pystitch.core.lens import LensProfile, builtin_profiles
from pystitch.core.project import alignment_from_dict
from pystitch.core.render import Renderer, seam_weights

SP = Path("/tmp/claude-1000/-mnt-d-projects-TouchlineAnalyst/"
          "ad7a4742-c0fc-4da7-a82f-d2c899b4dcca/scratchpad")
PROJ = Path("/mnt/f/Pictures/20260801_GoPro/pano_0001.pystitch.json")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
cc = cv2.cudacodec


def cpu_time():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def wp(p):
    return p.replace("F:\\", "/mnt/f/").replace("\\", "/")


d = json.loads(PROJ.read_text(encoding="utf-8"))
lens = LensProfile.load(builtin_profiles()[d["lens_profile"]])
a = alignment_from_dict(d["segments"][0]["alignment"])
u = d["user"]
el0, el1 = np.deg2rad(u["el_bottom_deg"]), np.deg2rad(u["el_top_deg"])
LF = [wp(p) for p in d["left_files"]]
RF = [wp(p) for p in d["right_files"]]
offset = d["offset_sec"]
t_start = float(d["segments"][0]["align_sec"])

w0, w1 = a.window(0.0)
half = (w1 - w0) / 2
yaw0, yaw1 = a.yaw_auto - half, a.yaw_auto + half
R_wl, R_wr = a.rotations(0.0, 0.0)

# 기준 렌더러 — 맵/가중치/게인의 단일 출처
r = Renderer(lens, R_wl, R_wr, yaw0, yaw1, el0, el1,
             scale=SCALE, feather_px=40)
OW, OH = r.out_w, r.out_h
print(f"출력 {OW}x{OH}, {N}프레임, scale={SCALE}")

fps_src = 29.97
f_l = int(round(t_start * fps_src))
f_r = int(round((t_start + offset) * fps_src))

# ---------------------------------------------------------------- CPU 경로
vl = ChapteredVideo(LF)
vr = ChapteredVideo(RF)
vl.seek_frame(f_l)
vr.seek_frame(f_r)
ok, il = vl.read()
ok2, ir = vr.read()
r.set_gains_from(il, ir)
vl.seek_frame(f_l)
vr.seek_frame(f_r)
c0, w_0 = cpu_time(), time.perf_counter()
cpu_last = None
for i in range(N):
    ok, il = vl.read()
    ok2, ir = vr.read()
    if not (ok and ok2):
        break
    cpu_last = r.render(il, ir)
cpu_w, cpu_c = time.perf_counter() - w_0, cpu_time() - c0
vl.release()
vr.release()
print(f"CPU 경로   : {i+1}프레임 wall {cpu_w:6.2f}s ({(i+1)/cpu_w:5.2f} fps)  "
      f"CPU {cpu_c:6.2f}s")

# ---------------------------------------------------------------- GPU 경로
# 맵을 GPU 로 (Renderer 의 float 맵 = 소스 픽셀 좌표)
(mxl, myl), (mxr, myr) = r._float_maps
g_mxl, g_myl = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
g_mxr, g_myr = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
g_mxl.upload(mxl); g_myl.upload(myl)
g_mxr.upload(mxr); g_myr.upload(myr)

# 심 가중치 (CPU 에서 1회 계산 후 상주)
wl_full = seam_weights(r._masks[0], r._masks[1], yaw0, yaw1,
                       (yaw0 + yaw1) / 2, max(2, int(40 * SCALE)))
# 채널 게인을 가중치에 미리 접어넣는다 — 프레임당 곱셈 2회 절약
gain_l = np.array(r.gain_l[:3], np.float32)
gain_r = np.array(r.gain_r[:3], np.float32)
w_l3 = cv2.cvtColor(wl_full, cv2.COLOR_GRAY2BGR) * gain_l[None, None, :]
w_r3 = cv2.cvtColor(1.0 - wl_full, cv2.COLOR_GRAY2BGR) * gain_r[None, None, :]
g_wl = cv2.cuda.GpuMat(); g_wl.upload(np.ascontiguousarray(w_l3, np.float32))
g_wr = cv2.cuda.GpuMat(); g_wr.upload(np.ascontiguousarray(w_r3, np.float32))

pl = cc.VideoReaderInitParams(); pl.firstFrameIdx = f_l
pr = cc.VideoReaderInitParams(); pr.firstFrameIdx = f_r
rl = cc.createVideoReader(LF[0], params=pl); rl.set(cc.BGR)
rr = cc.createVideoReader(RF[0], params=pr); rr.set(cc.BGR)

writer = None
out_mp4 = str(SP / "gpu_pipeline_out.mp4")
gpu_last = None
c0, w_0 = cpu_time(), time.perf_counter()
n = 0
for i in range(N):
    okl, gl = rl.nextFrame()
    okr, gr = rr.nextFrame()
    if not (okl and okr):
        break
    wl_ = cv2.cuda.remap(gl, g_mxl, g_myl, cv2.INTER_LINEAR)
    wr_ = cv2.cuda.remap(gr, g_mxr, g_myr, cv2.INTER_LINEAR)
    fl = wl_.convertTo(cv2.CV_32FC3)
    fr = wr_.convertTo(cv2.CV_32FC3)
    blended = cv2.cuda.add(cv2.cuda.multiply(fl, g_wl),
                           cv2.cuda.multiply(fr, g_wr))
    out8 = blended.convertTo(cv2.CV_8UC3)
    if writer is None:
        writer = cc.createVideoWriter(out_mp4, (OW, OH), cc.HEVC,
                                      fps_src, cc.BGR)
    writer.write(out8)
    gpu_last = out8
    n += 1
gpu_w, gpu_c = time.perf_counter() - w_0, cpu_time() - c0
if writer is not None:
    writer.release()
print(f"GPU 경로   : {n}프레임 wall {gpu_w:6.2f}s ({n/gpu_w:5.2f} fps)  "
      f"CPU {gpu_c:6.2f}s   (NVENC 인코딩 포함)")
print(f"  wall {cpu_w/max(gpu_w,1e-9):.2f}x 빠름,  CPU {cpu_c/max(gpu_c,1e-9):.1f}x 적음")

# ---------------------------------------------------------------- 일치도
if cpu_last is not None and gpu_last is not None:
    g = gpu_last.download()
    d_ = np.abs(g.astype(np.float64) - cpu_last.astype(np.float64))
    mse = (d_ ** 2).mean()
    print(f"\n출력 일치도 (마지막 프레임): PSNR "
          f"{'inf' if mse == 0 else f'{10*np.log10(255**2/mse):.1f}dB'}  "
          f"평균차 {d_.mean():.2f}  최대차 {d_.max():.0f}")
    cv2.imwrite(str(SP / "gpu_last.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(SP / "cpu_last.jpg"), cpu_last, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("인코드 산출:", out_mp4, os.path.getsize(out_mp4) if os.path.exists(out_mp4) else "없음")
