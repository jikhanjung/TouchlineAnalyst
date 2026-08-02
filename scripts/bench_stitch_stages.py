"""스티칭 단계별 벤치 — WSL vs Windows 처리량 차이의 출처를 분해한다 (devlog 096).

같은 머신에서 WSL 이 Windows 보다 1.66배 빨랐다. 9p 를 쓰는 WSL 이 이겼으므로
파일 I/O 는 원인이 아니다. 어디서 갈리는지 보려고 세 단계를 따로 잰다:

  1. decode  — ChapteredVideo 로 원본 4K 프레임 읽기 (I/O + 디코드)
  2. render  — Renderer.render() (cv2.remap 2회 + 블렌딩), 디코드 비용 없음
  3. pipe    — I420 변환 + ffmpeg stdin 으로 원시 프레임 밀어넣기

사용: python scripts/bench_stitch_stages.py <프로젝트.pystitch.json> [N]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.chapters import ChapteredVideo  # noqa: E402
from pystitch.core.encoders import ffmpeg_bin  # noqa: E402
from pystitch.core.export import pipe_format  # noqa: E402
from pystitch.core.lens import LensProfile, builtin_profiles  # noqa: E402
from pystitch.core.project import alignment_from_dict  # noqa: E402
from pystitch.core.render import Renderer  # noqa: E402

PROJ = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def cpu_time():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def localize(p: str) -> str:
    """프로젝트에 박힌 경로를 현재 OS 로 (F:\\... ↔ /mnt/f/...)."""
    if os.name == "nt":
        return p.replace("/mnt/f/", "F:\\").replace("/", "\\")
    return p.replace("F:\\", "/mnt/f/").replace("\\", "/")


d = json.loads(PROJ.read_text(encoding="utf-8"))
lens = LensProfile.load(builtin_profiles()[d["lens_profile"]])
a = alignment_from_dict(d["segments"][0]["alignment"])
u = d["user"]
el0, el1 = np.deg2rad(u["el_bottom_deg"]), np.deg2rad(u["el_top_deg"])
LF = [localize(p) for p in d["left_files"]]
RF = [localize(p) for p in d["right_files"]]
t_start = float(d["segments"][0]["align_sec"])
f_l = int(round(t_start * 29.97))
f_r = int(round((t_start + d["offset_sec"]) * 29.97))

print(f"플랫폼 {sys.platform} | cv2 {cv2.__version__} | "
      f"python {sys.version.split()[0]}")
print(f"ffmpeg {ffmpeg_bin()}")
ver = subprocess.run([ffmpeg_bin(), "-hide_banner", "-version"],
                     capture_output=True, text=True).stdout.splitlines()[0]
print(f"  {ver}")

w0, w1 = a.window(0.0)
half = (w1 - w0) / 2
r = Renderer(lens, *a.rotations(0.0, 0.0), a.yaw_auto - half,
             a.yaw_auto + half, el0, el1, scale=1.0, feather_px=40)
OW, OH = r.out_w, r.out_h
print(f"출력 {OW}x{OH}, N={N}\n")

# ---------------------------------------------------------------- 1. decode
vl, vr = ChapteredVideo(LF), ChapteredVideo(RF)
vl.seek_frame(f_l)
vr.seek_frame(f_r)
ok, il = vl.read()
ok2, ir = vr.read()
r.set_gains_from(il, ir)
vl.seek_frame(f_l)
vr.seek_frame(f_r)
c0, w_0 = cpu_time(), time.perf_counter()
n = 0
for _ in range(N):
    okl, _a = vl.read()
    okr, _b = vr.read()
    if not (okl and okr):
        break
    n += 1
dec_w, dec_c = time.perf_counter() - w_0, cpu_time() - c0
vl.release()
vr.release()
print(f"1. decode (4K×2)  {n/dec_w:6.2f} fps   wall {dec_w:6.2f}s  "
      f"CPU {dec_c:6.2f}s  ({dec_c/max(dec_w,1e-9):.2f} 코어)")

# ---------------------------------------------------------------- 2. render
c0, w_0 = cpu_time(), time.perf_counter()
for _ in range(N):
    out = r.render(il, ir)
ren_w, ren_c = time.perf_counter() - w_0, cpu_time() - c0
print(f"2. render         {N/ren_w:6.2f} fps   wall {ren_w:6.2f}s  "
      f"CPU {ren_c:6.2f}s  ({ren_c/max(ren_w,1e-9):.2f} 코어)")

# ------------------------------------------------------- 2b. I420 변환만
c0, w_0 = cpu_time(), time.perf_counter()
for _ in range(N):
    buf = cv2.cvtColor(out, cv2.COLOR_BGR2YUV_I420)
cvt_w = time.perf_counter() - w_0
print(f"2b. BGR→I420 변환 {N/cvt_w:6.2f} fps   wall {cvt_w:6.2f}s")

# ---------------------------------------------------------------- 3. pipe
fmt = pipe_format(OW, OH)
payload = (cv2.cvtColor(out, cv2.COLOR_BGR2YUV_I420) if fmt == "yuv420p"
           else out).tobytes()
cmd = [ffmpeg_bin(), "-v", "error", "-f", "rawvideo", "-pix_fmt", fmt,
       "-s", f"{OW}x{OH}", "-r", "30", "-i", "-", "-f", "null", "-"]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
c0, w_0 = cpu_time(), time.perf_counter()
try:
    for _ in range(N):
        p.stdin.write(payload)
except BrokenPipeError:
    pass
p.stdin.close()
p.wait()
pipe_w, pipe_c = time.perf_counter() - w_0, cpu_time() - c0
mb = len(payload) * N / 2 ** 20
print(f"3. pipe ({fmt:8s}) {N/pipe_w:6.2f} fps   wall {pipe_w:6.2f}s  "
      f"CPU {pipe_c:6.2f}s   {mb/pipe_w:7.1f} MB/s")

print(f"\n프레임당: decode {dec_w/max(n,1)*1000:6.1f}ms  "
      f"render {ren_w/N*1000:6.1f}ms  변환 {cvt_w/N*1000:5.1f}ms  "
      f"pipe {pipe_w/N*1000:6.1f}ms")
