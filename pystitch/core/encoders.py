"""사용 가능한 ffmpeg 비디오 인코더 감지."""
from __future__ import annotations

import subprocess
from functools import lru_cache

# 표시 이름 → (인코더, 추가 인자 빌더)
CANDIDATES = {
    "libx264 (H.264, CPU)": "libx264",
    "libx265 (HEVC, CPU)": "libx265",
    "h264_nvenc (H.264, NVIDIA GPU)": "h264_nvenc",
    "hevc_nvenc (HEVC, NVIDIA GPU)": "hevc_nvenc",
}


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    """ffmpeg 실행 파일 탐색: PATH → 플랫폼별 흔한 설치 위치.

    Windows 에서 winget 설치 직후에는 이전에 뜬 터미널의 PATH 에 없어서
    콤보가 libx264 폴백만 보여주는 사고가 있었다 — 직접 경로도 훑는다.
    """
    import os
    import shutil
    import sys
    p = shutil.which("ffmpeg")
    if p:
        return p
    if sys.platform == "win32":
        for c in (os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
                  r"C:\ffmpeg\bin\ffmpeg.exe",
                  os.path.expandvars(r"%ProgramData%\chocolatey\bin\ffmpeg.exe")):
            if os.path.exists(c):
                return c
    return "ffmpeg"


@lru_cache(maxsize=1)
def available_encoders() -> dict[str, str]:
    """실제 사용 가능한 인코더만 (표시 이름 → ffmpeg 인코더 이름)."""
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        out = ""
    result = {}
    for label, enc in CANDIDATES.items():
        if f" {enc} " in out:
            result[label] = enc
    if not result:
        result = {"libx264 (H.264, CPU)": "libx264"}
    return result


def resolve_codec(codec: str) -> str:
    """'auto' → NVENC(GPU) 있으면 h264_nvenc, 없으면 libx264.
    명시된 코덱(libx264/h264_nvenc 등)은 그대로 둔다."""
    if codec and codec != "auto":
        return codec
    for enc in available_encoders().values():
        if enc == "h264_nvenc":
            return enc
    return "libx264"


def encoder_args(encoder: str, crf: int) -> list[str]:
    """인코더별 품질/프리셋 인자."""
    if encoder.endswith("_nvenc"):
        # NVENC 는 CRF 대신 CQ. p4 = 중간 프리셋
        args = ["-c:v", encoder, "-preset", "p4", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0"]
    else:
        args = ["-c:v", encoder, "-preset", "fast", "-crf", str(crf)]
    if encoder in ("libx265", "hevc_nvenc"):
        args += ["-tag:v", "hvc1"]
    return args
