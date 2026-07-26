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


# h264 NVENC 하드웨어 한계 — 이보다 넓은 파노라마는 'No capable devices
# found' 로 죽는다 (5312px 실측). HEVC NVENC 는 8192 까지.
H264_NVENC_MAX_W = 4096
HEVC_NVENC_MAX_W = 8192


@lru_cache(maxsize=4)
def nvenc_works(encoder: str = "h264_nvenc") -> bool:
    """NVENC 가 **런타임에 실제로 동작**하는가 — 1프레임 테스트 인코드.

    -encoders 목록은 컴파일 여부일 뿐이다 (WSL/드라이버 불일치에서 목록엔
    있어도 실패). 256px 로 테스트 — 64px 는 NVENC 최소 해상도 미만이라
    가짜 실패가 난다 (실측)."""
    try:
        r = subprocess.run(
            [ffmpeg_bin(), "-v", "error", "-f", "lavfi",
             "-i", "color=black:s=256x256:d=0.1", "-frames:v", "1",
             "-c:v", encoder, "-f", "null", "-"],
            capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def resolve_codec(codec: str, width: int | None = None) -> str:
    """'auto' → 실제 동작하는 GPU 인코더, 없으면 libx264.

    width 를 알면 해상도 한계 반영: h264 NVENC 는 4096px 폭까지라 5~6K
    파노라마는 hevc_nvenc(8192px)로, 그것도 안 되면 libx264. width 없인
    h264 기준으로만 판정 (파노라마 인코드는 export 쪽에서 폭과 함께 재판정).
    명시된 코덱(libx264/h264_nvenc 등)은 그대로 둔다."""
    if codec and codec != "auto":
        return codec
    encs = set(available_encoders().values())
    w = int(width or 0)
    if w <= H264_NVENC_MAX_W and "h264_nvenc" in encs \
            and nvenc_works("h264_nvenc"):
        return "h264_nvenc"
    if w <= HEVC_NVENC_MAX_W and "hevc_nvenc" in encs \
            and nvenc_works("hevc_nvenc"):
        return "hevc_nvenc"
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
