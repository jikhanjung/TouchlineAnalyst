"""TouchlineAnalyst 빌드 스크립트 (P10 — CTHarvester build.py 이식·간소화).

PyInstaller onedir (PitchStitch.exe + PitchWatch.exe, 단일 폴더) →
InnoSetup 인스톨러. Windows 가 1차 대상이지만 PyInstaller 단계는 다른
OS 에서도 돈다 (인스톨러 단계만 Windows 전용).

사용:
    python build.py            # PyInstaller + (Windows 면) 인스톨러
    python build.py --no-installer
BUILD_NUMBER 환경변수는 CI 가 넣는다 (없으면 'local').
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows CI 콘솔은 cp1252 — 한글 로그가 UnicodeEncodeError 로 죽으며
# 진짜 빌드 오류를 가렸다 (첫 CI 실행 실측). UTF-8 로 강제.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from version import __version__ as VERSION  # noqa: E402

SPEC_FILE = "TouchlineAnalyst.spec"
DIST_NAME = "TouchlineAnalyst"


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def run_pyinstaller() -> bool:
    log(f"TouchlineAnalyst v{VERSION} — PyInstaller ({SPEC_FILE})")
    cmd = ["pyinstaller", str(PROJECT_ROOT / SPEC_FILE), "--clean",
           "--noconfirm"]
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        log(f"PyInstaller 실패: {e}")
        return False
    out = PROJECT_ROOT / "dist" / DIST_NAME
    for exe in ("PitchWatch", "PitchStitch"):
        suffix = ".exe" if platform.system() == "Windows" else ""
        if not (out / f"{exe}{suffix}").exists():
            log(f"산출물 누락: {exe}{suffix}")
            return False
    log(f"PyInstaller 완료 → {out}")
    return True


def prepare_iss() -> str | None:
    """템플릿의 {{VERSION}} 치환 + 상대 경로를 절대 경로로 재작성."""
    template = PROJECT_ROOT / "InnoSetup" / "TouchlineAnalyst.iss.template"
    if not template.exists():
        log(f"템플릿 없음: {template}")
        return None
    iss = template.read_text(encoding="utf-8")
    iss = iss.replace("{{VERSION}}", VERSION)
    iss = iss.replace("..\\dist\\", str(PROJECT_ROOT / "dist") + "\\")
    iss = iss.replace("..\\InnoSetup\\Output",
                      str(PROJECT_ROOT / "InnoSetup" / "Output"))
    tmp = Path(tempfile.gettempdir()) / f"TouchlineAnalyst_build_{os.getpid()}.iss"
    tmp.write_text(iss, encoding="utf-8")
    return str(tmp)


def build_installer() -> bool:
    if platform.system() != "Windows":
        log("인스톨러는 Windows 에서만 — 건너뜀")
        return True
    temp_iss = prepare_iss()
    if not temp_iss:
        return False
    iscc = next((p for p in (
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe") if Path(p).exists()), None)
    if not iscc:
        log("InnoSetup 미설치 — 인스톨러 건너뜀 (https://jrsoftware.org/isdl.php)")
        Path(temp_iss).unlink(missing_ok=True)
        return True
    build_number = os.environ.get("BUILD_NUMBER", "local")
    try:
        subprocess.run([iscc, f"/DBuildNumber={build_number}", temp_iss],
                       check=True)
    except subprocess.CalledProcessError as e:
        log(f"ISCC 실패: {e}")
        return False
    finally:
        Path(temp_iss).unlink(missing_ok=True)
    out = (PROJECT_ROOT / "InnoSetup" / "Output"
           / f"TouchlineAnalyst_v{VERSION}_build{build_number}_Installer.exe")
    log(f"인스톨러: {out}" if out.exists() else "인스톨러 산출물 확인 실패")
    return out.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-installer", action="store_true")
    args = ap.parse_args()
    if not run_pyinstaller():
        return 1
    if not args.no_installer and not build_installer():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
