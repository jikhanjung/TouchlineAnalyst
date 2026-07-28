"""TouchlineLabs / TouchlineAnalyst 사용자 경로 (P10).

CTHarvester utils/paths.py 패턴 이식. OS 설정 위치는 platformdirs 가
결정한다 (Windows %LOCALAPPDATA%, macOS ~/Library/Application Support,
Linux ~/.config — XDG·현지화 폴더명까지 정확). 회사명 세그먼트는
platformdirs 의 appauthor 가 Windows 에서만 동작하므로 직접 join.

Qt 를 여기 들이지 않는다 — 커맨드라인 스크립트(headless 등)도 이 모듈을
import 하기 때문 (QSettings 래퍼는 gui/settings.py).

환경변수 오버라이드는 테스트가 실제 홈 디렉터리를 건드리지 않고 경로를
고정하는 통로다.
"""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

COMPANY_NAME = "TouchlineLabs"
PROGRAM_NAME = "TouchlineAnalyst"

#: 설정 디렉터리 루트를 바꾸는 환경변수 (테스트 고정용).
CONFIG_DIR_ENV_VAR = "TOUCHLINE_CONFIG_DIR"
#: 데이터 디렉터리 루트를 바꾸는 환경변수.
DATA_DIR_ENV_VAR = "TOUCHLINE_DATA_DIR"

#: 설정 파일 이름 (QSettings INI — gui/settings.py 가 사용).
SETTINGS_FILENAME = "settings.ini"


def get_config_dir() -> Path:
    """사용자 설정 디렉터리 (…/TouchlineLabs/TouchlineAnalyst)."""
    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path(platformdirs.user_config_dir()) / COMPANY_NAME / PROGRAM_NAME


def get_data_dir() -> Path:
    """사용자 데이터 디렉터리 — **자리만** (P10 결정: 데이터 저장 위치는
    보류, 현행 '영상 옆 사이드카' 동작 불변). 향후 로그·캐시 등이 여기로
    올 수 있다: ~/TouchlineLabs/TouchlineAnalyst."""
    override = os.environ.get(DATA_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / COMPANY_NAME / PROGRAM_NAME


def settings_ini_path() -> Path:
    return get_config_dir() / SETTINGS_FILENAME
