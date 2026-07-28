"""앱 설정 (QSettings INI @ platformdirs config dir) — P10.

기존에는 QSettings("PyStitch360", "PyStitch360") 으로 레지스트리
(Windows)/플랫폼 기본 위치에 저장했다 — 설정 파일이 어디 있는지 보이지
않고 백업/이관이 어렵다. platformdirs 가 정한 config dir 의 settings.ini
파일로 통일한다 (사용자 방향, 위치는 core/paths.py).

최초 실행 시 구 레지스트리 설정을 1회 이관해 최근 파일 목록·UI 상태를
잃지 않는다.
"""
from __future__ import annotations

from PyQt6.QtCore import QSettings

from ..core.paths import get_config_dir, settings_ini_path

_LEGACY_ORG, _LEGACY_APP = "PyStitch360", "PyStitch360"


def app_settings() -> QSettings:
    """settings.ini 를 여는 QSettings (필요 시 구 설정 1회 이관)."""
    ini = settings_ini_path()
    first = not ini.exists()
    get_config_dir().mkdir(parents=True, exist_ok=True)
    s = QSettings(str(ini), QSettings.Format.IniFormat)
    if first:
        legacy = QSettings(_LEGACY_ORG, _LEGACY_APP)
        for k in legacy.allKeys():
            s.setValue(k, legacy.value(k))
        if legacy.allKeys():
            s.sync()
    return s
