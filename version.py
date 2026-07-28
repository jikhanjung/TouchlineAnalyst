"""TouchlineAnalyst 버전 정보 — Single Source of Truth.

pyproject.toml(dynamic version)·PyInstaller spec·InnoSetup·CI 가 전부
이 파일을 읽는다 (CTHarvester VERSION_MANAGEMENT.md 체계 이식, P10).
갱신은 scripts/bump_version.py 로.
"""

import semver

__version__ = "0.1.0"

# semver 라이브러리를 사용해 안전하게 파싱
_ver = semver.VersionInfo.parse(__version__)
__version_info__ = (_ver.major, _ver.minor, _ver.patch)
