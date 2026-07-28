# P10: 배포 체계 — CI · PyInstaller · InnoSetup · platformdirs

- 날짜: 2026-07-28
- 성격: 계획 (P 문서)
- 방향: Modan2 / CTHarvester 의 구성을 이식해 이 프로젝트도 "설치해서
  쓰는" 형태로. **브랜드는 TouchlineLabs** (PaleoBytes 아님 — 사용자
  방향). code quality(ruff 게이트 등)는 후속으로 미룸 (사용자 방향).

## 0. 템플릿 선택

**CTHarvester 를 주 템플릿**으로 한다 — 더 현대적:
- `version.py` SSOT + `scripts/bump_version.py` (Modan2 manage_version.py
  와 명령 체계 동일)
- `pyproject.toml` dynamic version, Python 3.12 단일 (CI 단순화 근거
  문서화돼 있음)
- workflows: test(lint+pytest, workflow_call 게이트) / build(+커밋수
  빌드번호) / reusable_build / release(tag) / manual-release
- `utils/paths.py` 의 platformdirs 패턴 (env 오버라이드 포함)
- InnoSetup `.iss.template` + build.py 치환

Modan2 는 대조용 (동일 명령 체계 확인).

## 1. 단계

### 1-1. 버전 SSOT
- `version.py` (0.1.0 시작) + `scripts/bump_version.py` 이식
- `pyproject.toml` 신설: dynamic version, 프로젝트 메타, deps

### 1-2. 경로·설정 (platformdirs)
- `pystitch/core/paths.py`: CTHarvester utils/paths.py 패턴 이식
  - config dir = `platformdirs.user_config_dir()/TouchlineLabs/TouchlineAnalyst`
  - env 오버라이드 `TOUCHLINE_CONFIG_DIR` (테스트 고정용)
  - **데이터 저장 위치는 보류** (사용자 검토 중) — data dir 함수는
    자리만 잡고 현행 동작(영상 옆 사이드카)은 불변
- 설정 파일: 기존 `QSettings("PyStitch360","PyStitch360")` (레지스트리)
  호출을 `app_settings()` 헬퍼로 교체 — QSettings INI 포맷으로 config
  dir 의 `settings.ini` 에 저장. 호출부 일괄 치환 (동작 동일, 위치만
  파일로).

### 1-3. PyInstaller
- onedir spec 2개: `PitchStitch.spec`(torch 미포함 — 설계상 torch 없이
  기동), `PitchWatch.spec`(torch/ultralytics 포함)
- **핵심 리스크 = torch 크기**: v1 은 CPU torch 번들 (설치본 큼, 동작
  보장). GPU 는 소스 실행/venv 경로로 문서화. 추후 최적화.
- presets/ (렌즈·venue·tracker yaml) 동봉. YOLO 가중치는 미동봉 —
  ultralytics 최초 실행 자동 다운로드.
- ffmpeg 는 v1 미번들 — 기존 `ffmpeg_bin()` 탐색(winget 경로 포함)에
  의존, 설치 문서에 명시. (번들은 라이선스·크기 검토 후 후속)

### 1-4. InnoSetup
- `InnoSetup/TouchlineAnalyst.iss.template` — 한 인스톨러에 두 앱
  (PitchStitch.exe, PitchWatch.exe), 시작메뉴 그룹 TouchlineLabs
- build.py 가 버전 치환 → iscc

### 1-5. CI (.github/workflows)
- `test.yml`: ruff(advisory)+pytest — Windows+Linux (macOS 는 대상
  아님; ffmpeg 필요 테스트는 시스템 ffmpeg 설치 스텝)
- `build.yml` + `reusable_build.yml`: 커밋수 빌드번호, Windows 빌드
  (PyInstaller 2종 + InnoSetup) → 아티팩트
- `release.yml`(tag v*.*.*) / `manual-release.yml`
- codeql/security/dependabot 은 후속 (code quality 묶음)

### 1-6. 검증·기록
- WSL 에서 pytest 통과 확인, (가능하면) pyinstaller 드라이런
- 실검증은 GitHub Actions 첫 실행으로 — 실패 시 반복 수정
- devlog + P10 갱신

## 2. 결정 사항 (기록)

| 항목 | 결정 |
|---|---|
| 브랜드/회사명 | **TouchlineLabs**, 앱 그룹명 TouchlineAnalyst |
| 설정 저장 | platformdirs config dir 의 INI (QSettings 파일 모드) |
| 데이터 저장 | **보류** — 현행(영상 옆 사이드카) 유지, paths.py 에 자리만 |
| torch | CPU 번들 (PitchWatch), PitchStitch 는 무-torch 경량 |
| ffmpeg / YOLO 가중치 | v1 미번들 (탐색/자동다운로드), 문서화 |
| Python | 3.12 단일 (CTHarvester 근거 준용) |
| code quality 게이트 | 후속 (advisory 만) |

## 3. 진행 상태

- [x] 1-1 버전 SSOT
- [x] 1-2 paths/설정 (42곳 치환, 구 설정 1회 이관)
- [x] 1-3 PyInstaller spec (단일 COLLECT, exe 2개)
- [x] 1-4 InnoSetup (TouchlineLabs, per-user)
- [x] 1-5 CI workflows (node24 세대 액션: checkout@v5/setup-python@v6/artifact@v5)
- [x] 1-6 CI 그린 (3라운드: spec/.gitignore 충돌 → ffmpeg·인코딩 → Windows 테스트 4종. 인스톨러 261MB 아티팩트)
