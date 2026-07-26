# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

아마추어 축구 경기 영상 파이프라인. 듀얼 GoPro 를 파노라마로 스티칭하고
(PitchStitch), 그 파노라마(또는 AX700 회전 캠 원본)에서 공/선수 검출·
추적·팀 분류·가상 PTZ·하이라이트·리포트를 만든다 (PitchWatch).

**파이프라인:** GoPro 챕터 체인 → 스티칭(equirect) → 검출/추적 분석 →
편집·검수 GUI → PTZ/하이라이트 내보내기. `--headless` 로 전 단계 무인 실행.

## 진입점

- `main.py` — 통합 GUI. `--headless <L_dir> <R_dir>` 로 무인 파이프라인
  (`pystitch/headless.py`: 스티칭→분석→호각→OCR→이벤트→프록시).
- `pitchstitch.py` — 스티칭 전용 앱 (torch 미설치 환경에서도 기동).
- `pitchwatch.py` — 경기 분석 앱 (PtzTab 승격, 파노라마/멀티캠 경기 열기).

## 구조 (실제)

- `pystitch/core/` — 수학·처리 코어 (GUI 무의존, 테스트 대상)
  - 스티칭: `lens` `align` `geometry` `perspective` `export` `render`
    `encoders` `chapters` `pairing` `sync` `sync_multi` `gpmf`
  - 분석: `ptz`(검출/추적/크롭계획/레이더) `tracklets` `ocr` `audio`(호각)
    `events` `highlights` `airborne` `metrics` `report`
  - 경기장/카메라: `field`(랜드마크·파노라마 캘리브) `rotcam`(회전캠
    자기캘리브: RANSAC PnP·프레임 간 이송·point-to-line) `match`(멀티캠)
- `pystitch/gui/` — PyQt6. `ptz_tab.py`(분석 UI 본체, 대형), `main_window`,
  `multicam`, `stats`, `widgets`, `workers`
- `presets/` — 렌즈 프로파일, `venues.json`(경기장 규격), `tracker_pano.yaml`
- `scripts/` — 실험/일회성 (rotcam_*, gapfill, referee, finetune_ball …)
- `tests/` — pytest (합성 검증 위주), `devlog/` — 번호제 작업 로그,
  `docs/heuristics.md` — 도메인 휴리스틱 카탈로그, `docs/ptz_workflow.md`

## 사이드카 규약 (영상 옆 파일)

`<video>.pystitch.json`(스티칭 프로젝트 — **존재 여부로 파노라마/rotcam
판별**), `.ptz.json`(사용자 편집: 역할·병합·번호·랜드마크·확정앵커 —
tid 로 연결), `.analysis.json`(검출 원본) + `.analysis.cache.json`(요약)
+ `.analysis.part.json`(체크포인트), `.ptz_hcache.npz`(rotcam 이송 H),
`.scrub.mp4`(프록시), `.whistle.json`, `.events.json`, `.match.json`(멀티캠).

**tid 네임스페이스:** ByteTrack 소형 정수 / 원경 타일 `_FarTracker`
800001+ / 수동 검출(extra) 900001+. 전체 재분석은 tid 를 갈아치워
`.ptz.json` 편집을 고아로 만든다 — 편집된 영상은 `--far-augment` 사용.

## 개발 명령어

```bash
# WSL 개발/검증 환경 (torch 포함) — 시스템 python 엔 torch 없음
source ~/venv/PyStitch360/bin/activate
python -m pytest tests/ -q          # 전체 테스트 (수십 초)
python -m py_compile <파일>         # 편집 후 최소 검증

# 무인 파이프라인 (실행은 보통 Windows PowerShell 의 PyStitch360 venv)
python main.py --headless <L_dir> <R_dir> --auto-el
#  --reanalyze(분석 처음부터, 편집 고아화 주의) --far-augment(원경만 추가)
```

실제 GUI 실행·대용량 배치는 Windows(D:\projects\TouchlineAnalyst)에서,
영상 원본은 F:\Pictures (WSL 에선 /mnt/f — 9p 라 대용량 I/O 느림).

## 작업 규약

- **작업 하나 끝나면 devlog 작성 + commit + push** (`devlog/YYYYMMDD_NNN_
  slug.md`, NNN 은 최대+1. 최근 devlog 형식·톤 참고).
- 수정 후 `py_compile` + pytest, 가능하면 실데이터(/mnt/f 사이드카)로
  검증. 합성 검증은 tests/ 에 남긴다.
- 무거운 계산은 "한 번 계산하면 캐시" 원칙 (top_black, H 캐시, 요약
  캐시 등 선례). 느린 단계는 로그(타임스탬프)로 보이게.
- GUI 스레드에서 수 초 이상 걸리는 작업은 QThread 워커 + 진행 로그.

## 도메인 휴리스틱

축구 도메인 사전지식 기반의 약한 규칙들(GK 단일성, 킥오프 기하, 공/선수
속도 상한 등)은 **docs/heuristics.md** 에 카탈로그로 관리한다. 새 휴리스틱
추가·파라미터 변경·폐기 시 반드시 그 문서에 등록하고, 가능하면 tests/ 에
합성 테스트를 함께 둔다.
