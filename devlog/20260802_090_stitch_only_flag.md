# 090 — headless --stitch-only (+ 20260801 신규 소재 확인)

2026-08-02. 8/1 촬영분이 들어와 스티칭 큐가 3주 만에 다시 찼다. 파노라마만
먼저 뽑고 분석은 뒤로 미루려는데 전용 플래그가 없어 추가했다.

## 왜 스티칭만인가

`process_pair` 는 스티칭 → 분석 → 호각 → OCR 을 한 줄로 잇는다. 지금
분석까지 태우면 TODOs #1(far 트랙릿 파편화 튜닝, 길이 중앙값 27프레임)
전에 만든 트랙릿이 되어 튜닝 후 재분석 대상이 된다. 파노라마는 분석
파라미터와 무관하므로 먼저 뽑아두는 게 손해가 없다.

## 바꾼 것

`--stitch-only` — 스티칭(+`--venue` 시딩)까지만 하고 반환.

- `_seed_venue` **뒤에** 반환한다. 경기장 시딩은 스티칭 산출물에 딸린
  것이고 비용이 0 이라 생략할 이유가 없다.
- 나중에 플래그를 빼고 **같은 명령을 다시 돌리면** 스티칭은 기존 산출물
  게이팅(`pano.exists()` → "있음 — 건너뜀")에 걸려 분석부터 이어간다.
  별도 재개 로직이 필요 없다.

`tests/test_headless_stitch_only.py` 4건 — 무거운 단계를 스텁으로 바꿔
호출 순서만 검증: 스티칭만 호출됨 / 플래그 없으면 전 단계 / 스티칭 후
재실행 시 stitch 건너뛰고 analyze 진입 / stitch-only 에서도 venue 시딩.

## 소재 (F:\Pictures\20260801_*)

`pair_directories` 실측 — 한 디렉터리에 세션 2개가 섞여 있는데 크기차가
0.01% 라 짝짓기는 무난하다.

| 출력 | L ↔ R | 챕터 | 크기 | 길이 |
|---|---|---|---|---|
| pano_0001 | GOPR0001 ↔ GOPR0397 | 2 | 5.86GB | 13.9분 |
| pano_0002 | GOPR0002 ↔ GOPR0398 | 8 | 27.26GB | 64.9분 |

- 3840×2160 h264 29.97fps, 짝 없는 체인 없음. 출력은 `20260801_GoPro/`.
- `20260801_GoPro_R/error/` 에 `GOPR0397.MP4` 등 **동일 크기 중복본**이
  있다(카드 복사 도구 격리 폴더로 추정). `group_directory` 가
  `glob("*.MP4")` — 비재귀라 파이프라인에 안 잡힌다. 실측으로도 0397 이
  2챕터로 잡혀 확인됨. 조치 불필요.
- `20260801_AX700` (C0006~C0010, 61.8GB) 도 같이 들어왔다 — 멀티캠
  `--ax700` 대상이지만 이번 스티칭에선 제외.

## Windows 환경 (conda PyStitch360) 점검

`main.py --headless --list-venues` 로 진입점 실동작 확인. python 3.12.13,
numpy 2.5.1, opencv 5.0.0, torch 2.11.0+cu128(CUDA True, 2080 Ti),
ultralytics 8.4.99, lap 0.9.4, ffmpeg 8.1.2(h264/hevc/av1 nvenc).

**미설치 3종 — 이번 스티칭엔 무관:**

| 패키지 | 쓰는 곳 | 영향 |
|---|---|---|
| `semver` | `version.py` (build.py·테스트만 import) | 헤드리스 무관 |
| `platformdirs` | `core/paths.py` ← `gui/settings.py` | GUI 전용 |
| `easyocr` | `core/ocr.py` | 나중에 OCR 돌릴 때 필요 |

`pystitch.headless` 자체는 import 통과한다. 분석까지 이어서 돌릴 때
`pip install easyocr` 만 있으면 되고, 나머지 둘은 GUI·빌드용이다.

## 다음

- 스티칭 실행(2건, ~2.3시간 추정) 후 **devlog 088 의 I420 파이프 개선폭
  측정** — TODOs 대기 항목. 0392 기준 17.44fps 대비 fps 와 ffmpeg CPU%.
- 경기장 프리셋은 미정 — 확인되면 `--venue` 로 시딩하거나 GUI 에서.
