# 헤드리스 작업 대상 목록 (F:/Pictures, 2026-07-20 조사)

`python main.py --headless <L> <R>` 대상. 경로는 WSL 기준
`/mnt/f/Pictures/...`. 짝 안의 영상(챕터 체인) 매칭은 크기 기반 자동.

## 처리 대상 (10쌍)

| # | Left | Right | 규모 | 비고 |
|---|------|-------|------|------|
| 1 | 20241013_GoPro5_1_1 | 20241013_GoPro5_2_1 | 영상 2개, 챕터 7, ~30GB | 방향 판정: 카메라 1 = Left (프레임 확인) |
| 2 | 20241013_GoPro5_1_2 | 20241013_GoPro5_2_2 | 영상 1개, 챕터 10, ~41GB | 같은 날 두 번째 경기, 카메라 1 = Left |
| 3 | 20241020_GoPro5_Match1_Left | 20241020_GoPro5_Match1_Right | L 영상 4 / R 영상 3 | 영상 수 불일치 — 짝 없는 1개는 자동 제외됨 |
| 4 | 20241020_GoPro5_Match2_Left | 20241020_GoPro5_Match2_Right | L 영상 3 / R 영상 2, 챕터 13 | 〃 |
| 5 | 20250420_GoPro5_11am_L | 20250420_GoPro5_11am_R | 영상 1개, 챕터 8 (~36GB/쪽) | 디렉터리 크기 차이는 부속 파일 탓, 체인은 대칭 |
| 6 | 20250427_GoPro_Left | 20250427_GoPro_Right | 영상 3개, 챕터 29/30, ~145GB/쪽 | 대용량 |
| 7 | "20251026_GoPro5 2" | 20251026_GoPro5 | 영상 1개, 챕터 25, 99GB/쪽 | 방향 판정: **"GoPro5 2"(GOPR0388) = Left**, 경로 공백 따옴표 필요 |
| 8 | 20251102_GoProLeft | 20251102_GoProRight | 43챕터, 161GB/쪽 | 대용량. L/R **실측 확정(2026-07-25): 폴더명대로** L=GoProLeft(0389) (매칭 1864 vs 107) |
| 9 | 20260621_GoPro5_Left | 20260621_GoPro5_Right | 녹화 여러 개 | L=_Left 확정. **하루 연속녹화** — 짝: L GOPR0391↔R GOPR0002(33GB,~80분), L GOPR0392↔R GOPR0003(142GB,~5~6h 여러경기), R GOPR0001=예전영상(제외). **80분 짝(0391↔0002)만 스티칭**하기로(2026-07-25) — 심링크 디렉터리로 그 녹화만, --out 20260621_GoPro5. 142GB는 파킹 |
| 10 | 20260712_GoPro5_L | 20260712_GoPro5_R | 영상 3(워밍업 포함)/챕터 4, 19GB/쪽 | 스모크 검증 완료 — 전체 길이 실행은 미완 |
| 11 | 20260725_GoPro_L | 20260725_GoPro_R | 영상 4챕터/쪽, 14GB/쪽, 3840×2160 | **효창운동장. GoPro 후반만** (전반 없음). L/R 방향 **실측 확정 2026-07-25: 폴더명대로 L=_L(GOPR0001), R=_R(GOPR0396)** (매칭 1423 vs 92, 스왑 불필요). AX700(20260725_AX700: C0003 전반15분·C0005 후반35분)이 별도 소스. venue: 효창(104×67, R9). 후반만 겹쳐 whistle sync 대역상관 검증 |

카메라 시리얼(GOPR5xxx/GOPR0xxx)이 날짜마다 좌우가 바뀌므로
(20241020 은 Left=5xxx, 20250420 은 L=0xxx) 시리얼로 방향 추정 불가.
1·2·7번은 중간 프레임 육안 판정으로 확정 (2026-07-20): **센터서클이
프레임 오른쪽 가장자리에 보이는 카메라가 Left** (겹침 영역 = L 의
오른쪽 / R 의 왼쪽이라는 match_overlap 규약과 동일 기준).

## 제외

- `20250823_GoPro5` — 단일 카메라 (짝 없음, 1영상 33챕터 254GB)
- `20250427_GoPro_Morning`, `20251026_GoPro`, `20251102_GoPro` — 편집
  산출물 모음 (.mov/.ptvb/합성 mp4), GoPro 원본 아님
- `20241013_AX700`, `20241020_AX700_*` 등 — 타 카메라

## 실행 예

```bash
cd /mnt/d/projects/PyStitch360
python main.py --headless /mnt/f/Pictures/20250427_GoPro_Left /mnt/f/Pictures/20250427_GoPro_Right
# 출력: /mnt/f/Pictures/20250427_GoPro/ (이름 공통부분 자동)
```

7번은 공백 경로 주의. **L = "GoPro5 2"(GOPR0388)** 순서 필수 —
match_overlap 은 겹침을 L 오른쪽/R 왼쪽으로 가정하므로 순서가 바뀌면
스티칭이 깨진다. 2026-07-25 실측 확인: L="GoPro5 2" 순서가 매칭점
1560개, 반대는 149개 (10배). match_overlap 이 R 입력폴더(GoPro5)와
동명이라 --out 을 명시:

```bash
python main.py --headless "/mnt/f/Pictures/20251026_GoPro5 2" \
    "/mnt/f/Pictures/20251026_GoPro5" \
    --out "/mnt/f/Pictures/20251026_GoPro5_pano" --auto-el
```
