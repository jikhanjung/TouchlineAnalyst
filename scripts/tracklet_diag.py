"""트랙릿 파편화 진단 — 재분석 없이 `.analysis.cache.json` 만으로 (TODOs #1).

무엇이 문제인지 먼저 정량화한다: 수명 분포, 단발 비율, 그리고 **끊긴 자리
바로 옆에서 새 트랙이 시작하는가** — 이게 매칭 반경/만료를 키우면 회수될
파편인지를 가린다.

tid 네임스페이스는 `pystitch.core.ptz` 의 판별식을 쓴다 (구 사이드카 호환).

사용: python scripts/tracklet_diag.py <pano>.analysis.cache.json [--link]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(sys.argv[1])
DO_LINK = "--link" in sys.argv

d = json.loads(CACHE.read_text(encoding="utf-8"))
spans, segs, foot = d["spans"], d["segs"], d["foot_med"]

# 같은 폴더의 analysis.json 헤더에서 메타 (fps, detect_every, pano_w)
meta = {}
apath = CACHE.with_name(CACHE.name.replace(".cache.json", ".json"))
if apath.exists():
    with apath.open("rb") as f:
        head = f.read(400).decode("utf-8", "replace")
    for k in ("fps", "detect_every", "pano_w", "total_frames"):
        i = head.find(f'"{k}":')
        if i >= 0:
            meta[k] = float(head[i + len(k) + 3:].split(",")[0].strip(' "}'))
FPS = meta.get("fps", 29.97)
DE = int(meta.get("detect_every", 3))
PANO_W = meta.get("pano_w", 6022)
SEC = DE / FPS                       # 샘플 1개 = 몇 초
print(f"{CACHE.name}  fps {FPS:.2f}  detect_every {DE}  pano_w {PANO_W:.0f}  "
      f"→ 샘플 간격 {SEC*1000:.0f}ms")


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pystitch.core.ptz import is_extra_tid, is_far_tid  # noqa: E402


def ns(tid):
    # 판별은 코어 규약을 그대로 쓴다 (구 사이드카 호환 포함)
    return ("far" if is_far_tid(tid)
            else "manual/extra" if is_extra_tid(tid) else "byte(소형)")


groups = defaultdict(list)
for tid, (a, b, n) in spans.items():
    groups[ns(tid)].append((int(tid), a, b, n))

print(f"\n총 트랙릿 {len(spans):,}개, 세그먼트 "
      f"{sum(len(v) for v in segs.values()):,}개")
print(f"{'네임스페이스':16s} {'개수':>9s} {'비율':>6s} {'관측 중앙값':>10s} "
      f"{'수명 중앙값':>10s} {'단발':>8s}")
for g, rows in sorted(groups.items()):
    n_obs = np.array([r[3] for r in rows])
    life = np.array([r[2] - r[1] + 1 for r in rows])
    single = (n_obs == 1).mean()
    print(f"{g:16s} {len(rows):9,d} {len(rows)/len(spans)*100:5.1f}% "
          f"{np.median(n_obs)*DE:8.0f}프레임 {np.median(life)*SEC:9.2f}초 "
          f"{single*100:7.1f}%")

far = groups.get("far", [])
if far:
    n_obs = np.array([r[3] for r in far])
    life = np.array([r[2] - r[1] + 1 for r in far])
    nseg = np.array([len(segs[str(r[0])]) for r in far])
    print(f"\n원경 트랙릿 {len(far):,}개 상세")
    for p in (10, 25, 50, 75, 90, 99):
        print(f"  {p:2d}분위: 관측 {np.percentile(n_obs, p)*DE:6.0f}프레임 "
              f"({np.percentile(n_obs, p)*DE/FPS:5.2f}초)  "
              f"수명 {np.percentile(life, p)*SEC:5.2f}초  "
              f"세그먼트 {np.percentile(nseg, p):.0f}개")
    print(f"  총 관측 {n_obs.sum():,} 샘플 = 트랙릿당 평균 "
          f"{n_obs.mean():.1f} (={n_obs.mean()*DE/FPS:.2f}초)")

if not DO_LINK:
    print("\n(--link 를 주면 '끊긴 자리 옆에서 새 트랙이 시작하는가' 분석)")
    sys.exit()

# ---------------------------------------------------------- 연결 후보 분석
print("\n=== 끊긴 직후 근처에서 시작하는 원경 트랙 (회수 가능 파편) ===")
starts = defaultdict(list)
for tid, a, b, n in far:
    p = foot.get(str(tid))
    if p:
        starts[a].append((tid, p[0], p[1]))
ends = [(tid, b, foot[str(tid)][0], foot[str(tid)][1])
        for tid, a, b, n in far if str(tid) in foot]
print(f"종료 트랙 {len(ends):,}개 대상")

CUR_R = PANO_W * 0.008
for K in (2, 5, 10, 20, 40):
    for R in (CUR_R, CUR_R * 2, CUR_R * 4):
        hit = 0
        for tid, b, x, y in ends:
            found = False
            for s in range(b + 1, b + 1 + K):
                for tid2, x2, y2 in starts.get(s, ()):
                    if (x - x2) ** 2 + (y - y2) ** 2 < R * R:
                        found = True
                        break
                if found:
                    break
            hit += found
        print(f"  창 {K*SEC:4.1f}초 ({K:2d}샘플)  반경 {R:5.0f}px "
              f"({R/PANO_W*100:.1f}% 폭)  → 후보 있음 "
              f"{hit/len(ends)*100:5.1f}%")

# ------------------------------------------- 간격 vs 거리 (반경 설계 근거)
print("\n=== 끊긴 뒤 재시작까지의 (간격, 거리) — 가장 가까운 후보 ===")
print("고정 반경이 맞는지, 간격에 비례해 넓혀야 하는지 가린다.")
MAXK = 60
pairs = []
for tid, b, x, y in ends:
    best = None
    for s in range(b + 1, b + 1 + MAXK):
        for tid2, x2, y2 in starts.get(s, ()):
            dd = (x - x2) ** 2 + (y - y2) ** 2
            if best is None or dd < best[1]:
                best = (s - b, dd)
    if best is not None:
        pairs.append((best[0], best[1] ** 0.5))

if pairs:
    gaps = np.array([p[0] for p in pairs])
    dist = np.array([p[1] for p in pairs])
    print(f"후보를 찾은 종료 트랙 {len(pairs):,}/{len(ends):,} "
          f"({len(pairs)/len(ends)*100:.1f}%), 창 {MAXK*SEC:.1f}초 내")
    print(f"{'간격(초)':>9s} {'개수':>8s} {'거리 25%':>9s} {'중앙':>8s} "
          f"{'75%':>8s}")
    for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 60)):
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum() < 50:
            continue
        d = dist[m]
        print(f"{lo*SEC:4.1f}~{hi*SEC:4.1f} {m.sum():8,d} "
              f"{np.percentile(d,25):8.0f}px {np.median(d):7.0f}px "
              f"{np.percentile(d,75):7.0f}px")
    # 샘플당 이동 속도 추정 (중앙값 기울기)
    med = [(g, np.median(dist[gaps == g])) for g in range(1, 21)
           if (gaps == g).sum() >= 50]
    if len(med) >= 3:
        gg = np.array([m[0] for m in med], float)
        dd = np.array([m[1] for m in med], float)
        a, b_ = np.polyfit(gg, dd, 1)
        print(f"\n중앙 거리 ≈ {a:.1f}px×간격 + {b_:.0f}px  "
              f"→ 샘플당 {a:.1f}px ({a/SEC:.0f}px/초)")
        print(f"현행 고정 반경 {CUR_R:.0f}px 는 간격 "
              f"{max(0,(CUR_R-b_)/max(a,1e-9)):.1f}샘플"
              f"({max(0,(CUR_R-b_)/max(a,1e-9))*SEC:.1f}초)까지만 커버")
