"""짧은 원경 트랙 제외 — 요약·팀분류 단계에서만 (devlog 104).

0392 실측: 원경 트랙의 절반 이상이 3샘플 이하인데 그것들이 나르는 관측은
전체의 7.3% 뿐이다. 분석 원본은 건드리지 않으므로 문턱은 언제든 바꿀 수
있어야 하고, **근경 트랙은 길이와 무관하게 남아야 한다**.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import (  # noqa: E402
    FAR_TID_BASE, analysis_summary, drop_short_far, far_min_obs,
)

NEAR_TID = 42
FAR_LONG = FAR_TID_BASE
FAR_SHORT = FAR_TID_BASE + 1


def _analysis(n_long=20, n_short=3, n_near=2, fps=30.0, de=3):
    """합성 분석: 긴 원경 1 + 짧은 원경 1 + 짧은 근경 1."""
    n = max(n_long, n_short, n_near)
    players = []
    for si in range(n):
        row = []
        if si < n_long:
            row.append([100.0, 200.0, 10.0, 20.0, FAR_LONG, 60.0, 40.0, 120.0])
        if si < n_short:
            row.append([300.0, 200.0, 10.0, 20.0, FAR_SHORT, 60.0, 40.0, 120.0])
        if si < n_near:
            row.append([500.0, 900.0, 40.0, 90.0, NEAR_TID, 20.0, 80.0, 200.0])
        players.append(row)
    return {"frames": [i * de for i in range(n)], "fps": fps,
            "detect_every": de, "players": players}


def test_threshold_scales_with_fps_and_detect_every():
    assert far_min_obs(_analysis(fps=30.0, de=3)) == 10      # 1.0초
    assert far_min_obs(_analysis(fps=30.0, de=1)) == 30
    assert far_min_obs(_analysis(fps=60.0, de=3)) == 20


def test_env_override_and_disable(monkeypatch):
    a = _analysis()
    monkeypatch.setenv("PYSTITCH_FAR_MIN_SEC", "2.0")
    assert far_min_obs(a) == 20
    monkeypatch.setenv("PYSTITCH_FAR_MIN_SEC", "0")
    assert far_min_obs(a) == 0                                # 필터 끔
    monkeypatch.setenv("PYSTITCH_FAR_MIN_SEC", "말도안됨")
    assert far_min_obs(a) == 10                               # 파싱 실패 → 기본


def test_only_far_tids_are_dropped():
    assert drop_short_far(FAR_SHORT, 3, 10)
    assert not drop_short_far(FAR_LONG, 20, 10)
    assert not drop_short_far(NEAR_TID, 1, 10)     # 근경은 길이 무관 보존
    assert not drop_short_far(FAR_SHORT, 3, 0)     # 문턱 0 = 끔


def test_summary_drops_short_far_keeps_near(tmp_path):
    a = _analysis()
    ap = tmp_path / "x.analysis.json"
    ap.write_text(json.dumps(a))
    s = analysis_summary(ap, a, log=lambda *_: None)
    assert FAR_LONG in s["spans"]
    assert FAR_SHORT not in s["spans"], "짧은 원경이 남았다"
    assert NEAR_TID in s["spans"], "근경은 3샘플이어도 남아야 한다"


def test_summary_dicts_stay_consistent(tmp_path):
    """spans 만 거르고 colors/foot_med/segs 를 안 거르면 KeyError 가 난다."""
    a = _analysis()
    ap = tmp_path / "y.analysis.json"
    ap.write_text(json.dumps(a))
    s = analysis_summary(ap, a, log=lambda *_: None)
    keys = set(s["spans"])
    for name in ("segs", "colors", "foot_med"):
        assert set(s[name]) == keys, f"{name} 가 spans 와 어긋남"


def test_cache_key_includes_threshold(tmp_path, monkeypatch):
    """문턱을 바꾸면 캐시가 무효화돼야 한다 — 아니면 옛 결과가 살아난다."""
    a = _analysis()
    ap = tmp_path / "z.analysis.json"
    ap.write_text(json.dumps(a))
    analysis_summary(ap, a, log=lambda *_: None)          # 캐시 생성
    monkeypatch.setenv("PYSTITCH_FAR_MIN_SEC", "0")       # 필터 끄고 재요약
    s = analysis_summary(ap, a, log=lambda *_: None)
    assert FAR_SHORT in s["spans"], "옛 캐시가 그대로 재사용됐다"


def test_team_features_excludes_short_far():
    from pystitch.core.ptz import team_features
    tf = team_features(_analysis())
    assert FAR_LONG in tf["ids"] and NEAR_TID in tf["ids"]
    assert FAR_SHORT not in tf["ids"], "유령이 팀 색 군집을 흔든다"
