"""headless --stitch-only — 스티칭까지만 하고 후속 단계를 건너뛴다.

파노라마만 먼저 뽑아두고 분석은 나중에 이어붙이는 용도. 나중에 플래그를
빼고 같은 명령을 돌리면 스티칭은 산출물 게이팅에 걸려 건너뛰어야 한다.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch import headless  # noqa: E402


def _args(**kw):
    base = dict(force=False, reanalyze=False, far_augment=False, no_ocr=False,
                venue=None, ax700=[], stitch_only=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def spy(monkeypatch, tmp_path):
    """무거운 단계를 전부 기록만 하는 스텁으로 교체."""
    calls = []

    def stub(name, *, makes_pano=False):
        def f(*a, **k):
            calls.append(name)
            if makes_pano:
                # _stitch(pair, pano, ...) — 실제로 파일을 남겨야 게이팅 검증이 된다
                Path(a[1]).write_bytes(b"\0" * (2 << 20))
            return None
        return f

    monkeypatch.setattr(headless, "_stitch", stub("stitch", makes_pano=True))
    monkeypatch.setattr(headless, "_analyze", stub("analyze"))
    monkeypatch.setattr(headless, "_whistle", stub("whistle"))
    monkeypatch.setattr(headless, "_ocr", stub("ocr"))
    monkeypatch.setattr(headless, "_far_augment", stub("far_augment"))
    monkeypatch.setattr(headless, "_build_match", stub("match"))
    monkeypatch.setattr(headless, "load_events_doc", lambda p: {})
    return calls


@pytest.fixture
def pair(tmp_path):
    left = tmp_path / "L"
    left.mkdir()
    chain = [left / "GOPR0001.MP4"]
    chain[0].write_bytes(b"x")
    return (chain, [tmp_path / "R" / "GOPR0397.MP4"], 0.01)


def test_stitch_only_skips_analysis_and_downstream(tmp_path, spy, pair):
    out = tmp_path / "out"
    out.mkdir()
    pano = headless.process_pair(pair, out, None, "lens", _args(stitch_only=True))
    assert spy == ["stitch"]
    assert pano.exists()


def test_without_flag_runs_full_pipeline(tmp_path, spy, pair):
    out = tmp_path / "out"
    out.mkdir()
    headless.process_pair(pair, out, None, "lens", _args())
    assert "analyze" in spy and "ocr" in spy


def test_rerun_without_flag_resumes_from_analyze(tmp_path, spy, pair):
    """스티칭만 해둔 뒤 플래그를 빼고 재실행 — 스티칭은 건너뛰고 분석부터."""
    out = tmp_path / "out"
    out.mkdir()
    headless.process_pair(pair, out, None, "lens", _args(stitch_only=True))
    spy.clear()
    headless.process_pair(pair, out, None, "lens", _args())
    assert "stitch" not in spy          # 이미 있는 파노라마를 다시 만들지 않는다
    assert "analyze" in spy


def test_stitch_only_still_seeds_venue(tmp_path, spy, pair, monkeypatch):
    """경기장 시딩은 스티칭 산출물에 딸린 것 — 생략 대상이 아니다."""
    seeded = []
    monkeypatch.setattr(headless, "_seed_venue",
                        lambda p, v, force=False: seeded.append(v))
    out = tmp_path / "out"
    out.mkdir()
    headless.process_pair(pair, out, None, "lens",
                          _args(stitch_only=True, venue="양재근린공원"))
    assert seeded == ["양재근린공원"]
    assert spy == ["stitch"]
