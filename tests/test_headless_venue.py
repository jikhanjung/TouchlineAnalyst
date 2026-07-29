"""headless --venue 사이드카 시딩 (.ptz.json 은 사용자 편집 영역 — 덮지 않는다)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch import headless  # noqa: E402

FAKE = {"테스트구장": {"length": 92.0, "width": 62.0, "circle_r": 9.0},
        "다른구장": {"length": 105.0, "width": 68.0}}


@pytest.fixture
def venues(monkeypatch):
    monkeypatch.setattr(headless, "load_venues", lambda: dict(FAKE))


def test_seeds_venue_into_fresh_sidecar(tmp_path, venues):
    pano = tmp_path / "pano_0001.mp4"
    headless._seed_venue(pano, "테스트구장")
    doc = json.loads(pano.with_suffix(".ptz.json").read_text(encoding="utf-8"))
    assert doc["venue"] == "테스트구장"
    assert doc["field_size"] == [92.0, 62.0]
    assert doc["field_circle_r"] == 9.0
    # GUI 의 venue 적용과 같아야 한다 — 규격을 명시했으니 자동 추정 끔
    assert doc["field_circle_auto"] is False


def test_circle_r_falls_back_when_preset_omits_it(tmp_path, venues):
    pano = tmp_path / "pano_0002.mp4"
    headless._seed_venue(pano, "다른구장")
    doc = json.loads(pano.with_suffix(".ptz.json").read_text(encoding="utf-8"))
    assert doc["field_circle_r"] == pytest.approx(9.15)


def test_preserves_other_user_edits(tmp_path, venues):
    pano = tmp_path / "pano_0003.mp4"
    sp = pano.with_suffix(".ptz.json")
    sp.write_text(json.dumps({"player_nums": {"7": 15}, "roles": {"3": 1}}),
                  encoding="utf-8")
    headless._seed_venue(pano, "테스트구장")
    doc = json.loads(sp.read_text(encoding="utf-8"))
    assert doc["player_nums"] == {"7": 15}
    assert doc["roles"] == {"3": 1}
    assert doc["venue"] == "테스트구장"


def test_does_not_overwrite_a_different_venue(tmp_path, venues):
    """사용자가 GUI 에서 고른 경기장을 배치가 갈아엎으면 안 된다."""
    pano = tmp_path / "pano_0004.mp4"
    sp = pano.with_suffix(".ptz.json")
    sp.write_text(json.dumps({"venue": "다른구장", "field_size": [105.0, 68.0]}),
                  encoding="utf-8")
    headless._seed_venue(pano, "테스트구장")
    doc = json.loads(sp.read_text(encoding="utf-8"))
    assert doc["venue"] == "다른구장"
    assert doc["field_size"] == [105.0, 68.0]
    # --force 면 덮는다
    headless._seed_venue(pano, "테스트구장", force=True)
    doc = json.loads(sp.read_text(encoding="utf-8"))
    assert doc["venue"] == "테스트구장"
    assert doc["field_size"] == [92.0, 62.0]


def test_unknown_venue_writes_nothing(tmp_path, venues):
    pano = tmp_path / "pano_0005.mp4"
    headless._seed_venue(pano, "없는구장")
    assert not pano.with_suffix(".ptz.json").exists()


def test_no_temp_file_left_behind(tmp_path, venues):
    pano = tmp_path / "pano_0006.mp4"
    headless._seed_venue(pano, "테스트구장")
    assert [p.name for p in tmp_path.iterdir()] == ["pano_0006.ptz.json"]


def test_shipped_presets_have_required_fields():
    """presets/venues.json 스키마 고정 — 빠진 키는 GUI/헤드리스 양쪽을 깬다."""
    venues = headless.load_venues()
    assert venues, "presets/venues.json 이 비었거나 못 읽음"
    for name, v in venues.items():
        assert v["length"] > 0 and v["width"] > 0, name
        assert 0 < v.get("circle_r", 9.15) < 15, name
