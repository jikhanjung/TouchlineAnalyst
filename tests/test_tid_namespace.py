"""tid 네임스페이스 — 원경/수동 분리와 범위 소진 가드 (devlog 103).

원경 `_FarTracker` 가 상한 없이 증가해 수동 검출 영역(900001+)을 침범했다
(0392 실측: 800001~900000 가득 + 900001~929952 연속). GUI 는 그 영역을
수동으로 보고 품질 필터에서 면제하므로 파편 3만 개가 "선수"로 둔갑했다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import (  # noqa: E402
    EXTRA_TID_BASE, FAR_TID_BASE, FAR_TID_MAX, _FarTracker,
    extra_tid_number, is_extra_tid, is_far_tid,
)


def test_ranges_do_not_overlap():
    assert FAR_TID_MAX < EXTRA_TID_BASE


@pytest.mark.parametrize("tid", [FAR_TID_BASE, FAR_TID_BASE + 1, FAR_TID_MAX,
                                 800_001, 900_000])           # 신·구 사이드카
def test_far_tids_are_far_and_never_extra(tid):
    """**핵심 회귀 방지** — 원경 tid 가 수동으로 판별되면 편집이 오염된다."""
    assert is_far_tid(tid)
    assert not is_extra_tid(tid)


@pytest.mark.parametrize("tid", [EXTRA_TID_BASE, EXTRA_TID_BASE + 5,
                                 900_001, 999_999])           # 신·구 사이드카
def test_extra_tids_are_extra_and_never_far(tid):
    assert is_extra_tid(tid)
    assert not is_far_tid(tid)


@pytest.mark.parametrize("tid", [1, 7, 316_425, 799_999])
def test_bytetrack_ids_are_neither(tid):
    assert not is_far_tid(tid) and not is_extra_tid(tid)


def test_extra_tid_number_for_both_conventions():
    assert extra_tid_number(EXTRA_TID_BASE) == 1
    assert extra_tid_number(EXTRA_TID_BASE + 41) == 42
    assert extra_tid_number(900_001) == 1        # 구 사이드카
    assert extra_tid_number(900_042) == 42


def test_far_tracker_allocates_from_base():
    t = _FarTracker(radius=10.0, max_miss=5)
    tids = t.update([[0.0, 0.0, 0.9, 4, 8], [100.0, 100.0, 0.8, 4, 8]], 0)
    assert tids == [FAR_TID_BASE, FAR_TID_BASE + 1]
    assert all(is_far_tid(x) for x in tids)


def test_far_tracker_reuses_track_within_radius():
    t = _FarTracker(radius=10.0, max_miss=5)
    a = t.update([[0.0, 0.0, 0.9, 4, 8]], 0)[0]
    b = t.update([[3.0, 3.0, 0.9, 4, 8]], 1)[0]      # 반경 안 → 같은 트랙
    c = t.update([[300.0, 300.0, 0.9, 4, 8]], 2)[0]  # 반경 밖 → 새 트랙
    assert a == b and c != a


def test_exhaustion_yields_none_and_never_crosses_into_extra():
    """범위가 차면 검출을 버린다 — 조용히 수동 영역으로 넘치면 안 된다."""
    logs = []
    t = _FarTracker(radius=1.0, max_miss=0, log=logs.append)
    t.next_id = FAR_TID_MAX                      # 마지막 한 칸만 남긴 상태
    first = t.update([[0.0, 0.0, 0.9, 4, 8]], 0)
    assert first == [FAR_TID_MAX]
    # 이후로는 계속 None (반경 1px·max_miss 0 이라 매번 새 트랙을 원한다)
    for si in range(1, 4):
        got = t.update([[si * 500.0, 0.0, 0.9, 4, 8]], si)
        assert got == [None], got
    assert t.exhausted == 3
    assert t.next_id <= EXTRA_TID_BASE
    assert logs and "소진" in logs[0]


def test_exhaustion_logs_once():
    logs = []
    t = _FarTracker(radius=1.0, max_miss=0, log=logs.append)
    t.next_id = FAR_TID_MAX + 1
    for si in range(5):
        t.update([[si * 500.0, 0.0, 0.9, 4, 8]], si)
    assert len(logs) == 1, "매 프레임 로그를 쏟으면 진행 로그를 덮는다"
