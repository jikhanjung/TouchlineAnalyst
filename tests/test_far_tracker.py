"""_FarTracker (원경 타일 검출 미니 트래커) — devlog 077/078."""
from pystitch.core.ptz import _FarTracker


def test_stable_ids_for_slow_motion():
    """샘플마다 조금씩 움직이는 두 사람 — tid 가 유지돼야 한다."""
    trk = _FarTracker(radius=50, max_miss=5)
    a, b = (100.0, 600.0), (400.0, 610.0)
    ids0 = trk.update([[a[0], a[1], 0.9, 30, 60],
                       [b[0], b[1], 0.8, 30, 60]], 0)
    for si in range(1, 20):
        a = (a[0] + 8, a[1] + 1)          # ~8px/샘플 이동
        b = (b[0] - 6, b[1])
        ids = trk.update([[a[0], a[1], 0.9, 30, 60],
                          [b[0], b[1], 0.8, 30, 60]], si)
        assert ids == ids0                # 동일 tid 유지


def test_new_id_when_far():
    trk = _FarTracker(radius=50, max_miss=5)
    (i0,) = trk.update([[100, 600, 0.9, 30, 60]], 0)
    (i1,) = trk.update([[900, 600, 0.9, 30, 60]], 1)   # 반경 밖 → 새 tid
    assert i1 != i0


def test_track_drop_after_miss():
    trk = _FarTracker(radius=50, max_miss=3)
    (i0,) = trk.update([[100, 600, 0.9, 30, 60]], 0)
    # 4샘플 동안 사라졌다가 같은 자리 재등장 — 트랙 종료 후 새 tid
    (i1,) = trk.update([[100, 600, 0.9, 30, 60]], 5)
    assert i1 != i0


def test_greedy_no_double_assign():
    """가까운 두 검출이 한 트랙을 나눠 갖지 않는다 (conf 우선 그리디)."""
    trk = _FarTracker(radius=50, max_miss=5)
    trk.update([[100, 600, 0.9, 30, 60]], 0)
    ids = trk.update([[102, 600, 0.9, 30, 60],
                      [110, 604, 0.5, 30, 60]], 1)
    assert len(set(ids)) == 2             # 하나는 기존, 하나는 새 tid


def test_ids_in_far_range():
    trk = _FarTracker(radius=50, max_miss=5)
    (i0,) = trk.update([[100, 600, 0.9, 30, 60]], 0)
    assert 800001 <= i0 < 900000          # ByteTrack/수동(900000+)과 분리
