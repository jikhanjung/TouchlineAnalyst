"""4번 탭: 가상 PTZ — 자동 공 검출을 기본으로 깔고 클릭으로 키프레임 교정.

워크플로우: 완성 파노라마 열기 → 자동 분석(캐시) → 타임라인을 훑으며
공이 아닌 곳을 보는 구간에서 화면 클릭(=키프레임) → PTZ 내보내기.
키프레임은 <파노라마>.ptz_keyframes.json 에 자동 저장된다.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEvent, QRect, QSettings, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt6.QtWidgets import QApplication, QCheckBox
from PyQt6.QtGui import (
    QColor, QCursor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollBar, QSlider, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..core.encoders import available_encoders
from ..core.field import (
    CENTER_CIRCLE_R, LANDMARKS, detect_sideline_points, field_outline,
    field_to_pano, fit_field_calibration, landmark_positions, pano_to_field,
)
from ..core.ptz import (
    accept_ball_tracks, analyze_video, build_plan, build_radar_data,
    classify_teams, draw_radar_panel, gapfill_analysis, gapfill_targets,
    ground_positions, link_ball_tracks, propagate_seed, ptz_available,
    analysis_summary, render_plan, same_spot_spans, tracklet_colors,
)
from .widgets import FramePane


# BGR — 인덱스 = 역할 번호 (core.ptz ROLE_*): 팀1, 팀2, 기타, 팀1 GK, 팀2 GK, 심판
TEAM_COLORS = [(60, 60, 230), (230, 140, 40), (160, 160, 160),
               (180, 105, 255), (230, 230, 0), (60, 200, 230),
               (250, 150, 60)]
ROLE_NAMES = ["팀1", "팀2", "기타", "팀1 GK", "팀2 GK", "주심", "선심"]
ROLE_TAGS = {3: "GK1", 4: "GK2", 5: "REF", 6: "AR"}
# 라인 point-to-line 캘리브(rotcam)용 — 값=(_marking_lines 패밀리 인덱스,
# 표시명, 표시색 BGR). 터치라인 다점을 소실점 구속으로 f 를 안정화한다.
LINE_FAMILIES = {
    "touch_near": (0, "가까운 터치라인", (0, 220, 220)),
    "touch_far": (1, "먼 터치라인", (0, 160, 255)),
    "goal_l": (2, "왼쪽 골라인", (255, 200, 0)),
    "goal_r": (3, "오른쪽 골라인", (255, 120, 0)),
    "halfway": (4, "하프라인", (200, 100, 255)),
}
# 미리보기 오버레이용 랜드마크 약칭 (cv2 폰트는 한글 불가)
LANDMARK_TAGS = {"corner_far_l": "FL", "corner_far_r": "FR",
                 "corner_near_l": "NL", "corner_near_r": "NR",
                 "half_far": "HF", "half_near": "HN",
                 "circle_far": "CF", "circle_near": "CN",
                 "sideline_near_l": "SL", "sideline_near_r": "SR",
                 "pen_l_far": "PLF", "pen_l_near": "PLN",
                 "pen_r_far": "PRF", "pen_r_near": "PRN",
                 "pen_l_box_far": "BLF", "pen_l_box_near": "BLN",
                 "pen_r_box_far": "BRF", "pen_r_box_near": "BRN",
                 "center_near": "CM",
                 "circle_l": "CL", "circle_r": "CR"}


def _dashed_circle(img, center, radius, color, thickness=2, dash_deg=18):
    """점선 원 — 무시된 공(=공 아님) 테두리용. dash_deg 간격으로 호를
    번갈아 그린다 (OpenCV 는 점선 원을 직접 지원하지 않음)."""
    radius = max(1, int(radius))
    for a in range(0, 360, dash_deg * 2):
        cv2.ellipse(img, center, (radius, radius), 0, a, a + dash_deg,
                    color, thickness, cv2.LINE_AA)


def _ffmpeg_frame(path, t, w, h, timeout=45.0):
    """ffmpeg 빠른 시크로 프레임 하나를 원본 해상도(BGR)로 디코드.

    OpenCV `set(POS_FRAMES)` 는 큰 프레임·긴 GOP(파노라마는 6540px,
    ~8초 GOP)에서 키프레임부터 순차 디코드라 40분 지점이 수십 초 걸린다.
    ffmpeg 는 `-ss` 를 입력 앞에 둬 키프레임으로 빠르게 시크한 뒤 정확
    위치까지만 멀티스레드 디코드 — 같은 프레임이 ~2~3초로 끝난다."""
    import subprocess

    from ..core.encoders import ffmpeg_bin
    # ffmpeg_bin() 리졸버 사용 필수 — 바 "ffmpeg" 는 winget 설치 후 앱
    # PATH 에 안 잡혀(ffmpeg_bin 독스트링 참고) FileNotFoundError 로
    # 조용히 실패했다. 그러면 원본 승격이 영영 안 돼 SCRUB 만 남는다.
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t):.6f}",
           "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    need = w * h * 3
    if len(p.stdout) < need:
        return None
    return np.frombuffer(p.stdout[:need], np.uint8).reshape(h, w, 3)


def _boost_bgr(bgr, s_gain=1.35, v_gain=1.55, v_floor=190):
    """그림자 보정: 유니폼 대표색을 실제 옷 색에 가깝게 밝고 진하게.

    상반신 샘플은 그림자가 섞여 실제 유니폼보다 어둡게 나온다 —
    표시용으로만 명도(V)·채도(S)를 끌어올린다.
    """
    h, s, v = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0, 0]
    s = min(int(s * s_gain), 255)
    v = min(max(int(v * v_gain), v_floor), 255)
    b, g, r = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(b), int(g), int(r))


class TimelineView(QWidget):
    """NLE 스타일 멀티트랙 타임라인.

    레인: 크롭/키프레임, 공(수락=초록·무시=빨강·승격=자홍), 팀1, 팀2, 기타.
    선수 트랙릿·공 트랙·키프레임이 가로 바/박스로 그려지고 클릭으로
    선택(pick 시그널)할 수 있다. Ctrl+휠 = 커서 기준 확대축소, 휠 = 팬,
    빈 곳 드래그 = 팬, 빈 곳 클릭 = 시크. 빨간 세로선 = 현재 위치.
    """

    seek = pyqtSignal(int)
    pick = pyqtSignal(str, int, int)  # (종류, 키, 클릭 프레임)
    angle_pick = pyqtSignal(int)     # 앵글 레인 클릭 (alt 인덱스) — 카메라 전환
    range_menu = pyqtSignal(int, object)   # 우클릭: (프레임, 전역 좌표)
    view_changed = pyqtSignal(float, float, int)   # (t0, 보이는 폭, total)

    RULER = 16
    LANE_H = 20                  # 기본 레인 높이 (개별 조절 가능)
    GUTTER = 64
    LANES = ["크롭/KF", "공", "뜬 공", "팀1", "팀2", "기타", "호각", "이벤트",
             "하이라이트", "랜드마크"]
    WHISTLE_MIN_DB = 20.0        # 이 피크 이상만 '확실한 호각'으로 표시

    def __init__(self):
        super().__init__()
        self.lanes = list(self.LANES)     # 인스턴스 사본 (팀 이름 반영)
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_timeline_lanes", None)
        try:
            self.lane_h = [max(10, min(240, int(v))) for v in saved]
            assert len(self.lane_h) <= len(self.lanes)
            # 레인이 추가된 구버전 저장값은 기본 높이로 채움
            self.lane_h += [self.LANE_H] * (len(self.lanes) - len(self.lane_h))
        except Exception:  # noqa: BLE001
            self.lane_h = [self.LANE_H] * len(self.lanes)
        # 접힌 레인 (우클릭 메뉴): 서브행 없이 얇은 한 줄 —
        # _apply_height 가 참조하므로 그 전에 초기화
        try:
            self.collapsed = {int(v) for v in
                              QSettings("PyStitch360", "PyStitch360")
                              .value("ptz_timeline_collapsed", []) or []}
        except (TypeError, ValueError):
            self.collapsed = set()
        self._apply_height()
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        # 정적 툴팁 없음 — 트랙릿 번호 툴팁(동적)과 겹쳐 두 개 뜨던 문제.
        # 레인 경계 조절 안내는 경계 위 호버에서만 동적으로 표시.
        self.total = 0
        self.fps = 30.0
        self.pos = 0
        self.spans: list = []
        self.ignores: list = []
        self.kfs: list = []
        self.promotes: list = []
        self._players: list = []     # [(tid, f0, f1, role, 서브행), ...]
        self._lane_rows: dict = {}   # {레인: 서브행 수} (동시 트랙릿 폭)
        self.t0 = 0.0                # 보이는 시작 프레임
        self.ppf = 0.0               # 픽셀/프레임 (0 = 전체 맞춤)
        self.selected = None         # (종류, 키)
        self.mark_in = None          # 내보내기 시작 프레임 (None=미지정)
        self.mark_out = None         # 내보내기 끝 프레임
        self._whistle = None         # (hop_s, 프로미넌스 배열, 이벤트)
        self.events = []             # [(frame, label, kind)] kind: auto|user
        self.landmarks = []          # [(frame, label, (b,g,r))] 랜드마크 찍은 프레임
        self.airborne = []           # [(f0, f1, apex_z)] 공중 구간
        self.highlights = []         # [(f0, f1, state, label)] 하이라이트 구간
        self.pauses = []             # [(f0, f1)] 경기 중단 (시계 정지) 구간
        # 멀티캠 앵글 레인 (P07): [{label, span:(f0,f1), whistles:[(t0,t1,db)]}]
        # whistles 는 primary 초 단위 (시계 모델 변환 후)
        self.angles = []
        self.possession = []         # [(f0, f1, team)] 소유 리본 (P08)
        self.role_palette = {}       # {역할: BGR} 실측/지정 팀 색 (바 색)
        self._press = None
        self._resize = None          # 레인 경계 드래그 상태

    def set_lane_names(self, team1, team2):
        self.lanes[3], self.lanes[4] = team1, team2
        self.update()

    def set_events(self, events):
        self.events = list(events)
        self.update()

    def set_landmarks(self, marks):
        """랜드마크 찍은 프레임 마커 [(frame, label, (b,g,r))] — 어느
        프레임에 무엇을 찍었는지 한눈에, 클릭하면 그 프레임으로 이동."""
        self.landmarks = list(marks or [])
        self.update()

    def set_airborne(self, segs):
        """공중 구간 [(f0, f1, apex_z)] — '뜬 공' 레인."""
        self.airborne = list(segs)
        self.update()

    def set_highlights(self, hls):
        """하이라이트 구간 [(f0, f1, state, label)] — 이벤트 레인 바."""
        self.highlights = list(hls)
        self.update()

    def set_pauses(self, pauses):
        """경기 중단 구간 [(f0, f1)] — 회색 세로 밴드 (시계 정지)."""
        self.pauses = list(pauses)
        self.update()

    def set_possession(self, spans):
        """소유 리본 [(f0, f1, team)] — 눈금자 바로 아래 팀 색 밴드 (P08)."""
        self.possession = list(spans or [])
        self.update()

    def set_angles(self, angles):
        """멀티캠 앵글 레인 구성 — 카메라별 레인이 기본 레인 뒤에 붙는다.

        각 레인: 커버리지 밴드 + 그 카메라의 호각 마커 (primary 시간축
        변환) — primary 호각 레인과의 정렬이 동기화 육안 검증이 된다.
        클릭 = 시크 + 해당 카메라로 전환 (angle_pick).
        """
        self.angles = list(angles or [])
        base_n = len(self.LANES)
        self.lanes = self.lanes[:base_n] + \
            [a["label"] for a in self.angles]
        self.lane_h = self.lane_h[:base_n] + \
            [self.LANE_H] * len(self.angles)
        self._apply_height()
        self.update()

    def set_role_palette(self, palette):
        """역할별 표시 색 {역할: BGR} — 트랙릿 바를 실제 팀 색과 일치."""
        if palette != self.role_palette:
            self.role_palette = dict(palette)
            self.update()

    def _emit_view(self):
        vis = (self.width() - self.GUTTER) / max(self._eff_ppf(), 1e-9)
        self.view_changed.emit(float(self.t0), float(vis), int(self.total))

    def set_view_start(self, f):
        """외부 스크롤바 → 보이는 시작 프레임 이동."""
        if abs(float(f) - self.t0) >= 1.0:
            self.t0 = float(f)
            self._clamp_view()
            self.update()

    def set_whistle(self, hop_s, prom, events):
        self._whistle = (float(hop_s), np.asarray(prom, dtype=np.float32),
                         list(events))
        self.update()

    def set_range(self, f0, f1):
        if (f0, f1) != (self.mark_in, self.mark_out):
            self.mark_in, self.mark_out = f0, f1
            self.update()

    def contextMenuEvent(self, ev):
        if self.total <= 1:
            return
        x, y = int(ev.pos().x()), int(ev.pos().y())
        f = int(min(max(self._f(max(x, self.GUTTER)), 0), self.total - 1))
        self._menu_lane = self._lane_at(y)
        # 우클릭 지점의 트랙릿(선수/공) — 타임라인 레인에서 직접 병합/해제
        self._menu_hit = self._hit(x, y) if y >= self.RULER else None
        self.range_menu.emit(f, ev.globalPos())

    # --------------------------------------------------------------- 데이터
    def set_data(self, total, spans, ignores, kfs, promotes=None):
        self.total = max(1, int(total))
        self.spans, self.ignores = list(spans), list(ignores)
        self.kfs = list(kfs)
        # 공 트랙 서브행 (선수 레인과 동일 그리디 간트) — 동시 트랙
        # (경기 공 + 옆 구장/워밍업 공)이 한 줄에 겹쳐 가려지던 문제
        ends: list = []
        self._ball_rows = []
        for i in sorted(range(len(self.spans)),
                        key=lambda j: self.spans[j][0]):
            f0, f1 = self.spans[i]
            for si, e in enumerate(ends):
                if e <= f0:
                    ends[si] = f1
                    break
            else:
                si = len(ends)
                ends.append(f1)
            while len(self._ball_rows) <= i:
                self._ball_rows.append(0)
            self._ball_rows[i] = si
        self._ball_nrows = max(1, len(ends))
        if promotes is not None:
            self.promotes = list(promotes)
        self._clamp_view()
        self.update()
        self._emit_view()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp_view()
        self._emit_view()

    def set_players(self, players, numbers=None, reps=None):
        """{tid: (f0, f1, role)} → 레인별 서브행 배치.

        번호가 부여된(=신원 확인된) 트랙릿은 **번호별 전용 행**으로 맨
        위에 — 같은 번호(같은 사람)의 조각들이 한 줄에 이어져 보이고,
        번호를 새로 부여하면 그 트랙릿이 위로 올라간다 (사용자 방향).
        무번호 트랙릿은 그 아래 그리디 간트. 상한 없음 — 행 두께는
        레인 높이/행 수, 레인 경계 드래그로 조절.
        numbers = {tid: 번호 문자열} (병합 대표 기준으로 풀어서 전달).
        """
        numbers = numbers or {}
        reps = reps or {}
        self._pnum = dict(numbers)
        self._prep = dict(reps)
        rep_of = lambda t: reps.get(t, t)
        by_lane: dict[int, list] = {}
        for tid, (f0, f1, role) in players.items():
            by_lane.setdefault(self._lane_of_role(role), []).append(
                (f0, f1, tid, role))
        out = []
        self._group_bands = []                 # [(lane, si, gf0, gf1, role)]
        self._lane_rows = {}
        for lane, items in by_lane.items():
            # 병합 그룹(rep) 단위 — 멤버는 한 행에 이어지고, 그룹 전체
            # 범위에 반투명 밴드(사이 갭 시각화). 번호 그룹은 위로.
            groups: dict = {}
            for f0, f1, tid, role in items:
                groups.setdefault(rep_of(tid), []).append((f0, f1, tid, role))
            def _numkey(rep):
                n = numbers.get(rep, "")
                return (not n, not n.isdigit() if n else True,
                        int(n) if n.isdigit() else 0)
            numbered = [g for g in groups if numbers.get(g)]
            unnumbered = [g for g in groups if not numbers.get(g)]
            numbered.sort(key=_numkey)
            row_of = {rep: i for i, rep in enumerate(numbered)}
            ends = []                          # 무번호 그룹 그리디 배치
            for rep in sorted(unnumbered,
                              key=lambda r: min(m[0] for m in groups[r])):
                g0 = min(m[0] for m in groups[rep])
                g1 = max(m[1] for m in groups[rep])
                for k, e in enumerate(ends):
                    if e <= g0:
                        ends[k] = g1
                        row_of[rep] = len(numbered) + k
                        break
                else:
                    row_of[rep] = len(numbered) + len(ends)
                    ends.append(g1)
            for rep, members in groups.items():
                si = row_of[rep]
                for f0, f1, tid, role in members:
                    out.append((tid, f0, f1, role, si))
                if len(members) > 1:           # 병합 그룹 밴드
                    g0 = min(m[0] for m in members)
                    g1 = max(m[1] for m in members)
                    self._group_bands.append((lane, si, g0, g1,
                                              members[0][3]))
            self._lane_rows[lane] = max(1, len(numbered) + len(ends))
        self._players = out
        self.update()

    def set_pos(self, f):
        if f != self.pos:
            self.pos = f
            self.update()

    def set_selection(self, kind, key):
        sel = (kind, key) if kind is not None else None
        if sel != self.selected:
            self.selected = sel
            self.update()

    @staticmethod
    def _lane_of_role(role):
        if role in (0, 3):
            return 3
        if role in (1, 4):
            return 4
        return 5

    # --------------------------------------------------------------- 좌표계
    def _eff_ppf(self):
        fit = (self.width() - self.GUTTER) / max(self.total, 1)
        return max(self.ppf, fit) if self.ppf > 0 else fit

    def _clamp_view(self):
        ppf = self._eff_ppf()
        vis = (self.width() - self.GUTTER) / max(ppf, 1e-9)
        self.t0 = min(max(self.t0, 0.0), max(0.0, self.total - vis))

    def _x(self, f):
        return int(self.GUTTER + (f - self.t0) * self._eff_ppf())

    def _f(self, x):
        return (x - self.GUTTER) / self._eff_ppf() + self.t0

    COLLAPSED_H = 12

    def _hscale(self):
        """레인 비례 배율 — 위젯 높이를 항상 꽉 채운다 (사용자 요청).

        lane_h 는 비율로 취급: 접힌 레인(고정 12px)을 뺀 나머지 공간을
        lane_h 비례로 나눈다. 패널을 키우면 레인이 같이 커진다.
        """
        fixed = sum(self.COLLAPSED_H for i in range(len(self.lane_h))
                    if i in self.collapsed)
        flex = sum(h for i, h in enumerate(self.lane_h)
                   if i not in self.collapsed)
        avail = self.height() - self.RULER - fixed
        return max(avail, 1) / flex if flex > 0 else 1.0

    def _eff_h(self, i):
        if i in self.collapsed:
            return self.COLLAPSED_H
        return self.lane_h[i] * self._hscale()

    def solo_lane(self, lane):
        """이 레인만 확대 — 나머지 전부 접기 (우클릭 메뉴).

        이미 단독 상태면 전체 펼치기로 토글.
        """
        others = set(range(len(self.lane_h))) - {lane}
        if self.collapsed == others:          # 이미 단독 → 복원
            self.collapsed = set()
        else:
            self.collapsed = others
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_timeline_collapsed", [int(v) for v in self.collapsed])
        self._apply_height()
        self.update()

    def expand_all(self):
        if self.collapsed:
            self.collapsed = set()
            QSettings("PyStitch360", "PyStitch360").setValue(
                "ptz_timeline_collapsed", [])
            self._apply_height()
            self.update()

    def toggle_collapse(self, lane):
        if lane in self.collapsed:
            self.collapsed.discard(lane)
        else:
            self.collapsed.add(lane)
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_timeline_collapsed", [int(v) for v in self.collapsed])
        self._apply_height()
        self.update()

    def _apply_height(self):
        # 꽉 채움 모드: 최소 높이만 보장, 실제 높이는 레이아웃이 준다
        self.setMinimumHeight(self.RULER
                              + 10 * max(len(self.lane_h), 1))
        self.updateGeometry()
        self.update()

    def _lane_rect(self, lane):
        y = self.RULER + sum(self._eff_h(i) for i in range(lane))
        return int(y), int(self._eff_h(lane))

    def _lane_at(self, y):
        acc = self.RULER
        for i in range(len(self.lane_h)):
            h = self._eff_h(i)
            if acc <= y < acc + h:
                return i
            acc += h
        return -1

    def _boundary_at(self, y, tol=4):
        """y 가 레인 경계(하단선) 근처면 그 레인 인덱스 (접힌 레인 제외)."""
        acc = self.RULER
        for i in range(len(self.lane_h)):
            acc += self._eff_h(i)
            if abs(y - acc) <= tol:
                return None if i in self.collapsed else i
        return None

    RESIZE_GAIN = 0.35           # 드래그 감쇠 — 많이 움직여도 조금만 조정

    def _apply_lane_resize(self, dy):
        """진행 중인 경계 드래그 적용 (Shift = 전체 비례).

        미세 조정이 가능하도록 감쇠(RESIZE_GAIN)를 둔다. 전체 비례
        모드는 레인 수 배가 아니라 합계 기준 — 구버전은 dy×레인 수라
        9배 속도로 움직여 원하는 크기에 세우기 어려웠다.
        """
        i, orig = self._resize["lane"], self._resize["orig"]
        if self._resize["all"]:
            total0 = sum(orig)
            factor = max(0.3, (total0 + dy * self.RESIZE_GAIN * 3) / total0)
            self.lane_h = [max(10, min(240, int(round(h * factor))))
                           for h in orig]
        else:
            self.lane_h = list(orig)
            self.lane_h[i] = max(10, min(240, orig[i]
                                         + int(round(dy * self.RESIZE_GAIN))))
        self._apply_height()
        self.update()

    # --------------------------------------------------------------- 그리기
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(34, 34, 38))
        W = self.width()
        # 레인 배경/라벨
        for i, name in enumerate(self.lanes):
            y, lh = self._lane_rect(i)
            if i % 2 == 0:
                p.fillRect(self.GUTTER, y, W - self.GUTTER, lh,
                           QColor(255, 255, 255, 8))
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRect(4, y, self.GUTTER - 8, lh),
                       Qt.AlignmentFlag.AlignVCenter, name)
            p.setPen(QColor(255, 255, 255, 25))
            p.drawLine(0, y + lh, W, y + lh)
        # 눈금자: 60px 이상 간격이 되는 단위 선택
        if self.total > 1:
            ppf = self._eff_ppf()
            for step_s in (1, 2, 5, 10, 30, 60, 120, 300, 600):
                if step_s * self.fps * ppf >= 60:
                    break
            f = int(self.t0 // (step_s * self.fps)) * step_s * self.fps
            p.setPen(QColor(180, 180, 180))
            while f <= self.t0 + (W - self.GUTTER) / ppf:
                x = self._x(f)
                if x >= self.GUTTER:
                    p.drawLine(x, 0, x, self.RULER - 4)
                    t = f / self.fps
                    p.drawText(x + 3, self.RULER - 4,
                               f"{int(t//60):02d}:{int(t%60):02d}")
                f += step_s * self.fps
        # 레인 0: 키프레임 (3요소 = 위치, 4요소 = 크롭 박스 폭 포함)
        y, lh = self._lane_rect(0)
        for i, k in enumerate(self.kfs):
            x = self._x(k[0])
            crop = len(k) > 3
            c = QColor(255, 130, 0) if crop else QColor(255, 190, 0)
            r = (x - 4, y + 3, 9, lh - 6)
            p.fillRect(*r, c)
            if self.selected == ("kf", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        # 레인 1: 공 — 수락 트랙 (동시 트랙은 서브행 분리), 무시, 승격
        y, lh = self._lane_rect(1)
        nrows = 1 if 1 in self.collapsed else getattr(self, "_ball_nrows", 1)
        pitch = (lh - 4) / max(nrows, 1)
        rows = getattr(self, "_ball_rows", [])
        for i, (f0, f1) in enumerate(self.spans):
            row = 0 if 1 in self.collapsed else \
                (rows[i] if i < len(rows) else 0)
            ry = y + 2 + row * pitch
            rh = max(3, int(pitch) - 1)
            r = (self._x(f0), int(ry),
                 max(2, self._x(f1) - self._x(f0)), rh)
            p.fillRect(*r, QColor(70, 200, 90))
            if self.selected == ("ball", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        for i, rg in enumerate(self.ignores):
            if len(rg) >= 4:
                # 위치 있는 무시 = "그 물체 하나가 공 아님" — 트랙과 같은
                # 굵기의 빨간 바 (레인 하단). 동시간 매치볼 위를 덮지 않는다
                bh2 = max(3, int(pitch) - 1)
                r = (self._x(rg[0]), y + lh - 2 - bh2,
                     max(2, self._x(rg[1]) - self._x(rg[0])), bh2)
                p.fillRect(*r, QColor(220, 70, 60, 200))
            else:
                # 구형 시간 전체 무시 — 실제로 그 시간대 전체를 기각하므로
                # 전체 높이 반투명 밴드 유지
                r = (self._x(rg[0]), y + 2,
                     max(2, self._x(rg[1]) - self._x(rg[0])), lh - 4)
                p.fillRect(*r, QColor(220, 70, 60, 90))
                p.setPen(QColor(220, 70, 60))
                p.drawRect(r[0], r[1], r[2], r[3])
            if self.selected == ("ignore", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        for pr in self.promotes:
            x = self._x(pr[0])
            p.fillRect(x - 1, y + 2, 3, lh - 4, QColor(200, 0, 255))
        # 병합 그룹 밴드: 멤버 사이 갭을 반투명하게 (하나로 합쳐졌음 표시)
        for lane, si, g0, g1, role in getattr(self, "_group_bands", []):
            y, lh = self._lane_rect(lane)
            fold = lane in self.collapsed
            pitch = (lh - 4) / (1 if fold
                                else max(self._lane_rows.get(lane, 3), 1))
            ry = y + 2 + int((0 if fold else si) * pitch)
            b, g, rr = self.role_palette.get(
                role, TEAM_COLORS[min(role, len(TEAM_COLORS) - 1)])
            x0_, x1_ = self._x(g0), self._x(g1)
            p.fillRect(x0_, int(ry), max(2, x1_ - x0_),
                       max(2, int(pitch) - 1), QColor(rr, g, b, 70))
        # 선수 레인: 팀/역할 색 바 — 서브행 수는 동시 트랙릿 수 (은폐 없음,
        # 접힌 레인은 얇은 한 줄로 겹쳐 그림)
        for tid, f0, f1, role, si in self._players:
            lane = self._lane_of_role(role)
            y, lh = self._lane_rect(lane)
            fold = lane in self.collapsed
            pitch = (lh - 4) / (1 if fold
                                else max(self._lane_rows.get(lane, 3), 1))
            bh = max(1, int(pitch) - (1 if pitch >= 3 else 0))
            ry = y + 2 + int((0 if fold else si) * pitch)
            b, g, rr = self.role_palette.get(
                role, TEAM_COLORS[min(role, len(TEAM_COLORS) - 1)])
            r = (self._x(f0), ry, max(2, self._x(f1) - self._x(f0)), bh)
            p.fillRect(*r, QColor(rr, g, b))
            num = getattr(self, "_pnum", {}).get(tid)
            if num and not fold:               # 신원 확인 — 테두리 강조
                p.setPen(QColor(255, 255, 255, 180))
                p.drawRect(r[0], r[1], r[2], max(1, r[3] - 1))
                if bh >= 10 and r[2] >= 22:    # 높이 되면 번호 표시
                    p.drawText(QRect(r[0] + 3, r[1], r[2] - 4, bh),
                               Qt.AlignmentFlag.AlignVCenter, num)
            if self.selected == ("player", tid):
                p.setPen(QPen(QColor(255, 40, 40), 2))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        # 뜬 공 레인: 공중 구간 바 (하늘색, 정점 높이 라벨)
        if self.airborne and self.total > 1:
            y, lh = self._lane_rect(2)
            c = QColor(120, 200, 255)
            for f0, f1, apex in self.airborne:
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                p.fillRect(max(x0_, self.GUTTER), y + 4,
                           max(3, x1_ - max(x0_, self.GUTTER)), lh - 8, c)
                if x1_ - x0_ > 44:
                    p.setPen(QColor(20, 40, 60))
                    p.drawText(QRect(x0_ + 3, y, x1_ - x0_ - 4, lh),
                               Qt.AlignmentFlag.AlignVCenter,
                               f"{apex:.1f}m")
        # 호각 레인: 확실한 이벤트(피크 ≥ WHISTLE_MIN_DB)만 불연속 마커로
        # — 연속 바는 노이즈처럼 보여 제거 (원본 트랙은 파일에 보관)
        if self._whistle is not None and self.total > 1:
            _, _, events = self._whistle
            y, lh = self._lane_rect(6)
            for t0_, t1_, db in events:
                if db < self.WHISTLE_MIN_DB:
                    continue
                x_ = self._x(t0_ * self.fps)
                if not (self.GUTTER - 4 <= x_ <= W):
                    continue
                w_ = max(3, self._x(t1_ * self.fps) - x_)
                c = QColor(255, 150, 40)
                p.fillRect(x_, y + 3, w_, lh - 6, c)
                p.setPen(c)
                # 긴 호각(킥오프·종료 후보)은 위 눈금까지 강조
                if t1_ - t0_ >= 0.8:
                    p.drawLine(x_, y + 1, x_ + w_, y + 1)
        # 하이라이트 레인: 구간 바 (후보=호박, 수락=초록, 제외=회색)
        if self.highlights and self.total > 1:
            y, lh = self._lane_rect(8)
            for i, (f0, f1, state, label) in enumerate(self.highlights):
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                c = {"accept": QColor(110, 210, 70, 200),
                     "reject": QColor(130, 130, 130, 80)}.get(
                    state, QColor(255, 176, 32, 180))
                r = (max(x0_, self.GUTTER), y + 3,
                     max(3, x1_ - max(x0_, self.GUTTER)), lh - 6)
                p.fillRect(*r, c)
                if self.selected == ("hl", i):
                    p.setPen(QColor(255, 255, 255))
                    p.drawRect(r[0], r[1], r[2] - 1, r[3] - 1)
                if x1_ - x0_ > 44 and state != "reject":
                    p.setPen(QColor(35, 28, 8))
                    p.drawText(QRect(r[0] + 3, y, x1_ - r[0] - 4, lh),
                               Qt.AlignmentFlag.AlignVCenter, label)
        # 이벤트 레인: 자동(킥오프)=초록, 사용자=시안 마커 + 라벨
        if self.events and self.total > 1:
            y, lh = self._lane_rect(7)
            for i, (f_, label, kind) in enumerate(self.events):
                x_ = self._x(f_)
                if not (self.GUTTER - 4 <= x_ <= W + 4):
                    continue
                c = (QColor(90, 220, 120) if kind == "auto"
                     else QColor(80, 200, 230))
                p.fillRect(x_ - 1, y + 2, 3, lh - 4, c)
                p.setPen(c)
                if self.selected == ("event", i):
                    p.drawRect(x_ - 4, y + 1, 8, lh - 2)
                p.drawText(QRect(x_ + 5, y, 140, lh),
                           Qt.AlignmentFlag.AlignVCenter, label)
        # 랜드마크 레인: 찍은 프레임마다 색 틱 (+ 라벨 있으면 표기)
        if self.landmarks and self.total > 1:
            y, lh = self._lane_rect(9)
            for i, (f_, label, col) in enumerate(self.landmarks):
                x_ = self._x(f_)
                if not (self.GUTTER - 4 <= x_ <= W + 4):
                    continue
                c = QColor(int(col[2]), int(col[1]), int(col[0]))
                p.fillRect(x_ - 1, y + 2, 2, lh - 4, c)
                p.setPen(c)
                if self.selected == ("landmark", i):
                    p.drawRect(x_ - 3, y + 1, 6, lh - 2)
                if label:
                    p.drawText(QRect(x_ + 4, y, 60, lh),
                               Qt.AlignmentFlag.AlignVCenter, label)
        # 경기 중단 구간: 회색 세로 밴드 (시계 정지 — hydration break 등)
        if self.pauses and self.total > 1:
            for f0, f1 in self.pauses:
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                p.fillRect(max(x0_, self.GUTTER), self.RULER,
                           max(2, x1_ - max(x0_, self.GUTTER)),
                           self.height() - self.RULER,
                           QColor(150, 150, 160, 45))
                p.setPen(QColor(170, 170, 180))
                p.drawText(max(x0_, self.GUTTER) + 3, self.RULER + 11, "II")
        # 소유 리본 (P08): 눈금자 바로 아래 팀 색 4px 밴드
        if self.possession and self.total > 1:
            for f0, f1, team in self.possession:
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                c = self.role_palette.get(team)
                col = QColor(c[2], c[1], c[0]) if c else                     QColor(80, 160, 255) if team == 0 else QColor(255, 120, 80)
                p.fillRect(max(x0_, self.GUTTER), self.RULER + 1,
                           max(2, x1_ - max(x0_, self.GUTTER)), 4, col)
        # 멀티캠 앵글 레인 (P07): 커버리지 밴드 + 그 카메라 호각 마커
        # (primary 시간축 변환) — primary 호각 레인과 정렬 = 동기화 육안 검증
        if self.angles and self.total > 1:
            for i, ang in enumerate(self.angles):
                y, lh = self._lane_rect(len(self.LANES) + i)
                f0, f1 = ang["span"]
                x0_ = self._x(max(0, f0))
                x1_ = self._x(min(self.total - 1, f1))
                if x1_ >= self.GUTTER and x0_ <= W:
                    p.fillRect(max(x0_, self.GUTTER), y + 2,
                               max(2, x1_ - max(x0_, self.GUTTER)), lh - 4,
                               QColor(50, 150, 160, 55))
                for t0_, t1_, db in ang.get("whistles", []):
                    if db < self.WHISTLE_MIN_DB:
                        continue
                    x_ = self._x(t0_ * self.fps)
                    if not (self.GUTTER - 4 <= x_ <= W):
                        continue
                    w_ = max(3, self._x(t1_ * self.fps) - x_)
                    c = QColor(80, 220, 230)
                    p.fillRect(x_, y + 3, w_, lh - 6, c)
                    if t1_ - t0_ >= 0.8:      # 긴 호각 강조 (primary 와 동일)
                        p.setPen(c)
                        p.drawLine(x_, y + 1, x_ + w_, y + 1)
        # 내보내기 구간: 바깥은 어둡게, IN/OUT 브래킷 표시
        if self.mark_in is not None or self.mark_out is not None:
            fi = 0 if self.mark_in is None else self.mark_in
            fo = self.total - 1 if self.mark_out is None else self.mark_out
            xa, xb = self._x(fi), self._x(fo)
            dim = QColor(0, 0, 0, 110)
            if xa > self.GUTTER:
                p.fillRect(self.GUTTER, self.RULER,
                           xa - self.GUTTER, self.height(), dim)
            if xb < W:
                p.fillRect(xb, self.RULER, W - xb, self.height(), dim)
            p.setPen(QColor(80, 230, 120))
            p.drawLine(xa, 0, xa, self.height())
            p.drawText(xa + 3, self.height() - 4, "IN")
            p.setPen(QColor(240, 120, 80))
            p.drawLine(xb, 0, xb, self.height())
            p.drawText(xb - 26, self.height() - 4, "OUT")
        # 거터 마스크 + 플레이헤드
        p.fillRect(0, self.RULER, self.GUTTER, self.height(), QColor(34, 34, 38))
        for i, name in enumerate(self.lanes):     # 라벨 다시 (마스크 위)
            y, lh = self._lane_rect(i)
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRect(4, y, self.GUTTER - 8, lh),
                       Qt.AlignmentFlag.AlignVCenter, name)
        x = self._x(self.pos)
        if x >= self.GUTTER:
            p.setPen(QColor(255, 60, 60))
            p.drawLine(x, 0, x, self.height())
        p.end()

    # --------------------------------------------------------------- 조작
    def _hit(self, x, y):
        f = self._f(x)
        lane = self._lane_at(y)
        if lane == 0:
            best, bd = None, 8.0
            for i, k in enumerate(self.kfs):
                d = abs(self._x(k[0]) - x)
                if d < bd:
                    best, bd = ("kf", i), d
            return best
        if lane == 1:
            for i, rg in enumerate(self.ignores):
                if rg[0] <= f <= rg[1]:
                    return ("ignore", i)
            # 서브행 우선 매칭 — 동시 트랙(겹친 시간대)에서 클릭한 행의
            # 트랙을 정확히 잡고, 행 밖 클릭은 시간 겹침으로 폴백
            ly, lh = self._lane_rect(1)
            nrows = 1 if 1 in self.collapsed else getattr(self, "_ball_nrows", 1)
            pitch = (lh - 4) / max(nrows, 1)
            rows = ([0] * len(self.spans) if 1 in self.collapsed
                    else getattr(self, "_ball_rows", []))
            hit_row = int((y - ly - 2) / max(pitch, 1e-9))
            fallback = None
            for i, (f0, f1) in enumerate(self.spans):
                if f0 <= f <= f1:
                    if (rows[i] if i < len(rows) else 0) == hit_row:
                        return ("ball", i)
                    if fallback is None:
                        fallback = ("ball", i)
            return fallback
        if lane in (3, 4, 5):
            ly, lh = self._lane_rect(int(lane))
            rows_n = 1 if lane in self.collapsed \
                else self._lane_rows.get(lane, 3)
            pitch = (lh - 4) / max(rows_n, 1)
            bh = max(1, int(pitch))
            for tid, f0, f1, role, si in self._players:
                if self._lane_of_role(role) != lane:
                    continue
                ry = ly + 2 + int((0 if lane in self.collapsed else si)
                                  * pitch)
                if f0 <= f <= f1 and ry - 1 <= y <= ry + bh:
                    return ("player", tid)
        if lane == 7:
            best, bd = None, 10.0
            for i, (f_, label, kind) in enumerate(self.events):
                d = abs(self._x(f_) - x)
                if d < bd:
                    best, bd = ("event", i), d
            return best
        if lane == 8:
            for i, (f0, f1, state, label) in enumerate(self.highlights):
                if f0 <= f <= f1:
                    return ("hl", i)
        if lane == 9:
            best, bd = None, 10.0
            for i, (f_, label, col) in enumerate(self.landmarks):
                d = abs(self._x(f_) - x)
                if d < bd:
                    best, bd = ("landmark", i), d
            return best
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        x, y = int(ev.position().x()), int(ev.position().y())
        if y >= self.height() - 4:          # 하단 가장자리 = 전체 비례 리사이즈
            self._resize = {"lane": len(self.lane_h) - 1, "y": y,
                            "orig": list(self.lane_h), "all": True}
            return
        if x < self.GUTTER:                 # 레이블 영역: 경계 드래그 = 높이
            b = self._boundary_at(y)
            if b is not None:
                self._resize = {"lane": b, "y": y, "orig": list(self.lane_h),
                                "all": bool(ev.modifiers()
                                            & Qt.KeyboardModifier.ShiftModifier)}
                return
        if self.total <= 1:
            return
        self._press = {"x": x, "t0": self.t0, "moved": False,
                       "hit": self._hit(x, y) if y >= self.RULER else None}

    def mouseMoveEvent(self, ev):
        x, y = int(ev.position().x()), int(ev.position().y())
        if self._resize is not None:
            self._apply_lane_resize(y - self._resize["y"])
            return
        if self._press is None:
            # 호버 커서: 하단 가장자리(전체) / 레이블 영역 경계(개별)
            on_edge = (y >= self.height() - 4
                       or (x < self.GUTTER
                           and self._boundary_at(y) is not None))
            self.setCursor(Qt.CursorShape.SizeVerCursor if on_edge
                           else Qt.CursorShape.ArrowCursor)
            # 동적 툴팁: 레인 경계=조절 안내 / 번호 트랙릿=번호 (택일)
            tip = ""
            if on_edge:
                tip = "레인 경계 드래그 = 높이 조절 (Shift = 전체 비례)"
            elif x >= self.GUTTER:
                hit = self._hit(x, y)
                if hit and hit[0] == "player":
                    tid = hit[1]
                    rep = getattr(self, "_prep", {}).get(tid, tid)
                    num = getattr(self, "_pnum", {}).get(rep)
                    tip = f"#{tid}"
                    if rep != tid:
                        tip += f" (병합→#{rep})"
                    if num:
                        tip += f"  {num}번"
            from PyQt6.QtWidgets import QToolTip
            if tip:
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
            else:
                QToolTip.hideText()
            return
        dx = x - self._press["x"]
        if abs(dx) > 3:
            self._press["moved"] = True
        if self._press["moved"] and self._press["hit"] is None:
            self.t0 = self._press["t0"] - dx / self._eff_ppf()   # 팬
            self._clamp_view()
            self.update()
            self._emit_view()

    def mouseReleaseEvent(self, ev):
        if self._resize is not None:
            self._resize = None
            QSettings("PyStitch360", "PyStitch360").setValue(
                "ptz_timeline_lanes", [int(v) for v in self.lane_h])
            return
        pr, self._press = self._press, None
        if pr is None or pr["moved"]:
            return
        if pr["hit"] is not None:
            self.selected = pr["hit"]
            self.update()
            fclick = int(min(max(self._f(int(ev.position().x())), 0),
                             self.total - 1))
            self.pick.emit(pr["hit"][0], pr["hit"][1], fclick)
        elif ev.position().x() >= self.GUTTER:
            self.seek.emit(int(min(max(self._f(int(ev.position().x())), 0),
                                   self.total - 1)))
            lane = self._lane_at(int(ev.position().y()))
            if lane >= len(self.LANES):     # 앵글 레인 클릭 → 카메라 전환
                self.angle_pick.emit(lane - len(self.LANES))

    def wheelEvent(self, ev):
        if self.total <= 1:
            return
        delta = ev.angleDelta().y() / 120.0
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            x = int(ev.position().x())
            f_at = self._f(max(x, self.GUTTER))
            fit = (self.width() - self.GUTTER) / self.total
            self.ppf = min(max(self._eff_ppf() * (1.25 ** delta), fit), 30.0)
            self.t0 = f_at - (max(x, self.GUTTER) - self.GUTTER) / self.ppf
        else:
            self.t0 -= delta * 80 / self._eff_ppf()
        self._clamp_view()
        self.update()
        self._emit_view()
        ev.accept()


class AnalyzeWorker(QThread):
    done = pyqtSignal(dict)
    progress = pyqtSignal(int, int, float)   # done_frames, total, fps
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, weights=None, checkpoint_path=None):
        super().__init__()
        self.video_path = video_path
        self.weights = weights
        self.checkpoint_path = checkpoint_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            d = analyze_video(self.video_path, weights=self.weights,
                              cancel=lambda: self._cancel,
                              progress=lambda i, t, f: self.progress.emit(i, t, f),
                              checkpoint_path=self.checkpoint_path,
                              log=lambda s: self.log.emit(s))
            if d is None:
                self.failed.emit("취소됨")
            else:
                self.done.emit(d)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class LinkWorker(QThread):
    """트랙 연결(느린 단계) 백그라운드 계산 — 분석에만 의존하므로 1회면 됨.

    analysis_path 가 있으면 디스크 캐시(.link.cache.npz) 경유 —
    재열기·헤드리스 예열 시 수십 초 → 즉시.
    """

    done = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self, analysis, analysis_path=None, video_path=None):
        super().__init__()
        self.analysis = analysis
        self.analysis_path = analysis_path
        self.video_path = video_path

    def run(self):
        import time
        from ..core.ptz import link_ball_tracks_cached, measure_top_black
        t0 = time.perf_counter()
        try:
            self.log.emit("[ptz] · 공 트랙 연결 중…")
            if self.analysis_path:
                linked = link_ball_tracks_cached(self.analysis_path,
                                                 self.analysis)
            else:
                linked = link_ball_tracks(self.analysis)
            self.log.emit(
                f"[ptz] · 팀 색 분류 중… (+{time.perf_counter()-t0:.1f}s)")
            linked["teams"] = classify_teams(self.analysis)
            if self.video_path:                # 상단 검은 경계 실측
                self.log.emit(
                    f"[plan] · 상단 검은 경계 실측 중 — 원본 5프레임 "
                    f"시크(성긴 GOP면 수 초)… (+{time.perf_counter()-t0:.1f}s)")
                linked["top_black"] = measure_top_black(self.video_path)
            self.log.emit(
                f"[ptz] · 트랙 연결 워커 완료 (+{time.perf_counter()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001 — 예외로 죽으면 대기 커서가
            # 영영 안 풀린다. 오류를 담아 done 을 반드시 방출 (_link_done
            # 이 커서 해제 후 안전 종료).
            self.log.emit(f"[ptz] 트랙 연결 워커 오류: {e}")
            self.done.emit({"_error": str(e)})
            return
        self.done.emit(linked)


class GapfillWorker(QThread):
    """갭필 2차 패스: 트랙 갭 보간 위치의 저문턱 타일 검출 (공+선수)."""

    progress = pyqtSignal(int, int, float)
    done = pyqtSignal(dict, int, int)     # analysis, n_ball, n_person
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pano_path, analysis, targets, weights=None):
        super().__init__()
        self.args = (pano_path, analysis, targets, weights)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, analysis, targets, weights = self.args
        try:
            nb, np_ = gapfill_analysis(
                pano, analysis, targets, weights=weights,
                progress=lambda i, t, f: self.progress.emit(i, t, f),
                cancel=lambda: self._cancel,
                log=lambda s: self.log.emit(s))
            self.done.emit(analysis, nb, np_)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class AirborneWorker(QThread):
    """공중볼 스캔 (correct_ball_track) — UI 프리즈 방지용 백그라운드."""

    done = pyqtSignal(int, dict, list, str)  # gen, 캐시, 구간(직렬화), 로그

    def __init__(self, gen, frames, fps, acc, calib, segments=None):
        super().__init__()
        self.args = (gen, frames, fps, acc, calib, segments)

    def run(self):
        gen, frames, fps, acc, calib, segments = self.args
        try:
            from ..core.airborne import correct_ball_track
            from ..core.field import pano_to_field
            fin = np.isfinite(acc[:, 0])
            if fin.sum() < 10:
                self.done.emit(gen, {}, [], "")
                return
            g = np.full((len(acc), 2), np.nan)
            g[fin] = pano_to_field(calib, acc[fin])
            t = frames / fps
            corr, z, segs = correct_ball_track(
                t, g, (calib["ex"], calib["ey"]), calib["h"],
                segments=segments)
            cache = {}
            for i0, i1, fit in segs:
                for si in range(i0, i1 + 1):
                    cache[si] = (float(corr[si, 0]), float(corr[si, 1]),
                                 float(z[si]))
            msg = ""
            if segs:
                cached = " (캐시 재사용)" if segments is not None else ""
                msg = (f"[air] 공중 구간 {len(segs)}개 (정점 최대 "
                       f"{max(f['apex_z'] for _, _, f in segs):.1f}m)"
                       + cached)
            self.done.emit(gen, cache,
                           [[int(i0), int(i1), f] for i0, i1, f in segs],
                           msg)
        except Exception as e:  # noqa: BLE001
            self.done.emit(gen, {}, [], f"[air] 공중볼 보정 실패: {e}")


class SeedWorker(QThread):
    """시드 전파: 수동 공/선수 인식을 앞뒤 샘플로 확장 (propagate_seed)."""

    done = pyqtSignal(str, list, object)   # kind, matches, ctx(tid 등)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pano_path, analysis, f0, x0, y0, kind,
                 weights=None, ctx=None):
        super().__init__()
        self.args = (pano_path, analysis, f0, x0, y0, kind, weights, ctx)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, analysis, f0, x0, y0, kind, weights, ctx = self.args
        try:
            m = propagate_seed(pano, analysis, f0, x0, y0, kind=kind,
                               weights=weights,
                               cancel=lambda: self._cancel,
                               log=lambda s: self.log.emit(s))
            self.done.emit(kind, m, ctx)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PlanWorker(QThread):
    """크롭 계획 미리보기 재계산 (키프레임/무시 변경 시 백그라운드)."""

    done = pyqtSignal(dict, tuple)   # plan, (out_w, out_h)

    def __init__(self, analysis, keyframes, ignores, wide, linked=None,
                 far_zoom=1.0, promotes=None, exclude_tids=None,
                 top_margin=None):
        super().__init__()
        self.args = (analysis, keyframes, ignores, wide, linked, far_zoom,
                     promotes or [], exclude_tids or set(), top_margin)

    def run(self):
        (analysis, kfs, ignores, wide, linked, far_zoom, promotes,
         exclude_tids, top_margin) = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes, linked=linked,
                              exclude_tids=exclude_tids,
                              far_zoom=far_zoom,
                              **({"top_margin": top_margin}
                                 if top_margin is not None else {}),
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0, log=None)
            self.done.emit(plan, (out_w, out_h))
        except Exception:  # noqa: BLE001
            pass


class ProxyWorker(QThread):
    """스크럽 프록시 생성: 1920px, 키프레임 0.5s 간격 → 즉시 탐색용."""

    progress = pyqtSignal(float)         # 0.0 ~ 1.0
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, proxy_path, duration_sec=0.0):
        super().__init__()
        self.pano_path, self.proxy_path = str(pano_path), str(proxy_path)
        self.duration = duration_sec

    def run(self):
        import subprocess

        from ..core.encoders import available_encoders, ffmpeg_bin
        encs = available_encoders().values()
        if "h264_nvenc" in encs:
            venc = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr",
                    "-cq", "27", "-b:v", "0"]
        else:
            venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]
        tmp = self.proxy_path + ".part.mp4"
        cmd = ([ffmpeg_bin(), "-y", "-v", "error", "-nostats",
                "-i", self.pano_path, "-vf", "scale=1920:-2"] + venc
               + ["-g", "15", "-an", "-progress", "pipe:1", tmp])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            tail = []
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms=") and self.duration > 0:
                    try:                     # ffmpeg 의 out_time_ms 는 마이크로초
                        t = int(line.split("=", 1)[1]) / 1e6
                        self.progress.emit(min(t / self.duration, 1.0))
                    except ValueError:
                        pass
                elif line and "=" not in line:
                    tail.append(line)        # 진행 키가 아닌 줄 = 오류 메시지 후보
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(" / ".join(tail[-3:]) or "ffmpeg 실패")
            Path(tmp).rename(self.proxy_path)
            self.finished_ok.emit(self.proxy_path)
        except Exception as e:  # noqa: BLE001
            Path(tmp).unlink(missing_ok=True)
            self.failed.emit(str(e))


class PtzPlayWorker(QThread):
    """파노라마 순차 재생 (자체 디코더) — 오버레이는 GUI 쪽 _redraw 가 담당."""

    frame_ready = pyqtSignal(object, int, float)   # (BGR, frame, disp_scale)

    def __init__(self, path, start_frame, fps, display_fps=15.0,
                 disp_w=1600):
        super().__init__()
        self.path, self.start_frame = str(path), start_frame
        self.fps, self.display_fps = fps, display_fps
        self.disp_w = int(disp_w)             # 재생 표시 폭 상한 (화질↓ 속도↑)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import time
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            step = max(1, int(round(self.fps / self.display_fps)))
            f = self.start_frame
            t0 = time.perf_counter()
            shown = 0
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                sc = 1.0
                if self.disp_w and frame.shape[1] > self.disp_w:
                    sc = self.disp_w / frame.shape[1]
                    frame = cv2.resize(
                        frame, (self.disp_w, int(frame.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
                self.frame_ready.emit(frame, f, sc)
                shown += 1
                lag = shown * step / self.fps - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
                for _ in range(step - 1):
                    cap.grab()
                f += step
        finally:
            cap.release()


class NativeFrameWorker(QThread):
    """프록시 표시 직후 원본(파노라마) 해상도 프레임을 백그라운드로 읽어
    교체용으로 넘긴다. 자체 디코더를 소유하고, 요청이 몰리면 **최신
    프레임만** 처리한다 (스크럽 중 밀린 요청은 버림)."""

    frame_ready = pyqtSignal(object, int)          # (BGR 원본, frame)

    def __init__(self):
        super().__init__()
        from PyQt6.QtCore import QMutex, QWaitCondition
        self._mx = QMutex()
        self._cv = QWaitCondition()
        self._req = None                           # (path, frame, t, w, h) 최신
        self._stop = False

    def request(self, path, frame, t, w, h):
        self._mx.lock()
        self._req = (str(path), int(frame), float(t), int(w), int(h))
        self._cv.wakeAll()
        self._mx.unlock()

    def stop(self):
        self._mx.lock()
        self._stop = True
        self._cv.wakeAll()
        self._mx.unlock()
        self.wait(3000)

    def run(self):
        while True:
            self._mx.lock()
            while self._req is None and not self._stop:
                self._cv.wait(self._mx)
            if self._stop:
                self._mx.unlock()
                break
            path, frame, t, w, h = self._req
            self._req = None
            self._mx.unlock()
            img = _ffmpeg_frame(path, t, w, h)     # 빠른 시크 (~2~3초)
            # 디코드하는 사이 더 새 요청이 들어왔으면 이 결과는 버린다.
            # 실패(img None)라도 방출 — GUI 가 대기 표시를 해제하게.
            self._mx.lock()
            stale = self._req is not None or self._stop
            self._mx.unlock()
            if not stale:
                self.frame_ready.emit(img, frame)


class PtzRenderWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_path, analysis, keyframes, codec, crf,
                 wide=False, ignores=None, far_zoom=1.0, promotes=None,
                 radar=None, start=0, end=None, clock=None):
        super().__init__()
        self.args = (pano_path, out_path, analysis, keyframes, codec, crf, wide,
                     ignores or [], far_zoom, promotes or [], radar, start, end,
                     clock)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        (pano, out, analysis, kfs, codec, crf, wide, ignores, far_zoom,
         promotes, radar, start, end, clock) = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes,
                              far_zoom=far_zoom,
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0,
                              log=lambda s: self.log.emit(s))
            render_plan(pano, out, plan, out_w=out_w, out_h=out_h,
                        codec=codec, crf=crf,
                        log=lambda s: self.log.emit(s),
                        progress=lambda d, t, f: self.progress.emit(d, t, f),
                        cancel=lambda: self._cancel, radar=radar,
                        start=start, end=end, clock=clock)
            if self._cancel:
                self.failed.emit("취소됨")
            else:
                self.finished_ok.emit(str(out))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportDialog(QDialog):
    """PTZ 내보내기 설정 대화창 — 구간(IN/OUT 마커 기준)·모드·코덱·미니맵."""

    def __init__(self, parent, total, fps, export_range, mode_idx,
                 encoders, crf, radar_on, default_dir, default_stem,
                 clock_on=None):
        super().__init__(parent)
        self.setWindowTitle("PTZ 내보내기")
        self.fps = fps
        self.total = total
        self.export_range = export_range      # (f0, f1) 정규화됨 or None
        self.default_dir = default_dir
        self.default_stem = default_stem
        form = QFormLayout(self)

        self.combo_range = QComboBox()
        f0, f1 = export_range if export_range else (0, total)
        dur = (f1 - f0) / fps
        if export_range:
            self.combo_range.addItem(
                f"마커 구간  {self._hms(f0/fps)} ~ {self._hms(f1/fps)} "
                f"(길이 {self._hms(dur)}, {f1-f0}프레임)")
        self.combo_range.addItem(
            f"전체  0:00:00 ~ {self._hms(total/fps)} ({total}프레임)")
        form.addRow("구간", self.combo_range)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.setCurrentIndex(mode_idx)
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("모드", self.combo_mode)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(list(encoders))
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_export_codec", "")
        labels = list(encoders)
        if saved in labels:                       # 직전 선택 기억
            self.combo_codec.setCurrentIndex(labels.index(saved))
        else:                                     # GPU 있으면 hevc_nvenc 기본
            for idx, lbl in enumerate(labels):
                if encoders[lbl] == "hevc_nvenc":
                    self.combo_codec.setCurrentIndex(idx)
                    break
        form.addRow("코덱", self.combo_codec)

        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=crf)
        form.addRow("CRF/CQ", self.spin_crf)

        self.check_radar = QCheckBox("우하단 반투명 탑다운 미니맵 (선수·공)")
        self.check_radar.setChecked(radar_on)
        form.addRow("미니맵", self.check_radar)

        self.check_clock = QCheckBox("좌상단 경기 시계 (분:초 누적, "
                                     "골1/골2 이벤트로 스코어)")
        if clock_on is None:
            self.check_clock.setEnabled(False)
            self.check_clock.setToolTip(
                "분석 메뉴 \"경기 정보\"에서 킥오프 앵커를 지정하세요")
        else:
            self.check_clock.setChecked(bool(clock_on))
        form.addRow("경기 시계", self.check_clock)

        path_row = QHBoxLayout()
        self.edit_path = QLineEdit(self._default_path())
        path_row.addWidget(self.edit_path, 1)
        b = QPushButton("찾아보기...")
        b.clicked.connect(self._browse)
        path_row.addWidget(b)
        form.addRow("출력 파일", path_row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("내보내기")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(560, self.sizeHint().height())

    @staticmethod
    def _hms(sec):
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{s:04.1f}"

    def _default_path(self):
        wide = self.combo_mode.currentIndex() == 1
        part = ""
        if self.export_range and self.combo_range.currentIndex() == 0:
            part = f"_{int(self.export_range[0]/self.fps)}s"
        return str(Path(self.default_dir)
                   / f"{self.default_stem}{part}"
                     f"{'_wide' if wide else '_ptz'}.mp4")

    def _mode_changed(self, _):
        self.edit_path.setText(self._default_path())

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "PTZ 출력 파일",
                                           self.edit_path.text(),
                                           "MP4 (*.mp4)")
        if p:
            self.edit_path.setText(p)

    def config(self):
        """선택 결과: {start, end, wide, codec_name, crf, radar, path}."""
        use_marks = self.export_range and self.combo_range.currentIndex() == 0
        f0, f1 = (self.export_range if use_marks else (0, self.total))
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_export_codec", self.combo_codec.currentText())
        return {"start": int(f0), "end": int(f1),
                "wide": self.combo_mode.currentIndex() == 1,
                "codec_name": self.combo_codec.currentText(),
                "crf": self.spin_crf.value(),
                "radar": self.check_radar.isChecked(),
                "clock": self.check_clock.isChecked(),
                "path": self.edit_path.text().strip()}


class MatchInfoDialog(QDialog):
    """경기 정보 — 시계 앵커(킥오프)·전/후반·하프 길이·중단 구간 요약.

    시계는 축구 관례의 분:초 누적 표기 (후반 30분 하프면 30:00부터,
    연장 포함 90/120분도 분 단위로 계속 커진다).
    """

    def __init__(self, parent, fps, total, kickoffs, info, cur_frame):
        super().__init__(parent)
        self.setWindowTitle("경기 정보")
        info = info or {}
        self.fps = fps
        form = QFormLayout(self)

        self.combo_half = QComboBox()
        self.combo_half.addItems(["전반", "후반"])
        self.combo_half.setCurrentIndex(1 if info.get("half", 1) == 2 else 0)
        form.addRow("이 영상의 하프", self.combo_half)

        self.spin_len = QDoubleSpinBox(minimum=5.0, maximum=90.0,
                                       value=float(info.get("half_len_min",
                                                            45.0)))
        self.spin_len.setSuffix(" 분")
        self.spin_len.setDecimals(0)
        form.addRow("하프 길이", self.spin_len)

        self.combo_anchor = QComboBox()
        self._anchor_frames: list = []
        for k in kickoffs:
            self.combo_anchor.addItem(
                f"킥오프 {ExportDialog._hms(k['t'])} (신뢰도 {k['score']:.2f})")
            self._anchor_frames.append(int(k["t"] * fps))
        self.combo_anchor.addItem(
            f"현재 프레임 {ExportDialog._hms(cur_frame / fps)}")
        self._anchor_frames.append(int(cur_frame))
        self.combo_anchor.addItem("지정 안 함 (시계 사용 불가)")
        self._anchor_frames.append(None)
        saved = info.get("anchor_f")
        if saved is not None:
            best = min(range(len(self._anchor_frames) - 1),
                       key=lambda i: abs(self._anchor_frames[i] - saved),
                       default=None)
            if best is not None \
                    and abs(self._anchor_frames[best] - saved) > 2 * fps:
                # 저장된 앵커가 목록에 없음 — 그대로 보존하는 항목 추가
                self.combo_anchor.insertItem(
                    0, f"저장된 앵커 {ExportDialog._hms(saved / fps)}")
                self._anchor_frames.insert(0, int(saved))
                best = 0
            self.combo_anchor.setCurrentIndex(best if best is not None else 0)
        else:
            self.combo_anchor.setCurrentIndex(len(self._anchor_frames) - 1)
        form.addRow("킥오프 앵커 (시계 0점)", self.combo_anchor)

        self.check_cum = QCheckBox("후반 시계에 전반 시간 누적 "
                                   "(하프 길이만큼 더해 30:00/45:00부터)")
        self.check_cum.setChecked(bool(info.get("cumulative", True)))
        form.addRow("", self.check_cum)

        n_pause = len(info.get("pauses") or [])
        form.addRow("중단 구간", QLabel(
            f"{n_pause}개 — 타임라인 우클릭 \"IN/OUT → 경기 중단\"으로 "
            "추가/삭제 (중단 동안 시계 정지)"))
        form.addRow("스코어", QLabel(
            "사용자 이벤트 라벨 \"골1\"/\"골2\" = 팀1/팀2 득점으로 집계 "
            "→ 시계 옆에 표시"))

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._pauses = [list(p) for p in info.get("pauses") or []]

    def config(self) -> dict:
        return {"half": self.combo_half.currentIndex() + 1,
                "half_len_min": float(self.spin_len.value()),
                "anchor_f": self._anchor_frames[
                    self.combo_anchor.currentIndex()],
                "cumulative": self.check_cum.isChecked(),
                "pauses": self._pauses}


class OcrWorker(QThread):
    """등번호 OCR (근측) — easyocr 로딩·프레임 시크가 느려 백그라운드."""

    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, analysis, picked):
        super().__init__()
        self.args = (pano_path, analysis, picked)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, analysis, picked = self.args
        try:
            from ..core.ocr import run_jersey_ocr
            out = run_jersey_ocr(
                pano, analysis, picked,
                progress=lambda d, t, f: self.progress.emit(d, t, f),
                cancel=lambda: self._cancel,
                log=lambda s: self.log.emit(s))
            self.done.emit(out)
        except ImportError:
            self.failed.emit("easyocr 미설치 — pip install easyocr")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class HighlightBatchWorker(QThread):
    """수락 하이라이트 구간들을 개별 클립으로 순차 렌더 (계획 1회 재사용)."""

    progress = pyqtSignal(int, int, float)   # (전체 누적, 전체 프레임, fps)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(int, str)       # (완료 개수, 출력 폴더)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_dir, stem, analysis, keyframes, codec,
                 crf, ignores=None, far_zoom=1.0, promotes=None, radar=None,
                 segments=(), clock=None, alt=None):
        super().__init__()
        self.args = (pano_path, out_dir, stem, analysis, keyframes, codec,
                     crf, ignores or [], far_zoom, promotes or [], radar,
                     list(segments), clock, alt)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        (pano, out_dir, stem, analysis, kfs, codec, crf, ignores, far_zoom,
         promotes, radar, segs, clock, alt) = self.args
        try:
            out_w, out_h = 1920, 1080
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              ignore_ranges=ignores, force_ranges=promotes,
                              far_zoom=far_zoom,
                              log=lambda s: self.log.emit(s))
            total = sum(e - s for s, e, _ in segs)
            base = 0
            for k, (s, e, name) in enumerate(segs, 1):
                if self._cancel:
                    break
                out = Path(out_dir) / f"{stem}_h{k:02d}_{name}.mp4"
                self.log.emit(f"[hl] {k}/{len(segs)} 렌더: {out.name}")
                render_plan(pano, out, plan, out_w=out_w, out_h=out_h,
                            codec=codec, crf=crf,
                            log=lambda s: self.log.emit(s),
                            progress=lambda d, _t, f, b=base:
                                self.progress.emit(b + d, total, f),
                            cancel=lambda: self._cancel, radar=radar,
                            start=s, end=e, clock=clock)
                if alt is not None and not self._cancel:
                    # 동기화된 다른 카메라 앵글 (P06-2 조기 성과물)
                    from ..core.sync_multi import cut_synced_clip
                    fps = analysis["fps"]
                    out_a = Path(out_dir) / (
                        f"{stem}_h{k:02d}_{name}_{alt['label']}.mp4")
                    try:
                        cut_synced_clip(alt["path"], alt, s / fps, e / fps,
                                        out_a, codec=codec, crf=crf)
                        self.log.emit(f"[hl] 대체 앵글: {out_a.name}")
                    except Exception as ex:  # noqa: BLE001
                        self.log.emit(f"[hl] 대체 앵글 실패: {ex}")
                base += e - s
            if self._cancel:
                self.failed.emit("취소됨")
            else:
                self.finished_ok.emit(len(segs), str(out_dir))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class HighlightExportDialog(QDialog):
    """하이라이트 일괄 내보내기 — 구간 체크 목록·출력 폴더·코덱/CRF/미니맵."""

    def __init__(self, parent, highlights, fps, encoders, crf, radar_on,
                 default_dir, clock_on=None, alt_label=None, alt_span=None):
        super().__init__(parent)
        self.setWindowTitle("하이라이트 일괄 내보내기")
        self.fps = fps
        form = QFormLayout(self)

        self.list = QListWidget()
        any_accept = any(h.get("state") == "accept" for h in highlights)
        for h in highlights:
            dur = h["t1"] - h["t0"]
            badge = ""                        # 앵글 뱃지: alt 가 구간을 덮는가
            if alt_label and alt_span:
                s0, s1 = alt_span
                if h["t0"] >= s0 and h["t1"] <= s1:
                    badge = f"  [{alt_label}]"
                elif h["t1"] > s0 and h["t0"] < s1:
                    badge = f"  [{alt_label} 일부]"
            it = QListWidgetItem(
                f"{ExportDialog._hms(h['t0'])} ~ {ExportDialog._hms(h['t1'])}"
                f"  ({dur:.0f}초)  {h.get('label', '')}"
                f"  점수 {h.get('score', 0):.1f}"
                + ("  [수락]" if h.get("state") == "accept" else "") + badge)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # 수락 항목이 하나라도 있으면 수락만 체크, 없으면 전부 체크
            it.setCheckState(Qt.CheckState.Checked
                             if h.get("state") == "accept" or not any_accept
                             else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.setMinimumHeight(180)
        form.addRow("구간", self.list)

        dir_row = QHBoxLayout()
        self.edit_dir = QLineEdit(default_dir)
        dir_row.addWidget(self.edit_dir, 1)
        b = QPushButton("찾아보기...")
        b.clicked.connect(self._browse)
        dir_row.addWidget(b)
        form.addRow("출력 폴더", dir_row)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(list(encoders))
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_export_codec", "")
        labels = list(encoders)
        if saved in labels:
            self.combo_codec.setCurrentIndex(labels.index(saved))
        else:
            for idx, lbl in enumerate(labels):
                if encoders[lbl] == "hevc_nvenc":
                    self.combo_codec.setCurrentIndex(idx)
                    break
        form.addRow("코덱", self.combo_codec)

        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=crf)
        form.addRow("CRF/CQ", self.spin_crf)
        self.check_radar = QCheckBox("우하단 반투명 탑다운 미니맵 (선수·공)")
        self.check_radar.setChecked(radar_on)
        form.addRow("미니맵", self.check_radar)
        self.check_clock = QCheckBox("좌상단 경기 시계 (분:초 누적)")
        if clock_on is None:
            self.check_clock.setEnabled(False)
            self.check_clock.setToolTip(
                "분석 메뉴 \"경기 정보\"에서 킥오프 앵커를 지정하세요")
        else:
            self.check_clock.setChecked(bool(clock_on))
        form.addRow("경기 시계", self.check_clock)

        self.check_alt = QCheckBox(
            f"동기화된 다른 카메라 앵글 동시 추출 ({alt_label})"
            if alt_label else "동기화된 다른 카메라 없음")
        if alt_label is None:
            self.check_alt.setEnabled(False)
            self.check_alt.setToolTip(
                "scripts/sync_cams.py 로 두 영상을 먼저 동기화하세요")
        else:
            self.check_alt.setChecked(True)
        form.addRow("대체 앵글", self.check_alt)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("일괄 내보내기")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(640, self.sizeHint().height())

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "출력 폴더",
                                             self.edit_dir.text())
        if d:
            self.edit_dir.setText(d)

    def config(self):
        """선택 결과: {indices, dir, codec_name, crf, radar}."""
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_export_codec", self.combo_codec.currentText())
        idx = [i for i in range(self.list.count())
               if self.list.item(i).checkState() == Qt.CheckState.Checked]
        return {"indices": idx, "dir": self.edit_dir.text().strip(),
                "codec_name": self.combo_codec.currentText(),
                "crf": self.spin_crf.value(),
                "radar": self.check_radar.isChecked(),
                "clock": self.check_clock.isChecked(),
                "alt": self.check_alt.isChecked()}


class PtzTab(QWidget):
    """가상 PTZ 탭. log_fn 은 메인 윈도우 로그 박스."""

    def __init__(self, log_fn, video_dir_fn=None, remember_dir_fn=None):
        super().__init__()
        self._ext_log = log_fn
        self._video_dir = video_dir_fn or (lambda: "")
        self._remember_dir = remember_dir_fn or (lambda p: None)
        self.pano_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.total = 0
        self.pano_w = self.pano_h = 0
        self.analysis: dict | None = None
        self.keyframes: list[list] = []   # [frame, x, y]
        self.ignores: list[list] = []     # [f0, f1] 사용자 지정 오인식 구간
        self.promotes: list[list] = []    # [f, x, y] 회색 공 → 트랙 강제 수락
        self.export_range: list = [None, None]   # 내보내기 IN/OUT 프레임
        self.team_names = ["팀1", "팀2"]  # 사용자 입력 팀 이름
        self.user_events: list[list] = []  # [[frame, label]] 사용자 이벤트
        self.highlights: list[dict] = []   # .events.json "highlights" 문서
        self.match_info: dict | None = None  # 경기 정보 (하프·앵커·중단)
        self._referees: dict = {}         # .events.json "referees" (근/원측 선심)
        self._ocr_nums: dict = {}         # .events.json "ocr_numbers" 제안
        self._radar_palette: dict = {}    # {역할: BGR} 레이더 오버레이 색
        self._radar_smooth: dict = {}     # {tid: [x, y]} 미니맵 EMA 상태
        self._radar_smooth_f = None       # EMA 마지막 프레임 (점프 리셋용)
        self._ar_side_cache: dict = {}    # {대표 tid: "ARN"|"ARF"} 자동 판정
        self.roles: dict[int, int] = {}   # {track_id: 역할} 사용자 지정 (GK 등)
        self.merges: dict[int, int] = {}  # {tid: 대표 tid} 트랙릿 병합 (비파괴)
        self.player_nums: dict[int, str] = {}  # {대표 tid: 등번호}
        self.rosters: dict[int, list] = {}     # {팀: ["7 이름", ...]} 명단
        self.hidden_players: set[int] = set()  # 숨긴 대표 tid (관중·오인식)
        self._outfield: set[int] = set()  # 캘리브 기반 장외 트랙릿 (자동·비파괴)
        self._foot_cache = None           # (analysis id, tids, 발밑 px) — 장외 판정
        self._rb = None                   # 좌드래그 러버밴드 (사람 다중 선택)
        self._pane_sel: set[int] = set()  # 러버밴드로 고른 대표 tid
        self.field_points: dict[str, list] = {}   # {랜드마크키: [x, y]}
        # 랜드마크를 찍은 프레임 {키: frame} — 팬 카메라(AX700)용: 프레임
        # 간 이송(rotcam)으로 기준 프레임에 합쳐 캘리브레이션한다.
        # 파노라마(고정 뷰)에선 정보로만 존재.
        self.field_point_frames: dict[str, int] = {}
        # {랜드마크키: [[f0,f1], ...]} — 그 구간에선 이송이 엉뚱해 무효
        # (원 프레임에선 유효, 특정 프레임대 이송만 제외). rotcam 전용.
        self.field_point_invalid: dict[str, list] = {}
        # {라인키: [[px,py,frame], ...]} — 터치라인 등 위의 다점 (point-to-
        # line 캘리브). 소실점으로 f 를 안정화. rotcam 전용. LINE_FAMILIES.
        self.field_lines: dict[str, list] = {}
        # {랜드마크키: [[px,py,frame], ...]} — 무효의 양성 짝: 안 찍었지만
        # 이송이 정확한 걸 눈으로 확인해 '확정'한 추가 앵커. 이후 이송은
        # 대상 프레임에 가장 가까운 앵커에서 출발해 체인을 짧게. rotcam 전용.
        self.field_point_anchors: dict[str, list] = {}
        self._Hcache = {}                 # (src,dst)→이송 H (open 시 재설정)
        self._graycache = {}              # 프레임→det_w 그레이 (open 시 재설정)
        self._lm_transferred = {}         # 현재 프레임 이송 랜드마크
        self._rc_pose_gen = 0             # 프레임별 자세 캐시 무효화 키
        self.line_points: list[list] = []  # 흰 선 검출 샘플 [x, y] (사이드라인)
        self.field_size = [105.0, 68.0]   # 경기장 길이×폭 (m)
        self.field_circle_r = CENTER_CIRCLE_R  # 센터서클 반지름(m) — 경기장별
        self.field_circle_auto = True     # 중앙선 4점에서 R 자동 추정
        self._field_calib = None          # fit_field_calibration 결과
        self.extra_players: dict[int, list] = {}  # {샘플si: [[cx,cy,w,h,id]]}
        self._next_extra_id = 900001      # 수동 검출 ID (분석 ID와 분리)
        self._adhoc = None                # 주변 재검출용 YOLO 캐시
        self._native_cap = None           # 프록시 표시 중 원본 프레임용
        self._cur_scale = 1.0             # 현재 _cur_frame 의 표시 스케일
        self._native_worker = None        # 프록시→원본 백그라운드 승격
        self._native_pending = None       # 원본 승격 대기 중인 프레임
        self.track_spans: list = []
        self._pcache_id = None            # 선수 요약 캐시 기준 분석 객체 id
        self._pcolors_id = None           # 대표색 캐시 키 (분석에만 의존)
        self._tfeat = None                # classify_teams 전처리 캐시
        self._tfeat_id = None
        self._pbgr = {}                   # {tid: BGR} 대표색 변환 캐시
        self._pbgr_id = None
        self._footmed = {}                # {tid: 발 위치 중앙값} (_ar_side 용)
        self._footmed_id = None
        self._pspans: dict = {}           # {tid: [f0, f1, 검출수]}
        self._pcolors: dict = {}          # {tid: (h, s, v)} 유니폼 대표색
        self._accepted_ball = None        # accept_ball_tracks 의 샘플별 수락 공
        self._air: dict[int, tuple] = {}  # {si: (X, Y, z)} 공중볼 보정 캐시
        self._hover = None                # 커서가 가리키는 오브젝트
        self._hover_key = None            # hover 변경 감지 키
        self._plan_box = None             # 현재 프레임에 그려진 크롭 박스 (x0,y0,w,h)
        self._box_hover = None            # 크롭 박스 hover 존: ("corner",i)|("border",None)
        self._box_edit = None             # 진행 중인 박스 드래그 상태
        self._box_commit = None           # 커밋 직후 낙관적 박스 (f, cx, cy, w)
        self._lm_drag = None              # 드래그 중인 랜드마크 키
        self._analyze_worker = None
        self._gapfill_worker = None
        self._seed_worker = None
        self._air_worker = None
        self._air_gen = 0
        self._render_worker = None
        self._plan_worker = None
        self._link_worker = None
        self._linked = None
        self._teams = {}
        self._role_colors: dict[int, tuple] = {}   # 역할별 유니폼 대표색(BGR)
        self.kit_colors: dict[int, list] = {}      # 사용자 지정 표시 색(BGR)
        self._play_worker = None
        self._playing = False
        self._play_until = None          # 구간 재생 끝 프레임 (자동 정지)
        self._audio = None                # QMediaPlayer (지연 초기화, False=불가)
        self._audio_out = None
        self._proxy_worker = None
        self.disp_path = None
        self.disp_scale = 1.0
        self._play_scale = None
        self.plan = None
        self.plan_out = (1920, 1080)
        self.match: dict | None = None    # 멀티캠 경기 문서 (P07, match.json)
        self.match_half = 0
        self.match_cam = 0                # 활성 카메라 (0=primary, P07 v2)
        self.mc = None                    # MulticamViewer — 경기 열 때 생성
        self._opening_match = False
        self._build_ui()
        st = QSettings("PyStitch360", "PyStitch360")
        for cb, key in ((self.check_players, "ptz_show_players"),
                        (self.check_ball, "ptz_show_ball"),
                        (self.check_crop, "ptz_show_crop"),
                        (self.check_radar, "ptz_show_radar"),
                        (self.check_radar_smooth, "ptz_radar_smooth")):
            cb.setChecked(st.value(key, "true") == "true")
            cb.toggled.connect(
                lambda on, k=key: (QSettings("PyStitch360", "PyStitch360")
                                   .setValue(k, "true" if on else "false"),
                                   self._redraw()))
        self.sld_radar_alpha.setValue(int(st.value("ptz_radar_alpha", 55)))
        self.sld_radar_alpha.valueChanged.connect(
            lambda v: (QSettings("PyStitch360", "PyStitch360")
                       .setValue("ptz_radar_alpha", int(v)),
                       self._redraw()))
        # 스페이스 = 재생/정지 — 포커스가 아니라 커서 위치 기준이라
        # (타임라인/영상 위에서만) 앱 필터로 처리, 그 외엔 통과
        QApplication.instance().installEventFilter(self)
        self._plan_timer = QTimer(singleShot=True, interval=150)
        self._plan_timer.timeout.connect(self._run_plan)
        self._save_timer = QTimer(singleShot=True, interval=1500)
        self._save_timer.timeout.connect(self._write_sidecar)

    def _smooth_radar(self, pts, ball_g, f):
        """미니맵 EMA 스무딩 — 검출 지터(±0.3m) 완화.

        pts = [(tid|None, x, y, role)]. tid 기준 상태 유지, 시간 점프
        (>3s)면 리셋. tid 없는 점(주입 검출)은 스무딩 없이 통과.
        반환: ([(x, y, role)], ball).
        """
        if not self.check_radar_smooth.isChecked():
            self._radar_smooth = {}
            self._radar_smooth_f = None
            return [(x, y, r) for _t, x, y, r in pts], ball_g
        if self._radar_smooth_f is None \
                or abs(f - self._radar_smooth_f) > 3 * self.fps:
            self._radar_smooth = {}
        self._radar_smooth_f = f
        a = 0.45
        st = self._radar_smooth
        out = []
        seen = set()
        for tid, x, y, role in pts:
            if tid is None:
                out.append((x, y, role))
                continue
            p = st.get(tid)
            if p is None:
                st[tid] = p = [x, y]
            else:
                p[0] += a * (x - p[0])
                p[1] += a * (y - p[1])
            seen.add(tid)
            out.append((p[0], p[1], role))
        if ball_g is not None:
            p = st.get("__ball__")
            if p is None:
                st["__ball__"] = p = list(ball_g)
            else:
                p[0] += a * (ball_g[0] - p[0])
                p[1] += a * (ball_g[1] - p[1])
            seen.add("__ball__")
            ball_g = (p[0], p[1])
        for k in list(st):               # 사라진 트랙 상태 정리
            if k not in seen:
                del st[k]
        return out, ball_g

    def _draw_radar_overlay(self, pts, ball_g):
        """우하단 반투명 탑다운 레이더 — 위젯 오버레이.

        예전엔 프레임(BGR)에 합성해서 영상 줌/팬을 따라 움직였다 —
        위젯으로 분리해 화면상 위치/크기가 고정된다 (내보내기 합성
        경로 draw_radar_panel 은 그대로).
        """
        from PyQt6.QtGui import QImage, QPainter, QPixmap
        radar = {"frames": [0], "points": [pts], "balls": [ball_g],
                 "length": float(self.field_size[0]),
                 "width": float(self.field_size[1]),
                 "palette": self._radar_palette}
        pw = max(160, self.pane.width() // 5) & ~1
        if self.pane.width() < 320:
            self._radar_lbl.hide()
            return
        panel = draw_radar_panel(radar, 0, pw)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, rgb.strides[0],
                    QImage.Format.Format_RGB888)
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setOpacity(self.sld_radar_alpha.value() / 100.0)
        painter.drawImage(0, 0, qi)
        painter.end()
        self._radar_lbl.setPixmap(pm)
        self._radar_lbl.resize(w, h)
        self._radar_lbl.move(self.pane.width() - w - 10,
                             self.pane.height() - h - 10)
        self._radar_lbl.show()

    def _hide_radar(self):
        self._radar_lbl.hide()

    def log(self, msg):
        """메인 윈도우 로그 + 탭 내 로그 미러. 각 줄 앞에 현재 시각(ms)
        — 단계 사이 지연을 눈으로 재기 위함."""
        from datetime import datetime
        msg = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
        self._ext_log(msg)
        lv = getattr(self, "log_view", None)
        if lv is not None:
            lv.appendPlainText(str(msg))

    def eventFilter(self, obj, ev):
        """커서가 타임라인/슬라이더/영상 위일 때 Space = 재생/정지.

        포커스 기반(QShortcut)이 아니라 커서 기반 — 목록·버튼이 포커스를
        가져도 동작하고, 커서가 딴 데 있으면 이벤트를 그대로 통과시켜
        다른 위젯의 Space 동작(목록 선택 등)을 깨지 않는다.
        """
        if ev.type() == QEvent.Type.Resize \
                and obj is getattr(self, "pane", None) \
                and self._radar_lbl.isVisible():
            pm = self._radar_lbl.pixmap()
            if pm is not None:
                self._radar_lbl.move(self.pane.width() - pm.width() - 10,
                                     self.pane.height() - pm.height() - 10)
        if ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat() \
                and self.cap is not None:
            k = ev.key()
            space = k == Qt.Key.Key_Space
            # 숫자 1..9 = 카메라 선택 (멀티캠 경기 열려 있을 때, 같은 관례)
            digit = (self.match is not None
                     and Qt.Key.Key_1 <= k <= Qt.Key.Key_9)
            # Del = 선택된 사람 숨기기 (오인식/관중) — 영상 위 선택 있을 때
            delsel = (k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                      and bool(self._pane_sel))
            if space or digit or delsel:
                w = QApplication.widgetAt(QCursor.pos())
                hot = (self.trackbar, self.slider, self.tl_scroll, self.pane,
                       self._pane_split)
                while w is not None:
                    if w in hot:
                        if space:
                            self._toggle_play()
                        elif delsel:
                            self._sel_hide()
                        else:
                            self._mc_select(k - Qt.Key.Key_1)
                        return True
                    w = w.parentWidget()
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_open = QPushButton("파노라마 영상 열기...")
        self.btn_open.clicked.connect(self._open_pano)
        self.lbl_file = QLabel("완성된 파노라마 mp4 를 열어주세요")
        self.btn_analyze = QPushButton("자동 공/선수 분석")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_analyze)
        self.btn_proxy = QPushButton("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setToolTip("1920px·조밀 키프레임 사본 — 타임라인 클릭 즉시 표시")
        self.btn_proxy.clicked.connect(self._make_proxy)
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        top.addWidget(QLabel("모델"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n (기본·빠름)", "yolov8s", "yolov8m",
                                   "yolo11n", "yolo11s", "yolo11m (정확)",
                                   "사용자 .pt 파일..."])
        self.combo_model.setToolTip(
            "공/선수 검출 모델. GPU 추론이라 큰 모델도 부담 적음 —\n"
            "정확도가 아쉬우면 yolo11s/m 권장. 이름 모델은 최초 1회 자동 다운로드.")
        saved_model = QSettings("PyStitch360", "PyStitch360").value("ptz_model", 0)
        self.combo_model.setCurrentIndex(int(saved_model))
        self.combo_model.currentIndexChanged.connect(self._model_changed)
        top.addWidget(self.combo_model)
        top.addWidget(self.btn_proxy)
        top.addWidget(self.btn_analyze)
        v.addLayout(top)

        # 영상은 전체 폭 사용, 목록·레이더는 하단 스트립으로
        self.pane = FramePane("클릭 = 오브젝트 조작 / 빈 곳 = 키프레임, 우클릭 = 메뉴",
                              interactive=True)
        # 레이더는 위젯 오버레이 — 영상 줌/팬과 무관하게 고정 위치/크기
        self._radar_lbl = QLabel(self.pane)
        # 부모(pane) 스타일시트의 background-color 상속 차단 — 이게
        # 남으면 반투명 픽스맵 뒤에 불투명 판이 깔린다
        self._radar_lbl.setStyleSheet("background: transparent;")
        self._radar_lbl.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._radar_lbl.hide()
        self.pane.clicked.connect(self._pane_clicked)
        self.pane.context_requested.connect(self._pane_context)
        self.pane.hover.connect(self._pane_hover)
        self.pane.pressed.connect(self._pane_pressed)
        self.pane.drag_moved.connect(self._pane_dragged)
        self.pane.released.connect(self._pane_released)
        # 우상단 오버레이 체크박스: 선수/공/크롭 박스/레이더 표시 토글
        self.check_players = QCheckBox("선수")
        self.check_ball = QCheckBox("공")
        self.check_crop = QCheckBox("크롭 박스")
        self.check_radar = QCheckBox("레이더")
        self.check_radar_smooth = QCheckBox("스무딩")
        self.check_radar_smooth.setToolTip("미니맵 위치 지터 완화 (EMA)")
        self.check_infield = QCheckBox("장외 숨김")
        self.check_infield.setToolTip(
            "경기장 캘리브레이션 기반 — 발밑이 피치 밖인 사람(관중·"
            "대기선수)을 표시·목록·레이더·크롭 계획에서 자동 제외 (비파괴)")
        ov = QVBoxLayout(self.pane)
        ov.setContentsMargins(8, 8, 8, 8)
        ov_row = QHBoxLayout()
        self._lbl_title = QLabel("")
        self._lbl_title.setStyleSheet(
            "QLabel { color: white; background: rgba(20,20,20,150);"
            " padding: 2px 10px; border-radius: 4px; font-weight: bold; }")
        self._lbl_title.hide()
        ov_row.addWidget(self._lbl_title)
        ov_row.addStretch(1)
        for cb in (self.check_players, self.check_ball, self.check_crop,
                   self.check_radar, self.check_radar_smooth,
                   self.check_infield):
            cb.setChecked(True)
            cb.setStyleSheet(
                "QCheckBox { color: white; background: rgba(20,20,20,150);"
                " padding: 2px 8px; border-radius: 4px; }")
            ov_row.addWidget(cb)
        # 장외 숨김: 저장된 설정 복원 + 토글 시 장외 재판정·계획 갱신
        self.check_infield.setChecked(
            QSettings("PyStitch360", "PyStitch360")
            .value("ptz_infield_only", "true") == "true")
        self.check_infield.toggled.connect(self._infield_toggled)
        # 멀티캠 카메라/모드 바 (경기 열면 채워짐 — _rebuild_mc_bar)
        self._mc_row = QHBoxLayout()
        self._mc_row.addStretch(1)
        self.sld_radar_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sld_radar_alpha.setRange(10, 95)
        self.sld_radar_alpha.setFixedWidth(80)
        self.sld_radar_alpha.setToolTip("미니맵 불투명도")
        self.sld_radar_alpha.setStyleSheet(
            "QSlider { background: rgba(20,20,20,150); border-radius: 4px;"
            " padding: 2px 6px; }")
        ov_row.addWidget(self.sld_radar_alpha)
        ov.addLayout(ov_row)
        ov.addLayout(self._mc_row)
        ov.addStretch(1)
        # pane/타임라인/정보 3행은 세로 스플리터로 — 아래에서 합침

        tl = QHBoxLayout()
        # 컴팩트 트랜스포트: 2행 그리드 — 1행 재생/트랙 이동 + 시각,
        # 2행 스텝 버튼 6개. 타임라인 왼쪽에 좁게 붙는다.
        grid = QGridLayout()
        grid.setSpacing(2)

        def _tbtn(text, tip, slot, row, col, colspan=1):
            b = QPushButton(text)
            b.setFixedWidth(30 * colspan + 2 * (colspan - 1))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            grid.addWidget(b, row, col, 1, colspan)
            return b

        self.btn_play = _tbtn("▶", "재생/정지 (Space)", self._toggle_play, 0, 0)
        _tbtn("|◀", "이전 공 트랙으로", lambda: self._jump_track(-1), 0, 1)
        _tbtn("▶|", "다음 공 트랙으로", lambda: self._jump_track(1), 0, 2)
        # 빈 곳 클릭 모드: 공(수동 공 위치, 줌 자동) / KF(크롭 키프레임)
        self.btn_mode_ball = _tbtn("공", "빈 곳 클릭 = 수동 공 위치 (줌 자동)",
                                   lambda: self._set_click_mode(True), 0, 3)
        self.btn_mode_kf = _tbtn("KF", "빈 곳 클릭 = 크롭 키프레임 "
                                 "(현재 계획 폭 고정)",
                                 lambda: self._set_click_mode(False), 0, 4)
        for b in (self.btn_mode_ball, self.btn_mode_kf):
            b.setCheckable(True)
        self.btn_mode_ball.setChecked(True)
        self.btn_mute = _tbtn("🔊", "재생 소리 켬/끔",
                              self._toggle_mute, 0, 5)
        self.btn_mute.setCheckable(True)
        self.btn_mute.setChecked(
            QSettings("PyStitch360", "PyStitch360")
            .value("ptz_muted", "false") == "true")
        if self.btn_mute.isChecked():
            self.btn_mute.setText("🔇")
        for col, (text, tip, d) in enumerate(
                [("≪", "-10초", -300), ("<", "-1초", -30), ("‹", "-1프레임", -1),
                 ("›", "+1프레임", 1), (">", "+1초", 30), ("≫", "+10초", 300)]):
            _tbtn(text, tip, lambda _, dd=d: self._step(dd), 1, col)
        self.lbl_time = QLabel("-:--:--.- / -:--:--")
        self.lbl_time.setToolTip("현재 위치 / 전체 길이")
        grid.addWidget(self.lbl_time, 2, 0, 1, 6,
                       Qt.AlignmentFlag.AlignCenter)
        grid.setRowStretch(3, 1)          # 남는 세로 공간은 아래로
        tl.addLayout(grid)
        # 멀티트랙 타임라인은 슬라이더 바로 위, 같은 폭
        self.trackbar = TimelineView()
        self.trackbar.seek.connect(lambda f: self.slider.setValue(f))
        self.trackbar.pick.connect(self._timeline_pick)
        self.trackbar.angle_pick.connect(lambda i: self._mc_select(i + 1))
        self.trackbar.range_menu.connect(self._timeline_menu)
        # I/O = 내보내기 구간 시작/끝 (NLE 관례)
        QShortcut(QKeySequence(Qt.Key.Key_I), self,
                  activated=lambda: self._set_export_mark("in",
                                                          self.slider.value()))
        QShortcut(QKeySequence(Qt.Key.Key_O), self,
                  activated=lambda: self._set_export_mark("out",
                                                          self.slider.value()))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._stop_play)
        self.slider.valueChanged.connect(self._on_slider)
        bar_col = QVBoxLayout()
        bar_col.setSpacing(1)
        bar_col.addWidget(self.trackbar, 1)   # 남는 세로 = 타임라인이 채움
        # 줌인 시 가로 스크롤바 (전체 보기에선 숨김)
        self.tl_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.tl_scroll.setMaximumHeight(12)
        self.tl_scroll.hide()
        self.tl_scroll.valueChanged.connect(
            lambda v: self.trackbar.set_view_start(v))
        self.trackbar.view_changed.connect(self._tl_view_changed)
        bar_col.addWidget(self.tl_scroll)
        bar_col.addWidget(self.slider)
        tl.addLayout(bar_col, 1)
        self._w_tl = QWidget()
        self._w_tl.setLayout(tl)
        self._slider_timer = QTimer(singleShot=True, interval=120)
        self._slider_timer.timeout.connect(self._show_frame)

        # 하단 스트립: [공 | 선수] 탭 + 레이더 (로그 위 공간 활용)
        strip = QHBoxLayout()
        w_ball = QWidget()
        ball_strip = QHBoxLayout(w_ball)
        ball_strip.setContentsMargins(0, 2, 0, 0)
        col_ball = QVBoxLayout()
        col_ball.addWidget(QLabel("공 — 자동 트랙 + 수동 지정 (↑↓=이동, →/Del=오인식으로)"))
        self.track_list = QListWidget()
        self.track_list.setMaximumHeight(150)
        # 클릭·키보드 화살표 선택 모두에서 이동 (currentRowChanged 는 둘 다 발생)
        self.track_list.currentRowChanged.connect(lambda _: self._goto_track())
        for key in (Qt.Key.Key_Right, Qt.Key.Key_Delete):   # → 또는 Del = 오인식
            QShortcut(QKeySequence(key), self.track_list,
                      activated=lambda: self._ignore_selected_track(
                          advance=True),   # 키보드 검수: 다음 항목으로 이동
                      context=Qt.ShortcutContext.WidgetShortcut)
        col_ball.addWidget(self.track_list, 1)
        ball_strip.addLayout(col_ball, 2)

        # 두 목록 사이 이동 버튼: >> = 오인식으로, << = 복원
        col_move = QVBoxLayout()
        col_move.addStretch(1)
        self.btn_ignore = QPushButton("≫")
        self.btn_ignore.setMaximumWidth(44)
        self.btn_ignore.setToolTip("선택한 공 트랙을 오인식으로 (→/Del)")
        self.btn_ignore.clicked.connect(self._to_ignore)
        col_move.addWidget(self.btn_ignore)
        self.btn_restore = QPushButton("≪")
        self.btn_restore.setMaximumWidth(44)
        self.btn_restore.setToolTip("선택한 오인식을 복원 (←)")
        self.btn_restore.clicked.connect(self._delete_kf)
        col_move.addWidget(self.btn_restore)
        col_move.addStretch(1)
        ball_strip.addLayout(col_move)

        col_ig = QVBoxLayout()
        col_ig.addWidget(QLabel("오인식 — 공 아님 (더블클릭=이동, ←=복원)"))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(150)
        self.kf_list.itemDoubleClicked.connect(self._goto_kf)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.kf_list,
                  activated=self._delete_kf,
                  context=Qt.ShortcutContext.WidgetShortcut)
        col_ig.addWidget(self.kf_list, 1)
        ball_strip.addLayout(col_ig, 2)

        # 선수 탭: 트랙릿 목록(유니폼 색 스와치) + 역할 지정 + 팀 색 범례
        w_players = QWidget()
        pl = QVBoxLayout(w_players)
        pl.setContentsMargins(0, 2, 0, 0)
        # 유니폼 색 범례: 역할별 스와치 버튼(클릭=컬러피커) + 인원수
        legend = QHBoxLayout()
        legend.setSpacing(4)
        self.lbl_team_colors = QLabel("팀 색: 분석 후 표시")
        legend.addWidget(self.lbl_team_colors)
        self._kit_btns, self._kit_lbls = {}, {}
        for r in (0, 1, 3, 4, 5, 6):
            b = QPushButton()
            b.setFixedSize(26, 18)
            b.setToolTip(f"{self._role_name(r)} 표시 색 — 클릭: 색 선택, "
                         "우클릭: 측정색으로 되돌리기")
            b.clicked.connect(lambda _=False, rr=r: self._pick_kit_color(rr))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda _pos, rr=r: self._reset_kit_color(rr))
            lb = QLabel("")
            self._kit_btns[r], self._kit_lbls[r] = b, lb
            b.hide()
            legend.addWidget(b)
            legend.addWidget(lb)
        legend.addStretch(1)
        pl.addLayout(legend)
        self.player_list = QListWidget()
        self.player_list.setUniformItemSizes(True)
        self.player_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.player_list.currentRowChanged.connect(
            lambda _: self._goto_player())
        self.player_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.player_list.customContextMenuRequested.connect(self._player_menu)
        pl.addWidget(self.player_list, 1)
        rl = QHBoxLayout()
        rl.addWidget(QLabel("선택한 트랙릿 →"))
        self._role_btns = {}
        for r in (0, 1, 5, 6, 3, 4):     # 자주 쓰는 팀/심판 먼저, GK 뒤로
            b = QPushButton(self._role_name(r))
            b.setMaximumWidth(90)
            b.clicked.connect(lambda _, rr=r: self._assign_selected_role(rr))
            self._role_btns[r] = b
            rl.addWidget(b)
        b = QPushButton("팀 이름...")
        b.setMaximumWidth(72)
        b.clicked.connect(self._edit_team_names)
        rl.addWidget(b)
        b = QPushButton("자동")
        b.setMaximumWidth(56)
        b.setToolTip("사용자 지정을 지우고 색 기반 자동 분류로")
        b.clicked.connect(lambda: self._assign_selected_role(None))
        rl.addWidget(b)
        rl.addStretch(1)
        pl.addLayout(rl)

        # 경기장 탭: 랜드마크 자동 매칭 찍기 + 캘리브레이션 상태
        w_field = QWidget()
        fl = QVBoxLayout(w_field)
        fl.setContentsMargins(0, 2, 0, 0)
        self.lbl_field_status = QLabel(
            "목록 순서대로(최외곽 선부터) 찍는 걸 권장 — 클릭=자동 매칭, "
            "우클릭=수동 지정, 드래그=이동. 외곽 6~7점이면 외곽선이 잡힙니다.")
        fl.addWidget(self.lbl_field_status)
        self.field_list = QListWidget()
        fl.addWidget(self.field_list, 1)
        fr = QHBoxLayout()
        self.btn_field_pick = QPushButton("랜드마크 찍기")
        self.btn_field_pick.setCheckable(True)
        self.btn_field_pick.setToolTip(
            "켜면 미리보기 클릭이 랜드마크 지정이 됩니다 (휴리스틱 자동 매칭)")
        self.btn_field_pick.toggled.connect(self._field_pick_toggled)
        fr.addWidget(self.btn_field_pick)
        b = QPushButton("선택 지우기")
        b.clicked.connect(self._field_clear_selected)
        fr.addWidget(b)
        b = QPushButton("모두 지우기")
        b.setToolTip("찍은 랜드마크 전체 삭제")
        b.clicked.connect(self._field_clear_all)
        fr.addWidget(b)
        b = QPushButton("흰 선 정밀화")
        b.setToolTip("예측 사이드라인 주변의 흰 픽셀을 검출해 "
                     "가까운 사이드라인을 실측 선에 맞춤")
        b.clicked.connect(self._refine_sideline)
        fr.addWidget(b)
        b = QPushButton("흰 선 취소")
        b.setToolTip("흰 선 정밀화 샘플을 지우고 랜드마크만으로 재피팅")
        b.clicked.connect(self._clear_line_points)
        fr.addWidget(b)
        fr.addWidget(QLabel("경기장(m)"))
        self.spin_field_len = QDoubleSpinBox()
        self.spin_field_len.setRange(80.0, 130.0)
        self.spin_field_len.setValue(105.0)
        self.spin_field_len.setSuffix(" L")
        self.spin_field_w = QDoubleSpinBox()
        self.spin_field_w.setRange(40.0, 90.0)
        self.spin_field_w.setValue(68.0)
        self.spin_field_w.setSuffix(" W")
        for sp in (self.spin_field_len, self.spin_field_w):
            sp.valueChanged.connect(self._field_size_changed)
            fr.addWidget(sp)
        # 센터서클 반지름 — 경기장별로 다름 (유소년은 작음). 자동=중앙선
        # 4점 교차비로 추정, 수동이면 값 고정.
        self.spin_circle_r = QDoubleSpinBox()
        self.spin_circle_r.setRange(3.0, 15.0)
        self.spin_circle_r.setSingleStep(0.1)
        self.spin_circle_r.setSuffix(" R")
        self.spin_circle_r.setValue(self.field_circle_r)
        self.spin_circle_r.setToolTip("센터서클 반지름(m) — 경기장별")
        self.spin_circle_r.valueChanged.connect(self._circle_r_changed)
        fr.addWidget(self.spin_circle_r)
        self.chk_circle_auto = QCheckBox("R자동")
        self.chk_circle_auto.setChecked(True)
        self.chk_circle_auto.setToolTip("중앙선 4점(half·circle)으로 R 자동 추정")
        self.chk_circle_auto.toggled.connect(self._circle_auto_toggled)
        fr.addWidget(self.chk_circle_auto)
        # venue 프리셋 (경기장 규격 기억)
        fr.addWidget(QLabel("venue"))
        self.combo_venue = QComboBox()
        self.combo_venue.setMinimumWidth(120)
        self.combo_venue.setToolTip("저장한 경기장 규격 불러오기")
        self.combo_venue.activated.connect(self._apply_venue_idx)
        fr.addWidget(self.combo_venue)
        b = QPushButton("venue 저장")
        b.setToolTip("현재 경기장 길이·폭·서클R 을 이름 붙여 저장")
        b.clicked.connect(self._save_venue_dialog)
        fr.addWidget(b)
        fr.addStretch(1)
        fl.addLayout(fr)
        self._refresh_venue_combo()

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        self.tabs_review = QTabWidget()
        self.tabs_review.addTab(w_ball, "공")
        self.tabs_review.addTab(w_players, "선수")
        self.tabs_review.addTab(w_field, "경기장")
        self.tabs_review.addTab(self.log_view, "로그")
        self.tabs_review.currentChanged.connect(self._review_tab_changed)
        strip.addWidget(self.tabs_review, 1)
        w_strip = QWidget()
        w_strip.setLayout(strip)
        # 영상 행: 가로 스플리터 — 평소엔 pane 단독, 멀티캠 분할 모드에서
        # alt 페인이 오른쪽에 붙는다 (multicam.MulticamViewer)
        self._pane_split = QSplitter(Qt.Orientation.Horizontal)
        self._pane_split.addWidget(self.pane)
        self._pane_split.setCollapsible(0, False)
        # 영상/타임라인/정보 3행 — 스플리터로 높이 조절 (크기 저장/복원)
        self._rows = QSplitter(Qt.Orientation.Vertical)
        self._rows.addWidget(self._pane_split)
        self._rows.addWidget(self._w_tl)
        self._rows.addWidget(w_strip)
        self._rows.setStretchFactor(0, 1)     # 창 리사이즈 여분은 영상에
        self._rows.setCollapsible(0, False)
        saved_rows = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_row_sizes", None)
        try:
            self._rows.setSizes([max(0, int(s)) for s in saved_rows])
        except Exception:  # noqa: BLE001
            self._rows.setSizes([520, 260, 210])
        self._rows.splitterMoved.connect(
            lambda *_: QSettings("PyStitch360", "PyStitch360").setValue(
                "ptz_row_sizes", [int(s) for s in self._rows.sizes()]))
        v.addWidget(self._rows, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("출력"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.currentIndexChanged.connect(lambda _: self._plan_dirty())
        bottom.addWidget(self.combo_mode)
        lbl_fz = QLabel("원경 줌")
        lbl_fz.setToolTip("공이 반대편(원경)에 있을 때 추가 줌인 배율 (1.0=없음)")
        bottom.addWidget(lbl_fz)
        self.spin_far_zoom = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=1.5,
                                            singleStep=0.05)
        self.spin_far_zoom.setValue(float(QSettings("PyStitch360", "PyStitch360")
                                          .value("ptz_far_zoom", 1.0)))
        self.spin_far_zoom.valueChanged.connect(self._far_zoom_changed)
        bottom.addWidget(self.spin_far_zoom)
        self.encoders = available_encoders()
        self.btn_export = QPushButton("PTZ 내보내기...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._start_render)
        bottom.addWidget(self.btn_export)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_render)
        bottom.addWidget(self.btn_cancel)
        bottom.addStretch(1)
        v.addLayout(bottom)
        self.progress = QProgressBar()
        v.addWidget(self.progress)

    # ------------------------------------------------------------ 파일/분석
    def _open_pano(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "완성 파노라마 영상", self._video_dir(), "영상 (*.mp4 *.MP4 *.mkv)")
        if path:
            self.open_path(path)

    # ------------------------------------------------------ 멀티캠 (P07)
    def open_match(self, doc: dict, half: int = 0, cam: int = 0,
                   quiet: bool = False, seek_s: float | None = None,
                   path: str | None = None):
        """match.json 문서 열기 — cam(활성 카메라)을 열고 나머지를 앵글로.

        P07 v2 활성 카메라 컨텍스트: cam=0 은 파노라마, 1.. 은 alt.
        어느 쪽이든 그 영상의 사이드카 일체(분석·타임라인·캘리브레이션)
        가 컨텍스트가 되고, 다른 멤버들은 활성 기준 상대 시계로 앵글.
        """
        with self._wait():
            from ..core.match import half_cameras, relative_clock
            from .multicam import MulticamViewer
            half = max(0, min(half, len(doc["halves"]) - 1))
            h = doc["halves"][half]
            cams = half_cameras(h)
            cam = max(0, min(cam, len(cams) - 1))
            target = cams[cam]["video"]
            self._opening_match = True
            try:
                self.open_path(target, quiet=quiet)
            finally:
                self._opening_match = False
            if self.pano_path is None \
                    or str(self.pano_path) != str(Path(target)):
                return                            # 열기 실패
            self.match, self.match_half, self.match_cam = doc, half, cam
            if path is not None:
                self.match_path = path
            if self.mc is None:
                self.mc = MulticamViewer(self.pane, self._pane_split, self.log)
            others = []
            for j, c in enumerate(cams):
                if j == cam:
                    continue
                others.append({"video": c["video"],
                               "clock": relative_clock(cams, cam, j),
                               "cam": j})
            self.mc.set_half(others, redraw=self._redraw)
            self._rebuild_mc_bar()
            self._set_angle_lanes()
            self._sync_match_teams(quiet=quiet)
            if seek_s is not None:
                self.slider.setValue(
                    int(max(0, min(seek_s * self.fps, self.total - 1))))
            if not quiet:
                names = " / ".join(f"{j + 1} {Path(c['video']).stem}"
                                   for j, c in enumerate(cams))
                self.log(f"[match] {doc.get('title') or '경기'} — {h['label']} "
                         f"[활성: {Path(target).stem}]  카메라: {names} "
                         "(전환 모드 숫자키 = 컨텍스트 전환)")

    def _update_titles(self):
        """페인 좌상단 영상 제목 + 현 시점 매칭 앵글 유무 (P07)."""
        if self.pano_path is None:
            self._lbl_title.hide()
            return
        t = self.slider.value() / self.fps
        main = self.pano_path.stem
        covered = []
        if self.mc is not None and self.mc.alts:
            for a in self.mc.alts:
                span = a.get("cover_span")
                if span and span[0] <= t <= span[1]:
                    covered.append(Path(a["video"]).stem)
            if self.mc.alt_on_main:       # 메인이 alt — 제목 교체
                i = self.mc._shown_alt()
                if i is not None:
                    main = Path(self.mc.alts[i]["video"]).stem
                    a = self.mc.alts[i]
                    span = a.get("cover_span")
                    out = span and not (span[0] <= t <= span[1])
                    self._lbl_title.setText(
                        f"{main}" + (" · 이 시점 영상 없음" if out else ""))
                    self._lbl_title.show()
                    self.mc.pane_alt.set_title(self.pano_path.stem)
                    return
        suffix = f" · 앵글: {', '.join(covered)}" if covered else " · 앵글 없음"
        self._lbl_title.setText(main + suffix)
        self._lbl_title.show()
        if self.mc is not None and self.mc.alts:
            i = self.mc._shown_alt()
            if i is not None:
                name = Path(self.mc.alts[i]["video"]).stem
                a = self.mc.alts[i]
                span = a.get("cover_span")
                out = span and not (span[0] <= t <= span[1])
                self.mc.pane_alt.set_title(
                    name + (" · 이 시점 영상 없음" if out else ""))

    def _sync_match_teams(self, quiet=False):
        """같은 경기의 멤버 영상 간 팀 번호/이름/색 정렬 (사용자 방향).

        영상마다 classify_teams 군집 순서가 제멋대로라 전반의 팀1이
        후반의 팀2 가 될 수 있다. 경기 레벨 팀 정체성(match.json
        "teams": 이름+색+등번호)과 현 영상을 대조 — 판정은
        decide_team_mapping (① 등번호 겹침: 카메라 간 색 차이 면역,
        ② 색: 마진 확실할 때만, ③ 애매하면 사용자 확인 대화).
        역할(0/1) 사이드카는 불변, 표시 계층(team_names)만 정렬.
        """
        if self.match is None:
            return
        c0 = self._role_colors.get(0)
        c1 = self._role_colors.get(1)
        if c0 is None or c1 is None:
            return                        # 측정 색 없음 (분석/역할 전)
        nums0 = sorted({n for t, n in self.player_nums.items()
                        if self._role_of(t) in (0, 3)})
        nums1 = sorted({n for t, n in self.player_nums.items()
                        if self._role_of(t) in (1, 4)})
        teams = self.match.get("teams")
        if not teams:                     # 첫 영상 — 정체성 등록
            self.match["teams"] = [
                {"name": self.team_names[0],
                 "color": [int(v) for v in c0], "nums": nums0},
                {"name": self.team_names[1],
                 "color": [int(v) for v in c1], "nums": nums1}]
            self._save_match_doc()
            self.log("[match] 팀 정체성 등록 (이름·색·등번호) — 다른 "
                     "영상/하프에서 자동 정렬됩니다")
            return
        from ..core.match import decide_team_mapping
        verdict = decide_team_mapping(teams, (c0, c1), (nums0, nums1))
        if verdict == "ask":
            if quiet:
                self.log("[match] 팀 매칭 애매 (색 차이 큼·등번호 없음) — "
                         "분석 메뉴 대신 화면에서 확인 필요")
                return
            verdict = self._ask_team_mapping(teams, (c0, c1),
                                             (nums0, nums1))
            if verdict is None:
                return                    # 사용자가 보류
        order = (0, 1) if verdict == "same" else (1, 0)
        want_names = [teams[order[0]]["name"], teams[order[1]]["name"]]
        # 확정된 매핑으로 정체성의 등번호 보강 (다음 영상 판정 재료)
        for side, ns in ((order[0], nums0), (order[1], nums1)):
            merged = sorted(set(teams[side].get("nums") or []) | set(ns))
            if merged != teams[side].get("nums"):
                teams[side]["nums"] = merged
                self._save_match_doc()
        if want_names != list(self.team_names):
            self.team_names = want_names
            self._apply_team_names()
            self._save_keyframes()
            self.log(f"[match] 팀 라벨 정렬: 팀1={want_names[0]} / "
                     f"팀2={want_names[1]}"
                     + (" (반전 감지)" if order == (1, 0) else ""))

    def _save_match_doc(self):
        mp = getattr(self, "match_path", None)
        if mp and self.match:
            from ..core.match import save_match
            try:
                save_match(mp, self.match)
            except OSError as e:
                self.log(f"[match] 저장 실패: {e}")

    def _ask_team_mapping(self, teams, colors, nums):
        """A 영상 팀 정체성 ↔ 현 영상 팀 대응 확인 대화 (스와치+등번호).

        색이 카메라마다 다르게 보일 수 있어 자동 판정이 애매할 때만
        뜬다. 반환 "same"/"flip"/None(보류).
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("팀 매칭 확인")
        v = QVBoxLayout(dlg)

        def sw(bgr):
            lbl = QLabel()
            px = QPixmap(28, 28)
            px.fill(QColor(int(bgr[2]), int(bgr[1]), int(bgr[0])))
            lbl.setPixmap(px)
            return lbl

        v.addWidget(QLabel("카메라가 달라 색이 다르게 보일 수 있습니다 — "
                           "이 영상의 <b>팀1</b>이 경기의 어느 팀인지 "
                           "확인해 주세요."))
        row = QHBoxLayout()
        row.addWidget(QLabel("이 영상 팀1:"))
        row.addWidget(sw(colors[0]))
        row.addWidget(QLabel("등번호 " + (", ".join(nums[0]) or "—")))
        row.addStretch(1)
        row.addWidget(QLabel("팀2:"))
        row.addWidget(sw(colors[1]))
        row.addWidget(QLabel(", ".join(nums[1]) or "—"))
        v.addLayout(row)
        result = []
        for verdict, side in (("same", 0), ("flip", 1)):
            t = teams[side]
            b = QPushButton()
            h = QHBoxLayout(b)
            h.setContentsMargins(10, 6, 10, 6)
            h.addWidget(sw(t["color"]))
            h.addWidget(QLabel(
                f"팀1 = <b>{t['name']}</b>"
                + (f"  (등번호 {', '.join(t.get('nums') or [])})"
                   if t.get("nums") else "")))
            h.addStretch(1)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, vd=verdict:
                              (result.append(vd), dlg.accept()))
            v.addWidget(b)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec()
        return result[0] if result else None

    def _rebuild_mc_bar(self):
        """페인 오버레이의 카메라/모드 바 재구성."""
        while self._mc_row.count():
            it = self._mc_row.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self._mc_row.addStretch(1)
        alts = self.mc.alts if (self.match and self.mc) else []
        if not alts:
            return
        style = ("QPushButton { color: white; background: rgba(20,20,20,150);"
                 " padding: 2px 8px; border-radius: 4px; }"
                 "QPushButton:checked { background: rgba(0,120,215,200); }")
        # 모드가 상위 개념 (좌측), 카메라 선택은 모드 안의 배치 (우측)
        self._mc_mode_btns = {}
        for key, name in (("pip", "PiP"), ("split", "분할"), ("swap", "전환")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setStyleSheet(style)
            b.setChecked(self.mc.mode == key)
            b.clicked.connect(lambda _, k=key: self._mc_mode(k))
            self._mc_row.addWidget(b)
            self._mc_mode_btns[key] = b
        self._mc_row.addSpacing(12)
        self._mc_btns = []
        names = [f"1 {self.pano_path.stem if self.pano_path else '활성'}"] \
            + [f"{i + 2} {Path(a['video']).stem}"
               for i, a in enumerate(alts)]
        for i, name in enumerate(names):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setStyleSheet(style)
            b.setChecked(i == self.mc.focus)
            b.clicked.connect(lambda _, idx=i: self._mc_select(idx))
            self._mc_row.addWidget(b)
            self._mc_btns.append(b)
        b = QPushButton("✎ 앵글 편집")
        b.setStyleSheet(style)
        b.setToolTip("표시 중인 앵글을 단독 영상으로 열기 — 경기장 "
                     "캘리브레이션(rotcam 랜드마크)·검수용. 경기 컨텍스트는 "
                     "해제되며 최근 경기에서 다시 열 수 있음")
        b.clicked.connect(self._mc_edit_alt)
        self._mc_row.addWidget(b)
        self._mc_cam_enable()

    def _mc_cam_enable(self):
        """분할 모드는 둘 다 보이므로 카메라 선택이 무의미 — 비활성."""
        both = self.mc is not None and self.mc.mode == "split"
        for b in getattr(self, "_mc_btns", []):
            b.setEnabled(not both)

    def _mc_edit_alt(self):
        """표시 중인 alt 를 단독으로 열기 — AX700 캘리브레이션 경로 (P07).

        alt 페인은 읽기 전용이라 랜드마크를 찍을 수 없다 — 단독으로
        열면 기존 캘리브레이션 UI 전부 사용 가능. 랜드마크는 그 영상의
        .ptz.json 에 저장되어 rotcam_ref_from_ptz.py 가 읽는다.
        """
        if self.mc is None or not self.mc.alts:
            return
        i = self.mc._shown_alt()
        if i is None:
            i = 0
        video = self.mc.alts[i]["video"]
        self.log(f"[match] 앵글 단독 편집: {Path(video).name} "
                 "(경기 컨텍스트 해제 — 파일 메뉴 > 최근 경기로 복귀)")
        self.open_path(video)

    def _mc_select(self, idx: int):
        """카메라 선택: 0=활성(현재 열린 영상), 1..=다른 앵글.

        PiP = 흘끗 보기(읽기 전용), 전환 = 컨텍스트 스위치 — 그 영상을
        실제로 열어 타임라인·오버레이·편집이 전부 그 카메라 기준이 된다
        (P07 v2, 사용자 방향 2026-07-22). 시간은 시계 모델로 이어진다.
        """
        if self.mc is None or self.match is None:
            return
        alts = self.mc.alts
        if idx > len(alts) or self.mc.mode == "split":
            return
        if self.mc.mode == "swap" and idx > 0:
            from ..core.match import to_alt_time
            a = alts[idx - 1]
            t_here = self.slider.value() / self.fps
            t_there = to_alt_time(a["clock"], t_here)
            self.open_match(self.match, half=self.match_half,
                            cam=a.get("cam", idx), quiet=True,
                            seek_s=max(0.0, t_there))
            self.log(f"[match] 컨텍스트 전환 → {Path(a['video']).stem} "
                     f"(t {t_here:.0f}s → {max(0.0, t_there):.0f}s)")
            return
        self.mc.set_focus(idx)
        for i, b in enumerate(getattr(self, "_mc_btns", [])):
            b.setChecked(i == idx)
        # focus 가 alt 인 PiP 는 파노라마를 안쪽에 즉시 공급
        self.mc.primary_tick(getattr(self, "_cur_frame", None))
        self._update_titles()
        self._redraw()

    def _mc_mode(self, mode: str):
        if self.mc is None:
            return
        self.mc.set_mode(mode)
        for k, b in getattr(self, "_mc_mode_btns", {}).items():
            b.setChecked(k == mode)
        self._mc_cam_enable()
        self.mc.primary_tick(getattr(self, "_cur_frame", None))
        self._update_titles()
        self._redraw()

    def _set_angle_lanes(self):
        """다른 카메라별 타임라인 레인: 커버리지 + 호각 트랙.

        활성 카메라 기준 상대 시계(mc.alts — P07 v2)로 변환하므로 어느
        카메라에 앉아 있어도 맞는다: AX700 활성이면 파노라마가 레인으로.
        현재 영상 범위 밖 구간은 그리기에서 클리핑 (짧은 활성 영상이면
        긴 상대 영상도 그만큼만 보인다).
        """
        from ..core.audio import load_whistle_track
        from ..core.match import alt_coverage, to_primary_time
        angles = []
        alts = self.mc.alts if (self.match and self.mc) else []
        for a in alts:
            cap = cv2.VideoCapture(a["video"])
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT)
                   / (cap.get(cv2.CAP_PROP_FPS) or 30.0)) if cap.isOpened() \
                else 0.0
            cap.release()
            if dur <= 0:
                continue
            t0, t1 = alt_coverage(a["clock"], dur)
            a["cover_span"] = (t0, t1)    # 타이틀 앵글 유무 판정용 (초)
            whistles = []
            try:                              # 그 카메라의 호각 → primary 초
                _tr, ev = load_whistle_track(a["video"])
                whistles = [(to_primary_time(a["clock"], w0),
                             to_primary_time(a["clock"], w1), db)
                            for w0, w1, db in (ev or [])]
            except Exception as e:  # noqa: BLE001 — 호각 없인 밴드만
                self.log(f"[match] {Path(a['video']).name} 호각 트랙 무시: {e}")
            angles.append({"label": Path(a["video"]).stem,
                           "span": (int(t0 * self.fps), int(t1 * self.fps)),
                           "whistles": whistles})
        self.trackbar.set_angles(angles)

    def open_path(self, path: str, quiet: bool = False):
        """파노라마 열기 (프로젝트 복원 경로 포함). 분석/키프레임 사이드카 자동 로드."""
        with self._wait():
            if not self._opening_match and self.match is not None:
                # 단독 파노라마 열기 = 멀티캠 경기 컨텍스트 해제
                self.match = None
                if self.mc is not None:
                    self.mc.set_half([])
                self._rebuild_mc_bar()
                self.trackbar.set_angles([])
            if not Path(path).exists():
                from ..core.project import _cross_platform_candidates
                for cand in _cross_platform_candidates(path):
                    if Path(cand).exists():
                        path = cand
                        break
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                if not quiet:
                    QMessageBox.warning(self, "열기 실패", path)
                else:
                    self.log(f"[ptz] 파노라마 없음 — 건너뜀: {path}")
                return
            if self.cap is not None:
                self.cap.release()
            if self._native_cap is not None:
                self._native_cap.release()
                self._native_cap = None
            self.cap = cap
            self.pane.reset_view()            # 새 영상 = 전체 보기 (줌/팬 리셋)
            self.pano_path = Path(path)
            # 파노라마(.pystitch.json 동반) 아니면 회전 카메라(AX700 등)로
            # 본다 — 필드 캘리브레이션이 프레임 간 랜드마크 이송 + 원근
            # 모델을 쓴다 (P06-3 GUI 통합).
            self._is_rotcam = not self.pano_path.with_suffix(
                ".pystitch.json").exists()
            self._Hcache = {}                 # (src,dst) → 이송 호모그래피
            self._graycache = {}              # frame → det_w 그레이
            self._lm_transferred = {}         # 현재 프레임 이송 랜드마크
            self._remember_dir(str(self.pano_path.parent))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.trackbar.fps = self.fps
            self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.lbl_file.setText(f"{self.pano_path.name} — {self.pano_w}x{self.pano_h}, "
                                  f"{self.total/self.fps/60:.1f}분")
            with self._busy(f"호각 트랙 읽기"):
                try:
                    from ..core.audio import load_whistle_track, \
                        whistle_prominence
                    tr, ev = load_whistle_track(self.pano_path)
                    if tr is not None:
                        self.trackbar.set_whistle(tr["hop_s"],
                                                  whistle_prominence(tr), ev)
                        self.log(f"[ptz] 호각 트랙 로드: 이벤트 {len(ev)}개")
                except Exception as e:  # noqa: BLE001
                    self.log(f"[ptz] 호각 트랙 무시: {e}")
            self.slider.setEnabled(True)
            self.slider.setRange(0, max(0, self.total - 1))
            self.slider.setValue(0)
            self._use_display_source()
            self.btn_analyze.setEnabled(ptz_available())
            if not ptz_available():
                self.btn_analyze.setToolTip("ultralytics 미설치 (pip install ultralytics)")
            self.analysis = None
            self._load_sidecar()
            with self._busy("목록/이벤트 표시 갱신"):
                self._apply_team_names()
                self._refresh_events()
                self._show_frame()
            self._update_export_enabled()

    def _refresh_events(self):
        """자동(킥오프)·사용자 이벤트 + 하이라이트(.events.json) → 타임라인."""
        items = []
        doc = {}
        try:
            from ..core.events import load_events_doc
            doc = load_events_doc(self.pano_path)
        except Exception as e:  # noqa: BLE001
            self.log(f"[ptz] 이벤트 파일 무시: {e}")
        for k in doc.get("kickoffs", []):
            items.append((int(k["t"] * self.fps),
                          "킥오프" + (" ●" if k.get("long_whistle")
                                     else ""), "auto"))
        for f, label in self.user_events:
            items.append((int(f), label, "user"))
        items.sort()
        self.trackbar.set_events(items)
        self._referees = doc.get("referees") or {}
        self._ocr_nums = {int(k): v for k, v in
                          (doc.get("ocr_numbers") or {}).items()}
        self.highlights = doc.get("highlights", [])
        self._refresh_highlight_lane()
        self.trackbar.set_pauses(
            (self.match_info or {}).get("pauses") or [])

    def _refresh_highlight_lane(self):
        self.trackbar.set_highlights(
            [(int(h["t0"] * self.fps), int(h["t1"] * self.fps),
              h.get("state", "cand"), h.get("label", ""))
             for h in self.highlights])

    def _save_highlights(self):
        try:
            from ..core.events import save_events
            save_events(self.pano_path, highlights=self.highlights)
        except Exception as e:  # noqa: BLE001
            self.log(f"[hl] 저장 실패: {e}")
        self._refresh_highlight_lane()

    def _proxy_path(self) -> Path:
        return self.pano_path.with_suffix(".scrub.mp4")

    def _use_display_source(self):
        """표시/재생용 소스 선택: 프록시가 있으면 프록시 (즉시 탐색)."""
        proxy = self._proxy_path()
        if proxy.exists():
            cap = cv2.VideoCapture(str(proxy))
            if cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                self.cap = cap
                self.disp_path = proxy
                pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.disp_scale = pw / max(self.pano_w, 1)
                self.btn_proxy.setText("프록시 사용 중")
                self.btn_proxy.setEnabled(False)
                self.log(f"[ptz] 스크럽 프록시 사용: {proxy.name} (즉시 탐색)")
                return
        self.disp_path = self.pano_path
        self.disp_scale = 1.0
        self.btn_proxy.setText("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(True)

    def _make_proxy(self):
        if self.pano_path is None:
            return
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setText("프록시 생성 중...")
        self.log("[ptz] 스크럽 프록시 생성 중 (NVENC 가용 시 수 분)...")
        w = ProxyWorker(self.pano_path, self._proxy_path(),
                        duration_sec=self.total / max(self.fps, 1e-9))
        w.progress.connect(self._proxy_progress)
        w.finished_ok.connect(lambda _: (self.log("[ptz] 프록시 완료"),
                                         self._use_display_source(),
                                         self._show_frame()))
        w.failed.connect(lambda m: (self.log(f"[오류] 프록시: {m}"),
                                    self.btn_proxy.setText("스크럽 프록시 생성"),
                                    self.btn_proxy.setEnabled(True)))
        self._proxy_worker = w
        w.start()

    def _proxy_progress(self, p: float):
        self.btn_proxy.setText(f"프록시 생성 {p:.0%}")
        # 내보내기가 진행바를 쓰고 있지 않을 때만 진행바에도 표시
        if self._render_worker is None or not self._render_worker.isRunning():
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(p * 1000))
            self.progress.setFormat(f"프록시 생성 %p%")

    def _sidecar_path(self) -> Path:
        """사용자 에디트 사이드카: {keyframes, ignores, promotes}."""
        return self.pano_path.with_suffix(".ptz.json")

    def _analysis_path(self) -> Path:
        """순수 분석 결과 (분석 완료 시 1회 기록, 검수로는 불변)."""
        return self.pano_path.with_suffix(".analysis.json")

    def _kf_path(self) -> Path:                 # 구버전 (마이그레이션용)
        return self.pano_path.with_suffix(".ptz_keyframes.json")

    def _load_sidecar(self):
        """에디트(.ptz.json)와 순수 분석(.analysis.json)을 분리 로드.

        구 통합본(.ptz.json 안에 analysis 포함)은 두 파일로 쪼개 이전한다.
        검수 마킹은 비파괴라 통합본의 analysis 키가 곧 순수 원본이다.
        .analysis.json 이 사이드카보다 새로우면(외부 분석) 그쪽을 채택.
        """
        self.keyframes, self.ignores, self.promotes, self.analysis = \
            [], [], [], None
        self.roles = {}
        self.field_points = {}
        self.field_point_frames = {}
        self.field_point_invalid = {}
        self.field_lines = {}
        self.field_point_anchors = {}
        self.line_points = []
        self._top_black = None                # 상단 검은 경계 캐시 (영상 고정값)
        self.extra_players = {}
        self.kit_colors = {}
        self.user_events = []
        self.match_info = None
        self.merges = {}
        self.player_nums = {}
        self.rosters = {}
        self.hidden_players = set()
        self._outfield = set()
        self._foot_cache = None
        self._pane_sel = set()
        self._rb = None
        self._ar_side_cache = {}
        sp = self._sidecar_path()
        doc = None
        if sp.exists():
            try:
                with self._busy("편집 사이드카 읽기 (.ptz.json)"):
                    doc = json.loads(sp.read_text())
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 사이드카 무시: {e}")
        if doc is not None:
            self.keyframes = [list(k) for k in doc.get("keyframes", [])]
            self.ignores = [list(r) for r in doc.get("ignores", [])]
            self.promotes = [list(p) for p in doc.get("promotes", [])]
            self.roles = {int(t): int(r)
                          for t, r in (doc.get("roles") or {}).items()}
            self.field_points = {k: [float(v[0]), float(v[1])]
                                 for k, v in
                                 (doc.get("field_points") or {}).items()}
            self.field_point_frames = {
                k: int(f) for k, f in
                (doc.get("field_point_frames") or {}).items()}
            self.field_point_invalid = {
                k: [[int(r[0]), int(r[1])] for r in rngs]
                for k, rngs in (doc.get("field_point_invalid") or {}).items()}
            self.field_lines = {
                k: [[float(p[0]), float(p[1]), int(p[2])] for p in pts]
                for k, pts in (doc.get("field_lines") or {}).items()}
            self.field_point_anchors = {
                k: [[float(p[0]), float(p[1]), int(p[2])] for p in pts]
                for k, pts in (doc.get("field_point_anchors") or {}).items()}
            self.line_points = [list(p) for p in doc.get("line_points", [])]
            _tb = doc.get("top_black")
            self._top_black = int(_tb) if _tb is not None else None
            self.extra_players = {int(si): [list(p) for p in rows]
                                  for si, rows in
                                  (doc.get("extra_players") or {}).items()}
            self.kit_colors = {int(r): [int(v) for v in c] for r, c in
                               (doc.get("kit_colors") or {}).items()}
            er = doc.get("export_range")
            self.export_range = ([None if v is None else int(v) for v in er]
                                 if er else [None, None])
            r = self._norm_export_range()
            self.trackbar.set_range(*(r if r else (None, None)))
            tn = doc.get("team_names")
            if tn and len(tn) == 2:
                self.team_names = [str(tn[0]), str(tn[1])]
            self.user_events = [[int(e[0]), str(e[1])]
                                for e in doc.get("user_events", [])]
            self.match_info = doc.get("match_info")
            self.merges = {int(t): int(r) for t, r in
                           (doc.get("merges") or {}).items()}
            self.player_nums = {int(t): str(n) for t, n in
                                (doc.get("player_nums") or {}).items()}
            self.rosters = {int(k): [str(e) for e in v] for k, v in
                            (doc.get("rosters") or {}).items()}
            self.hidden_players = {int(t) for t in
                                   doc.get("hidden_players") or []}
            ids = [int(p[4]) for rows in self.extra_players.values()
                   for p in rows]
            self._next_extra_id = max(ids, default=900000) + 1
            self.field_circle_r = float(doc.get("field_circle_r",
                                                CENTER_CIRCLE_R))
            self.field_circle_auto = bool(doc.get("field_circle_auto", True))
            self.spin_circle_r.blockSignals(True)
            self.spin_circle_r.setValue(self.field_circle_r)
            self.spin_circle_r.blockSignals(False)
            self.chk_circle_auto.blockSignals(True)
            self.chk_circle_auto.setChecked(self.field_circle_auto)
            self.chk_circle_auto.blockSignals(False)
            fs = doc.get("field_size")
            if fs:
                self.field_size = [float(fs[0]), float(fs[1])]
                for sp, v in ((self.spin_field_len, fs[0]),
                              (self.spin_field_w, fs[1])):
                    sp.blockSignals(True)
                    sp.setValue(float(v))
                    sp.blockSignals(False)
            self.analysis = doc.get("analysis")   # 구 통합본 (이전 대상)
        migrated = self.analysis is not None
        ap = self._analysis_path()
        if ap.exists() and (self.analysis is None or
                            ap.stat().st_mtime > sp.stat().st_mtime):
            try:
                with self._busy("분석 읽기 (.analysis.json, 수 MB)"):
                    self.analysis = json.loads(ap.read_text())
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 분석 파일 무시: {e}")
        if doc is None and self._kf_path().exists():
            try:
                d = json.loads(self._kf_path().read_text())
                if isinstance(d, list):
                    self.keyframes = [list(k) for k in d]
                else:
                    self.keyframes = [list(k) for k in d.get("keyframes", [])]
                    self.ignores = [list(r) for r in d.get("ignores", [])]
                migrated = True
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 구형 키프레임 무시: {e}")
        # 분석 유효성 (프레임 수 허용 오차)
        if self.analysis is not None:
            if abs(self.analysis.get("total_frames", 0) - self.total) \
                    > max(60, self.total // 100):
                self.log("[ptz] 사이드카 분석이 다른 영상 기준 — 무시")
                self.analysis = None
        if migrated:
            self._write_analysis()
            self._write_sidecar()
            self.log("[ptz] 사이드카 분리: 분석(.analysis.json) / "
                     "에디트(.ptz.json)")
        if self.analysis is not None:
            self.log(f"[ptz] 분석 불러옴 (검출 샘플 "
                     f"{len(self.analysis['frames'])}개), 키프레임 "
                     f"{len(self.keyframes)}개, 무시 {len(self.ignores)}개")
            self._start_link()
        self._refresh_lists()
        self._refresh_team_label()
        self._refresh_player_list()
        self._refit_field()
        self._refresh_field_list()

    def _write_sidecar(self):
        """사용자 에디트만 저장 — 분석과 분리돼 있어 작고 즉시 쓴다."""
        if self.pano_path is None:
            return
        sp = self._sidecar_path()
        tmp = sp.with_suffix(".ptz.json.tmp")
        tmp.write_text(json.dumps({"keyframes": self.keyframes,
                                   "ignores": self.ignores,
                                   "promotes": self.promotes,
                                   "roles": self.roles,
                                   "field_points": self.field_points,
                                   "field_point_frames":
                                       self.field_point_frames,
                                   "field_point_invalid":
                                       self.field_point_invalid,
                                   "field_lines": self.field_lines,
                                   "field_point_anchors":
                                       self.field_point_anchors,
                                   "top_black": getattr(self, "_top_black",
                                                        None),
                                   "field_size": self.field_size,
                                   "field_circle_r": self.field_circle_r,
                                   "field_circle_auto": self.field_circle_auto,
                                   "line_points": self.line_points,
                                   "extra_players": self.extra_players,
                                   "kit_colors": self.kit_colors,
                                   "export_range": self.export_range,
                                   "team_names": self.team_names,
                                   "user_events": self.user_events,
                                   "match_info": self.match_info,
                                   "merges": {str(t): r for t, r in
                                              self.merges.items()},
                                   "player_nums": {str(t): n for t, n in
                                                   self.player_nums.items()},
                                   "rosters": {str(k): v for k, v in
                                               self.rosters.items()},
                                   "hidden_players":
                                       sorted(self.hidden_players)}))
        tmp.replace(sp)

    def _write_analysis(self):
        """순수 분석 결과 저장 — 분석 완료/이전 시에만, 검수로는 안 바뀜."""
        if self.pano_path is None or self.analysis is None:
            return
        with self._busy("분석 저장 (.analysis.json, 수 MB)"):
            ap = self._analysis_path()
            tmp = Path(str(ap) + ".tmp")
            tmp.write_text(json.dumps(self.analysis))
            tmp.replace(ap)

    _MODEL_NAMES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", None]

    def _model_changed(self, i):
        st = QSettings("PyStitch360", "PyStitch360")
        if self._MODEL_NAMES[i] is None:      # 사용자 .pt
            path, _ = QFileDialog.getOpenFileName(
                self, "YOLO 가중치 (.pt)", "", "PyTorch 가중치 (*.pt)")
            if path:
                st.setValue("ptz_model_custom", path)
            else:
                self.combo_model.setCurrentIndex(0)
                return
        st.setValue("ptz_model", i)

    def _model_weights(self):
        """선택된 모델의 가중치 경로/이름 (None=내장 기본 yolov8n)."""
        name = self._MODEL_NAMES[self.combo_model.currentIndex()]
        if name is None:
            custom = QSettings("PyStitch360", "PyStitch360").value("ptz_model_custom", "")
            return str(custom) if custom else None
        if name == "yolov8n.pt":
            return None                        # presets/yolov8n.pt (내장)
        local = Path(__file__).resolve().parents[2] / "presets" / name
        return str(local) if local.exists() else name   # 없으면 자동 다운로드

    def _run_analyze(self):
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            self._analyze_worker.cancel()          # 버튼이 취소 역할
            self.log("[ptz] 분석 취소 요청...")
            return
        if self.pano_path is None:
            return
        self.btn_analyze.setText("분석 취소")
        self.progress.setRange(0, 0)
        self.progress.setFormat("분석 중... (로그 참조)")
        weights = self._model_weights()
        self.log(f"[ptz] 자동 분석 시작 (모델: {weights or 'yolov8n(내장)'}) — "
                 "진행은 로그에 표시")
        ckpt = str(self.pano_path.with_suffix(".analysis.part.json"))
        w = AnalyzeWorker(str(self.pano_path), weights=weights,
                          checkpoint_path=ckpt)
        w.progress.connect(self._analyze_progress)
        w.log.connect(self.log)
        w.done.connect(self._analyze_done)
        w.failed.connect(self._analyze_failed)
        self._analyze_worker = w
        w.start()

    def _analyze_progress(self, done, total, fps):
        if self._render_worker is not None and self._render_worker.isRunning():
            return                              # 내보내기가 진행바 사용 중
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(
            f"분석 %p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _analyze_done(self, d):
        self.analysis = d
        self._write_analysis()
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("분석 완료")
        n_ball = sum(1 for b in d["balls"] if b is not None)
        self.log(f"[ptz] 분석 완료: 샘플 {len(d['frames'])}개, "
                 f"공 검출 {n_ball}개 ({n_ball/max(len(d['frames']),1):.0%}) → 캐시 저장")
        self._update_export_enabled()
        self._start_link()
        self._show_frame()

    def _analyze_failed(self, msg):
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log(f"[오류] 분석: {msg}")

    # ------------------------------------------------------------ 타임라인/표시
    def _step(self, d):
        self._stop_play()
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    @staticmethod
    def _hms(sec, tenth=False):
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return (f"{int(h)}:{int(m):02d}:{s:04.1f}" if tenth
                else f"{int(h)}:{int(m):02d}:{int(s):02d}")

    @contextmanager
    def _wait(self):
        """웨이트 커서만 (로그 없이) — 파일 열기 전체 흐름 감싸기용.
        Qt 오버라이드 커서는 스택이라 _busy 와 중첩해도 안전."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()

    @contextmanager
    def _busy(self, msg):
        """동기(UI 차단) 작업 래퍼: 시작 로그 + 웨이트 커서 + 완료 로그."""
        self.log(f"[작업] {msg}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()          # 커서·로그 즉시 반영
        t0 = time.perf_counter()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            el = time.perf_counter() - t0
            if el >= 0.5:                     # 짧은 작업은 완료 로그 생략
                self.log(f"[작업] {msg} 완료 ({el:.1f}s)")

    def _on_slider(self, _):
        # 재생 중 사용자 시크(워커 갱신이 아님) → 그 지점부터 재생 계속
        if self._playing and not getattr(self, "_play_sync", False):
            self._restart_play_at(self.slider.value())
            return
        self.trackbar.set_pos(self.slider.value())
        t = self.slider.value() / self.fps
        self.lbl_time.setText(f"{self._hms(t, tenth=True)} / "
                              f"{self._hms(self.total / self.fps)}")
        self._slider_timer.start()

    def _restart_play_at(self, f):
        """재생 중 시크: 이전 워커를 버리고 f 부터 재생 재시작."""
        w_old = self._play_worker
        if w_old is not None:
            try:
                w_old.frame_ready.disconnect()
                w_old.finished.disconnect()
            except TypeError:
                pass
            w_old.stop()                  # 남은 프레임 방출은 무시됨
        self._playing = False
        self._toggle_play()               # slider 값(=f)부터 새로 시작

    def _toggle_mute(self):
        muted = self.btn_mute.isChecked()
        self.btn_mute.setText("🔇" if muted else "🔊")
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_muted", "true" if muted else "false")
        if self._audio:
            self._audio_out.setMuted(muted)

    def _ensure_audio(self):
        """오디오 플레이어 지연 초기화 — 백엔드 없으면 조용히 무음."""
        if self._audio is not None:
            return self._audio or None
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            self._audio_out = QAudioOutput()
            self._audio_out.setMuted(self.btn_mute.isChecked())
            p = QMediaPlayer()
            p.setAudioOutput(self._audio_out)
            self._audio = p
        except Exception as e:  # noqa: BLE001
            self.log(f"[ptz] 오디오 재생 불가 (QtMultimedia): {e}")
            self._audio = False
        return self._audio or None

    def _toggle_play(self):
        self._play_until = None           # 수동 조작은 구간 재생 해제
        if self._playing:
            self._stop_play()
            return
        if self.cap is None:
            return
        # 재생 표시 폭 상한 (QSettings ptz_play_w, 기본 1600) — 원본이 이미
        # 작으면(프록시) 그대로. 화질을 약간 낮춰 재생 부하를 줄인다.
        play_w = int(QSettings("PyStitch360", "PyStitch360")
                     .value("ptz_play_w", 1600))
        # 재생 소스: 프록시(.scrub.mp4)가 있으면 그것 (원본 5900px 디코드
        # 회피). 없으면 원본을 읽으며 disp_w 로 온더플라이 축소.
        proxy = self._proxy_path()
        src = str(proxy) if proxy.exists() else (self.disp_path
                                                 or self.pano_path)
        w = PtzPlayWorker(src, self.slider.value(), self.fps,
                          display_fps=30.0 if self.disp_scale < 1.0 else 20.0,
                          disp_w=play_w)
        w.frame_ready.connect(self._play_frame)
        w.finished.connect(self._play_finished)
        self._play_worker = w
        self._playing = True
        self.btn_play.setText("⏸")
        a = self._ensure_audio()
        if a is not None:
            from PyQt6.QtCore import QUrl
            url = QUrl.fromLocalFile(str(self.pano_path))
            if a.source() != url:
                a.setSource(url)      # 오디오는 항상 원본에서 (프록시 무관)
            a.setPosition(int(self.slider.value() / self.fps * 1000))
            a.play()
        w.start()

    def _restore_still(self):
        self._play_scale = None
        self._show_frame()                    # 원해상도 정지 프레임 재표시

    def _stop_play(self):
        if self._play_worker is not None and self._play_worker.isRunning():
            self._play_worker.stop()
        if self._audio:
            self._audio.pause()

    def closeEvent(self, ev):
        if self._native_worker is not None:
            self._native_worker.stop()
            self._native_worker = None
        super().closeEvent(ev)

    def _play_frame(self, frame, f, disp_scale=1.0):
        self._cur_frame, self._cur_frame_idx = frame, f
        # 재생 중 표시 스케일 — _redraw 가 오버레이 좌표에 반영
        self._play_scale = self.disp_scale * disp_scale
        self._cur_scale = self._play_scale   # 정지 후 리드로 대비 일관성
        if self.mc is not None:
            self.mc.update(f / self.fps, playing=True)
            self.mc.primary_tick(frame, playing=True)
        if f % 15 == 0:
            self._update_titles()
        self._play_sync = True      # 워커 발 갱신 표시 — 사용자 시크와 구분
        try:
            self.slider.setValue(f)  # _show_frame 은 재생 중 가드로 무시됨
        finally:
            self._play_sync = False
        # 오디오 드리프트 보정 (±0.3s 넘으면 재동기)
        if self._audio and f % 150 < 5:
            want = int(f / self.fps * 1000)
            if abs(self._audio.position() - want) > 300:
                self._audio.setPosition(want)
        self._redraw()
        if self._play_until is not None and f >= self._play_until:
            self._play_until = None       # 구간 끝 — 자동 정지
            self._stop_play()

    def _play_segment(self, f0, f1):
        """구간 재생: f0 부터 f1 에서 자동 정지 (하이라이트/마커 검수용)."""
        if self.cap is None or f1 <= f0:
            return
        if not self._playing:
            self.slider.setValue(int(f0))
            self._toggle_play()
        else:
            self.slider.setValue(int(f0))   # 재생 중 시크 → 자동 재시작
        self._play_until = int(f1)
        self.log(f"[ptz] 구간 재생 {self._hms(f0 / self.fps)}~"
                 f"{self._hms(f1 / self.fps)}")

    def _play_finished(self):
        self._playing = False
        self.btn_play.setText("▶")
        if self._audio:
            self._audio.pause()
        self._restore_still()             # 정지 = 원해상도 정지 프레임

    def _current_sample(self):
        """현재 시각 근처(±0.5s)의 분석 샘플 인덱스."""
        if self.analysis is None:
            return None
        idx = np.asarray(self.analysis["frames"])
        i = int(np.argmin(np.abs(idx - self.slider.value())))
        if abs(idx[i] - self.slider.value()) > 0.5 * self.fps:
            return None
        return i

    def _current_auto_ball(self):
        i = self._current_sample()
        return None if i is None else self.analysis["balls"][i]

    def _ball_in_ignore(self, f, bx, by):
        """(f, bx, by) 공이 어떤 무시 구간에 걸리는가 (자리 있으면 150px)."""
        return any(
            r[0] <= f <= r[1] and (len(r) < 4 or
                                   ((r[2] - bx) ** 2 + (r[3] - by) ** 2)
                                   ** 0.5 <= 150)
            for r in self.ignores)

    def _promote_near(self, f, bx, by):
        """(f, bx, by) 부근(±0.5s, 150px)에 걸린 승격 항목들 (토글/표시용)."""
        return [p for p in self.promotes
                if abs(p[0] - f) <= 0.5 * self.fps
                and (p[1] - bx) ** 2 + (p[2] - by) ** 2 <= 150 ** 2]

    # ---------------------------------------------------- 오브젝트 질의/조작
    def _candidates_at(self, si):
        """샘플 si 의 공 후보 [(x, y, conf), ...] (신형 ball_cands 우선)."""
        if self.analysis is None or si is None:
            return []
        bc = self.analysis.get("ball_cands")
        if bc is not None and si < len(bc):
            return [(float(p[0]), float(p[1]), float(p[2])) for p in bc[si]]
        b = self.analysis["balls"][si]
        return [(float(b[0]), float(b[1]), float(b[2]))] if b else []

    def _track_for(self, si, x, y):
        """샘플 si 에서 (x, y) 를 지나는 링크 트랙 (없으면 None)."""
        if self._linked is None or si is None:
            return None
        best, bestd = None, 80.0 ** 2
        for t in self._linked["tracks"]:
            hits = np.where(t["i"] == si)[0]
            if len(hits) == 0:
                continue
            k = int(hits[0])
            d = (t["pts"][k][0] - x) ** 2 + (t["pts"][k][1] - y) ** 2
            if d <= bestd:
                best, bestd = t, d
        return best

    def _track_span(self, t):
        idx = self._linked["idx"]
        return int(idx[t["i"][0]]), int(idx[t["i"][-1]])

    def _cand_state(self, f, si, x, y):
        """공 후보 상태: 'ignored' | 'promoted' | 'accepted' | 'rejected'."""
        if self._ball_in_ignore(f, x, y):
            return "ignored"
        if self._promote_near(f, x, y):
            return "promoted"
        ab = self._accepted_ball
        if (ab is not None and si is not None and si < len(ab)
                and not np.isnan(ab[si, 0])
                and (ab[si, 0] - x) ** 2 + (ab[si, 1] - y) ** 2 <= 60.0 ** 2):
            return "accepted"
        return "rejected"

    def _objects_at(self):
        """현재 프레임의 조작 가능한 오브젝트 (공 후보 + 근처 키프레임)."""
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        si = self._current_sample()
        objs = []
        for (x, y, conf) in self._candidates_at(si):
            objs.append({"kind": "ball", "x": x, "y": y, "conf": conf,
                         "state": self._cand_state(f, si, x, y), "si": si})
        for i, k in enumerate(self.keyframes):
            if abs(k[0] - f) <= 1.0 * self.fps:
                objs.append({"kind": "kf", "x": float(k[1]), "y": float(k[2]),
                             "i": i})
        return objs

    def _hit(self, x, y, r=80.0):
        """(x, y) 파노라마 좌표에서 반경 r 안 가장 가까운 오브젝트 (없으면 None)."""
        best, bestd = None, r * r
        for o in self._objects_at():
            d = (o["x"] - x) ** 2 + (o["y"] - y) ** 2
            if d <= bestd:
                best, bestd = o, d
        return best

    def _ignore_covers(self, entry):
        """entry(=[f0,f1,x,y]) 와 사실상 동일한 무시가 이미 있는가."""
        f0, f1 = entry[0], entry[1]
        for r in self.ignores:
            if r[0] <= f0 and f1 <= r[1] and (
                    len(r) < 4 or ((r[2] - entry[2]) ** 2
                                   + (r[3] - entry[3]) ** 2) ** 0.5 <= 150):
                return True
        return False

    def _mark_dirty_and_redraw(self):
        self._save_keyframes()
        self._recompute_tracks()
        self._redraw()

    def _promote_ball(self, f, si, x, y):
        """(x,y) 공을 경기 공으로 승격 + 같은 프레임 다른 후보 트랙 자동 무시."""
        if not self._promote_near(f, x, y):
            self.promotes.append([f, round(x, 1), round(y, 1)])
            self.promotes.sort()
        auto = 0
        for (cx, cy, cc) in self._candidates_at(si):
            if (cx - x) ** 2 + (cy - y) ** 2 <= 60.0 ** 2:
                continue                       # 승격 대상 자신
            t = self._track_for(si, cx, cy)
            if t is None:
                continue
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
            # 승격한 공을 함께 잡을 수 있는 무시는 건너뜀 (자기 트랙 보호)
            if (f0 <= f <= f1 and
                    ((med[0] - x) ** 2 + (med[1] - y) ** 2) ** 0.5 <= 150):
                continue
            if not self._ignore_covers(entry):
                self.ignores.append(entry)
                auto += 1
        self.ignores.sort()
        self._mark_dirty_and_redraw()
        extra = f", 경쟁 후보 {auto}개 자동 무시" if auto else ""
        self.log(f"[ptz] 경기 공 승격 {f/self.fps:.1f}s "
                 f"→ ({x:.0f}, {y:.0f}){extra}")

    def _unpromote_at(self, f, x, y):
        near = self._promote_near(f, x, y)
        if not near:
            return
        for p in near:
            self.promotes.remove(p)
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 승격 취소 {f/self.fps:.1f}s")

    def _ignore_track_at(self, f, si, x, y):
        """(x,y) 후보의 트랙을 오인식으로 무시 (링크 없으면 그 시점만)."""
        t = self._track_for(si, x, y)
        if t is not None:
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
        else:
            entry = [int(f), int(f), round(x, 1), round(y, 1)]
        for p in self._promote_near(f, x, y):    # 무시가 승격보다 우선
            self.promotes.remove(p)
        if not self._ignore_covers(entry):
            self.ignores.append(entry)
            self.ignores.sort()
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 검출 무시 {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _restore_at(self, f, x, y):
        """(x,y) 위치에 걸린 무시 구간을 해제 (복원)."""
        keep, removed = [], 0
        for r in self.ignores:
            if (r[0] <= f <= r[1] and
                    (len(r) < 4 or ((r[2] - x) ** 2 + (r[3] - y) ** 2)
                     ** 0.5 <= 150)):
                removed += 1
            else:
                keep.append(r)
        if removed:
            self.ignores = keep
            self._mark_dirty_and_redraw()
            self.log(f"[ptz] 무시 해제 {f/self.fps:.1f}s ({removed}개 구간)")

    def _set_click_mode(self, ball: bool):
        self.btn_mode_ball.setChecked(ball)
        self.btn_mode_kf.setChecked(not ball)

    def _add_keyframe(self, f, x, y, width=None):
        """빈 곳 클릭 마크: 3요소 = 공(줌 자동), 4요소 = 크롭 키프레임."""
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        entry = [f, round(x, 1), round(y, 1)]
        if width is not None:
            entry.append(round(float(width), 1))
        self.keyframes.append(entry)
        self.keyframes.sort()
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()
        kind = "키프레임" if width is not None else "수동 공"
        self.log(f"[ptz] {kind} {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _toggle_kf_type(self, i):
        """공(3요소) ↔ 크롭 키프레임(4요소, 현재 계획 폭) 전환."""
        if not (0 <= i < len(self.keyframes)):
            return
        k = self.keyframes[i]
        if len(k) > 3:
            self.keyframes[i] = k[:3]
            self.log(f"[ptz] {k[0]/self.fps:.1f}s 키프레임 → 공 (줌 자동)")
        else:
            f = int(k[0])
            if self.plan is None or f >= len(self.plan["crop_w"]):
                self.log("[ptz] 계획이 아직 없어 폭을 정할 수 없음")
                return
            self.keyframes[i] = [k[0], k[1], k[2],
                                 round(float(self.plan["crop_w"][f]), 1)]
            self.log(f"[ptz] {k[0]/self.fps:.1f}s 공 → 키프레임 "
                     f"(폭 {self.keyframes[i][3]:.0f} 고정)")
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()

    def _delete_keyframe_idx(self, i):
        if 0 <= i < len(self.keyframes):
            k = self.keyframes[i]
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_kf_list()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes,
                                   promotes=self.promotes)
            self._plan_dirty()
            self._redraw()
            self.log(f"[ptz] 키프레임 삭제 {k[0]/self.fps:.1f}s")

    # ------------------------------------------------- 크롭 박스 드래그 편집
    def _disp_to_pano(self):
        """표시 픽셀 1개가 파노라마 몇 px 인가 (히트 톨러런스 환산)."""
        return self.pano_w / max(self.pane.displayed_width(), 1)

    def _box_hit(self, x, y):
        """크롭 박스 히트: ("corner", 0~3) | ("border", None) | None.

        모서리(핸들) = 리사이즈(줌), 테두리 밴드 = 이동. 내부는 공 클릭용으로
        비워 둔다 (공은 대개 박스 안에 있다).
        """
        if self._plan_box is None:
            return None
        x0, y0, w, h = self._plan_box
        k = self._disp_to_pano()
        ct = 18 * k                          # 모서리 히트 반경
        for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                       (x0, y0 + h), (x0 + w, y0 + h)]):
            if abs(x - hx) <= ct and abs(y - hy) <= ct:
                return ("corner", ci)
        bt = 10 * k                          # 테두리 밴드 폭
        on_v = (abs(x - x0) <= bt or abs(x - x0 - w) <= bt) \
            and y0 - bt <= y <= y0 + h + bt
        on_h = (abs(y - y0) <= bt or abs(y - y0 - h) <= bt) \
            and x0 - bt <= x <= x0 + w + bt
        if on_v or on_h:
            return ("border", None)
        return None

    def _landmark_at(self, x, y, r=60.0):
        """(x, y) 근처의 랜드마크 키 (반경 r px, 없으면 None).

        rotcam 모드에선 마커가 이송 위치에 그려지므로 이송 위치로 매칭
        (엉뚱하게 이송된 마커를 우클릭·드래그로 집을 수 있게)."""
        best, bestd = None, r * r
        rc = getattr(self, "_is_rotcam", False)
        tr = getattr(self, "_lm_transferred", {})
        for k, p in self.field_points.items():
            q = (tr.get(k) if rc else None) or p
            d = (q[0] - x) ** 2 + (q[1] - y) ** 2
            if d <= bestd:
                best, bestd = k, d
        return best

    def _pane_pressed(self, fx, fy):
        """좌버튼 프레스: 랜드마크(경기장 탭) 또는 크롭 박스 편집 시작."""
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        x, y = fx * self.pano_w, fy * self.pano_h
        if self._field_tab_active() or self.btn_field_pick.isChecked():
            self._lm_drag = self._landmark_at(x, y)   # 마커 잡으면 드래그 이동
            if self._lm_drag is not None:
                return
        if self.btn_field_pick.isChecked():   # 찍기 모드: 박스 편집 비활성
            return
        if self.analysis is None:
            return
        if self._hit(x, y) is not None:      # 공/키프레임이 우선 (클릭 동작)
            return
        hit = self._box_hit(x, y) if self.plan is not None else None
        if hit is None or self._plan_box is None:
            # 빈 곳(또는 크롭 박스 밖) 좌드래그 = 사람 다중 선택 러버밴드.
            # Shift = 기존 선택에 추가(합집합), 없으면 새 선택.
            shift = bool(QApplication.keyboardModifiers()
                         & Qt.KeyboardModifier.ShiftModifier)
            self._rb = {"x0": x, "y0": y, "x1": x, "y1": y, "shift": shift}
            return
        x0, y0, w, h = self._plan_box
        box = [x0 + w / 2, y0 + h / 2, float(w)]
        corners = [(x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h)]
        self._box_edit = {"mode": hit[0], "corner": hit[1],
                          "box": list(box), "orig": list(box),
                          # 리사이즈 앵커 = 잡은 코너의 대각 반대편 (고정점)
                          "anchor": corners[3 - hit[1]] if hit[0] == "corner"
                          else None,
                          "start": (x, y),
                          "frame": getattr(self, "_cur_frame_idx",
                                           self.slider.value())}

    def _pane_dragged(self, fx, fy):
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self._rb is not None:             # 러버밴드 확장 중
            self._rb["x1"] = fx * self.pano_w
            self._rb["y1"] = fy * self.pano_h
            self._redraw()
            return
        if self._lm_drag is not None:        # 랜드마크 이동 (미세조정)
            nx = round(fx * self.pano_w, 1)
            ny = round(fy * self.pano_h, 1)
            self.field_points[self._lm_drag] = [nx, ny]
            self.field_point_frames[self._lm_drag] = int(self.slider.value())
            # 라이브 마커만 이동 — 무거운 재피팅은 놓을 때(_pane_released).
            # 매 픽셀 재캘리브레이션은 미세조정을 렉걸리게 했다.
            self._lm_transferred[self._lm_drag] = (nx, ny)
            self._redraw()
            return
        e = self._box_edit
        if e is None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        ow, oh = self.plan_out
        cx0, cy0, w0 = e["orig"]
        top = int(self.plan.get("top_margin", 0)) if self.plan else 0
        max_w = min(self.pano_w, (self.pano_h - top) * ow / oh)
        if e["mode"] == "border":            # 이동
            cx = cx0 + (x - e["start"][0])
            cy = cy0 + (y - e["start"][1])
            w = min(max(w0, ow / 6.0), max_w)
            h = w * oh / ow
            cx = min(max(cx, w / 2), self.pano_w - w / 2)
            cy = min(max(cy, h / 2), self.pano_h - h / 2)
        else:                                # 모서리 = 반대편 코너 고정 리사이즈
            ax, ay = e["anchor"]
            sx = -1 if e["corner"] in (0, 2) else 1   # 왼쪽 코너 = 앵커의 왼쪽으로
            sy = -1 if e["corner"] in (0, 1) else 1   # 위 코너 = 앵커의 위로
            w = max(abs(x - ax), abs(y - ay) * ow / oh)
            # 앵커를 고정한 채 파노라마 안에 들어가는 최대 폭
            max_w = min(max_w,
                        (self.pano_w - ax) if sx > 0 else ax,
                        ((self.pano_h - ay) if sy > 0 else ay) * ow / oh)
            w = min(max(w, ow / 6.0), max_w)
            h = w * oh / ow
            cx, cy = ax + sx * w / 2, ay + sy * h / 2
        e["box"] = [cx, cy, w]
        self._redraw()

    def _pane_released(self, fx, fy):
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self._rb is not None:
            rb, self._rb = self._rb, None
            x1, x2 = sorted((rb["x0"], rb.get("x1", rb["x0"])))
            y1, y2 = sorted((rb["y0"], rb.get("y1", rb["y0"])))
            if (x2 - x1) < 6 or (y2 - y1) < 6:   # 사실상 클릭
                if not rb.get("shift"):          # shift 없으면 선택 해제
                    self._pane_sel = set()
            else:
                hits = self._players_in_rect(x1, y1, x2, y2)
                self._pane_sel = (self._pane_sel | hits if rb.get("shift")
                                  else hits)
                self._sync_list_to_pane_sel()
                self.log(f"[ptz] 드래그 선택: 선수 {len(self._pane_sel)}명")
            self._redraw()
            return
        if self._lm_drag is not None:
            key, self._lm_drag = self._lm_drag, None
            self._refit_field(log_result=True)
            self._save_keyframes()
            self._refresh_field_list()
            self._redraw()
            return
        e = self._box_edit
        if e is None:
            return
        self._box_edit = None
        cx, cy, w = e["box"]
        cx0, cy0, w0 = e["orig"]
        # 사실상 클릭(수 px 미만 이동)이면 커밋하지 않음
        if abs(cx - cx0) + abs(cy - cy0) + abs(w - w0) < 6 * self._disp_to_pano():
            self._redraw()
            return
        f = e["frame"]
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        self.keyframes.append([f, round(cx, 1), round(cy, 1), round(w, 1)])
        self.keyframes.sort()
        # 새 계획이 올 때까지 이 위치를 그대로 그린다 — 이전 박스 깜박임 방지
        self._box_commit = (f, cx, cy, w)
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] 크롭 키프레임 {f/self.fps:.1f}s → "
                 f"중심 ({cx:.0f}, {cy:.0f}), 폭 {w:.0f}px")

    def _show_frame(self):
        """현재 프레임 디코딩 + 오버레이. 디코딩 결과는 캐시된다."""
        if self.cap is None or self._playing:
            return
        f = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._cur_frame, self._cur_frame_idx = frame, f
        self._cur_scale = self.disp_scale
        if self.mc is not None:
            self.mc.update(f / self.fps, playing=False)
            self.mc.primary_tick(frame)
        self._update_titles()
        if getattr(self, "_is_rotcam", False) and self.field_points:
            with self._wait():            # 랜드마크 이송(호모그래피)은 무거움
                self._rc_transfer_landmarks()
        self._redraw()
        # 프록시로 즉시 표시했으니, 원본(파노라마) 해상도를 백그라운드로
        # 읽어 교체 (스크럽은 프록시로 빠르게, 멈추면 선명한 원본으로).
        # 멀티캠 경기에서도 메인 페인이 주 카메라(파노라마)를 보일 때는
        # 승격한다 — alt 카메라를 메인에 띄운 동안만 제외.
        alt_main = self.mc is not None and self.mc.alt_on_main
        if self.disp_scale < 1.0 and not alt_main:
            self._request_native(f)

    def _request_native(self, f):
        if self.pano_path is None:
            return
        if self._native_worker is None:
            self._native_worker = NativeFrameWorker()
            self._native_worker.frame_ready.connect(self._native_frame_ready)
            self._native_worker.start()
        self._native_pending = int(f)
        self._native_worker.request(self.pano_path, f, f / max(self.fps, 1e-9),
                                    self.pano_w, self.pano_h)
        self.log(f"[ptz] 원본 승격 요청: 프레임 {f} "
                 f"({self.pano_w}x{self.pano_h}) 백그라운드 디코드…")

    def _native_frame_ready(self, frame, f):
        """백그라운드 원본 프레임 도착 — 아직 그 프레임이면 선명하게 교체.
        frame 이 None 이면 디코드 실패 — 대기 표시만 해제하고 SCRUB 유지."""
        was_pending = getattr(self, "_native_pending", None) == f
        if was_pending:
            self._native_pending = None
        if self._playing or self.slider.value() != f \
                or getattr(self, "_cur_frame_idx", -1) != f:
            self.log(f"[ptz] 원본 승격 결과 폐기: 프레임 {f} "
                     "(그새 다른 프레임/재생으로 이동)")
            return
        if frame is None:
            if was_pending:
                self.log("[ptz] 원본 프레임 읽기 실패/시간초과 — SCRUB 유지")
                self._redraw()            # 배지 loading→SCRUB
            return
        self._cur_frame, self._cur_scale = frame, 1.0
        self.log(f"[ptz] 원본 승격 완료: 프레임 {f} → ORIG")
        self._redraw()

    def _redraw(self):
        """오버레이만 다시 그림 — 키프레임/무시/계획 변경 시 디코딩 없이 즉시."""
        if not self.check_radar.isChecked() or self.analysis is None:
            self._hide_radar()            # 조건 미충족 잔상 방지
        if self.mc is not None and self.mc.alt_on_main:
            sf = self.mc.main_frame()     # focus 카메라(alt) 원본 표시
            if sf is not None:
                self.pane.set_frame(sf)
                return
        if getattr(self, "_cur_frame", None) is None:
            return
        frame = self._cur_frame.copy()
        f = self._cur_frame_idx
        # 재생 중엔 다운스케일된 재생 프레임 → 그 스케일로 오버레이 좌표
        sc = (self._play_scale if self._playing
              and getattr(self, "_play_scale", None) is not None
              else self._cur_scale)
        # 랜드마크 찍기 모드: 크롭 박스/공/키프레임/선수 오버레이 숨김 —
        # 경기장 선·마커만 보이게 해서 클릭 대상이 헷갈리지 않게 한다.
        picking = self.btn_field_pick.isChecked()
        # 계획된 크롭 창 (render_plan 과 동일한 클램프) — 결과 미리보기.
        # 드래그(이동)/모서리 핸들(줌)로 편집 가능 — 놓으면 줌 키프레임 커밋.
        self._plan_box = None
        if self.plan is not None and not picking \
                and self.check_crop.isChecked() and f < len(self.plan["cx"]):
            ow, oh = self.plan_out
            tl = self.plan.get("top_lim")
            top = (int(tl[f]) if tl is not None and f < len(tl)
                   else int(self.plan.get("top_margin", 0)))
            commit = self._box_commit
            if self._box_edit is not None:      # 편집 중: 미확정 박스
                bcx, bcy, bw = self._box_edit["box"]
                w = int(round(bw)) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(bcx - w / 2))
                y0 = int(round(bcy - h / 2))
                box_color = (0, 255, 255)
            elif commit is not None and abs(f - commit[0]) <= 0.5 * self.fps:
                # 커밋 직후 (새 계획 계산 중): 놓은 자리 그대로 유지
                w = int(round(commit[3])) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(commit[1] - w / 2))
                y0 = int(round(commit[2] - h / 2))
                box_color = (255, 200, 0)
            else:
                w = int(round(min(self.plan["crop_w"][f], self.pano_w,
                                  self.pano_h * ow / oh))) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(self.plan["cx"][f] - w / 2))
                y0 = int(round(self.plan["cy"][f] - h / 2))
                x0 = max(0, min(x0, self.pano_w - w))
                y0 = max(min(top, self.pano_h - h), min(y0, self.pano_h - h))
                box_color = (255, 200, 0)
            self._plan_box = (x0, y0, w, h)
            thick = max(2, int(6 * sc))
            if self._box_edit is None and self._box_hover == ("border", None):
                thick += max(1, int(2 * sc))    # 이동 가능 표시: 테두리 두껍게
            cv2.rectangle(frame, (int(x0 * sc), int(y0 * sc)),
                          (int((x0 + w) * sc), int((y0 + h) * sc)),
                          box_color, thick)
            hs = max(6, int(16 * sc))           # 모서리 핸들 (리사이즈 = 줌)
            for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                           (x0, y0 + h), (x0 + w, y0 + h)]):
                hov = (self._box_edit is None
                       and self._box_hover == ("corner", ci))
                r = hs + (max(3, int(8 * sc)) if hov else 0)
                cv2.rectangle(frame, (int(hx * sc) - r, int(hy * sc) - r),
                              (int(hx * sc) + r, int(hy * sc) + r),
                              (0, 255, 255) if hov else box_color, -1)
        # 공 후보 전부 표시 — 상태별 색: 초록=수락, 자홍=승격, 빨강X=무시,
        # 회색=자동 기각. 커서가 가리키는 오브젝트는 노란 링으로 하이라이트.
        si = self._current_sample()
        hv = self._hover_key[0] if self._hover_key else None

        def _is_hover(kind, ox, oy):
            return hv is not None and hv == (kind, round(ox), round(oy))

        show_ball = self.check_ball.isChecked()
        for (bx, by, conf) in ([] if picking or not show_ball
                               else self._candidates_at(si)):
            rad = self._ball_rad(bx, by, sc)
            p = (int(bx * sc), int(by * sc))
            st = self._cand_state(f, si, bx, by)
            if st == "ignored":
                # 공이 아님 = 회색 점선 테두리만 (글씨 없이). 세부 사유는
                # 커서 올리면 툴팁으로 (_pane_tooltip).
                _dashed_circle(frame, p, rad + max(3, int(6 * sc)),
                               (150, 150, 150), max(1, int(2 * sc)))
            elif st == "promoted":
                cv2.circle(frame, p, rad, (200, 0, 255), 6)
            elif st == "accepted":
                cv2.circle(frame, p, rad, (0, 220, 0), 6)
            else:
                cv2.circle(frame, p, rad, (150, 150, 150), 4)
            if _is_hover("ball", bx, by):
                cv2.circle(frame, p, rad + max(6, int(12 * sc)), (0, 255, 255), 2)
        for k in ([] if picking else self.keyframes):
            if len(k) > 3 and not self.check_crop.isChecked():
                continue                      # 크롭 키프레임은 크롭 토글에
            if len(k) <= 3 and not show_ball:
                continue                      # 수동 공은 공 토글에
            if abs(k[0] - f) <= 1.0 * self.fps:
                kx, ky = float(k[1]), float(k[2])
                p = (int(kx * sc), int(ky * sc))
                th = max(2, int(6 * sc))
                if len(k) > 3:
                    # 크롭 키프레임: 동서남북 4방향 화살표 (이동+줌 앵커)
                    arm = max(12, int(34 * sc))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        cv2.arrowedLine(frame, p,
                                        (p[0] + dx * arm, p[1] + dy * arm),
                                        (0, 0, 255), th, tipLength=0.45)
                else:
                    # 수동 공: 주황 이중 링 (공 후보 원과 구별)
                    r0 = self._ball_rad(kx, ky, sc) + 2
                    cv2.circle(frame, p, r0, (0, 140, 255), th)
                    cv2.circle(frame, p, max(2, r0 // 3),
                               (0, 140, 255), -1)
                if _is_hover("kf", kx, ky):
                    cv2.circle(frame, p, max(16, int(40 * sc)), (0, 255, 255), 2)
        # 선수 박스(팀 색) + 레이더
        radar_pts = []
        if si is not None:
            prow = self._players_row(si)
            # rotcam 장외 숨김: 트랙릿 일괄 판정(파노라마) 대신 현재
            # 프레임 자세로 발밑을 지면 투영해 피치 밖을 거른다 — 자세를
            # 못 푼 프레임은 필터 없이 전부 표시 (엉뚱하게 숨기느니 안전).
            if (getattr(self, "_is_rotcam", False)
                    and getattr(self, "_rc_calib", None) is not None
                    and self.check_infield.isChecked() and prow):
                pose = self._rc_current_pose(int(f))
                if pose is not None:
                    from ..core.rotcam import pixel_to_field
                    ok_rows = [pp for pp in prow if len(pp) >= 4]
                    if ok_rows:
                        fxy = pixel_to_field(
                            pose["K"], pose["R"],
                            np.asarray(self._rc_calib["cam_pos"], float),
                            [(pp[0], pp[1] + pp[3] / 2.0) for pp in ok_rows])
                        hl = self.field_size[0] / 2.0 + self._INFIELD_MARGIN
                        hw = self.field_size[1] / 2.0 + self._INFIELD_MARGIN
                        drop = {id(pp) for pp, (gx, gy) in zip(ok_rows, fxy)
                                if not (np.isfinite(gx) and abs(gx) <= hl
                                        and abs(gy) <= hw)}
                        prow = [pp for pp in prow if id(pp) not in drop]
            sel = self.trackbar.selected
            sel_rep = (self._rep(sel[1]) if sel and sel[0] == "player"
                       else None)
            if self.check_players.isChecked() and not picking:
                for pp in prow:
                    if len(pp) < 4:
                        continue
                    pid = int(pp[4]) if len(pp) >= 5 else None
                    team = self._role_of(pid) if pid is not None else 2
                    color = self._role_color(team)
                    x1 = int((pp[0] - pp[2] / 2) * sc)
                    y1 = int((pp[1] - pp[3] / 2) * sc)
                    x2 = int((pp[0] + pp[2] / 2) * sc)
                    y2 = int((pp[1] + pp[3] / 2) * sc)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                                  max(1, int(2 * sc)))
                    if pid is not None and pid >= 0 \
                            and self._rep(pid) in self._pane_sel:
                        # 드래그 다중 선택: 청록 이중 테두리
                        g = max(3, int(5 * sc))
                        cv2.rectangle(frame, (x1 - g, y1 - g),
                                      (x2 + g, y2 + g), (255, 255, 0),
                                      max(2, int(3 * sc)))
                    if pid is not None and sel_rep is not None \
                            and pid >= 0 and self._rep(pid) == sel_rep:
                        # 선택된 트랙릿(병합 그룹 포함): 빨간 박스 + 이중 테두리
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), max(2, int(3 * sc)))
                        g = max(3, int(6 * sc))
                        cv2.rectangle(frame, (x1 - g, y1 - g),
                                      (x2 + g, y2 + g), (0, 0, 255),
                                      max(1, int(2 * sc)))
                    tag = ROLE_TAGS.get(team)
                    if team in (5, 6) and pid is not None:
                        tag = self._ref_tag(pid, team)    # 주심/선심 태그
                    if pid is not None and pid >= 0:
                        num = self._num_of(pid)           # 등번호
                        if num and num.isascii():
                            tag = f"{tag} {num}" if tag else num
                    if tag:
                        cv2.putText(frame, tag, (x1, y1 - max(4, int(8 * sc))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    max(0.5, 1.1 * sc), color,
                                    max(1, int(3 * sc)))
                # 무시(숨긴) 사람도 회색 점선 박스로 — 누구를 숨겼는지 보이고
                # 우클릭 복원 가능 (_players_row 는 제외하므로 raw 에서 별도로)
                if self.hidden_players:
                    raw = (list(self.analysis["players"][si])
                           + self.extra_players.get(int(si), []))
                    for pp in raw:
                        if len(pp) < 5 or pp[4] < 0:
                            continue
                        if self._rep(int(pp[4])) not in self.hidden_players:
                            continue
                        x1 = int((pp[0] - pp[2] / 2) * sc)
                        y1 = int((pp[1] - pp[3] / 2) * sc)
                        x2 = int((pp[0] + pp[2] / 2) * sc)
                        y2 = int((pp[1] + pp[3] / 2) * sc)
                        gc = (110, 110, 110)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), gc,
                                      max(1, int(1 * sc)))
                        cv2.putText(frame, "X",
                                    (x1, y1 - max(3, int(6 * sc))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    max(0.4, 0.8 * sc), gc, max(1, int(2 * sc)))
            ball_g = None
            # 레이더 공 = 수락 트랙 기준 — 원시 검출을 쓰면 무시한
            # 오인식 공이 미니맵에 계속 공으로 남는다
            if self._accepted_ball is not None \
                    and si < len(self._accepted_ball):
                ab = self._accepted_ball[si]
                bb = (None if ab is None or not np.all(np.isfinite(ab))
                      else (float(ab[0]), float(ab[1])))
            else:
                bb = self.analysis["balls"][si]
            if self._field_calib is not None:
                # 캘리브레이션 완료: 경기장 절대 좌표로 표시
                feet = [(pp[0], pp[1] + pp[3] / 2.0,
                         int(pp[4]) if len(pp) >= 5 else -1)
                        for pp in prow if len(pp) >= 4]
                if feet:
                    fc = pano_to_field(self._field_calib,
                                       [(a, b) for a, b, _ in feet])
                    for (gx, gy), (_, _, tid) in zip(fc, feet):
                        if np.isfinite(gx):
                            radar_pts.append((tid if tid >= 0 else None,
                                              gx, gy, self._role_of(tid)))
                if si in self._air:                 # 공중볼: 보정 XY 사용
                    ball_g = self._air[si][:2]
                elif bb is not None:
                    g = pano_to_field(self._field_calib, [[bb[0], bb[1]]])[0]
                    if np.isfinite(g[0]):
                        ball_g = (float(g[0]), float(g[1]))
            else:
                # 캘리브레이션 전: 카메라 기준 근사를 경기장 좌표계로 이동
                # (build_radar_data 와 동일한 가정 위치)
                cy0 = -(self.field_size[1] / 2.0 + 5.0)
                for X, Y, tid, j in ground_positions(prow, self.pano_w,
                                                     self.pano_h):
                    radar_pts.append((tid if tid >= 0 else None,
                                      X, cy0 + Y, self._role_of(tid)))
                if bb is not None:
                    g = ground_positions([[bb[0], bb[1], 0.0, 0.0]],
                                         self.pano_w, self.pano_h)
                    ball_g = (g[0][0], cy0 + g[0][1]) if g else None
            if self.check_radar.isChecked():
                pts, ball_sm = self._smooth_radar(radar_pts, ball_g, f)
                self._draw_radar_overlay(pts, ball_sm)
            else:
                self._hide_radar()
        # 경기장 탭: 랜드마크 마커 + (캘리브레이션 후) 예상 경기장 선
        if self._field_tab_active() or self.btn_field_pick.isChecked():
            rc = getattr(self, "_rc_calib", None) \
                if getattr(self, "_is_rotcam", False) else None
            if rc is not None:
                # 회전 카메라: 필드 라인을 현재 프레임 카메라 자세로 투영.
                # 기준 프레임 자세 → 현재 프레임 호모그래피로 이송.
                # 이송(회전 계산)이 실패하면 그리지 않는다 — 엉뚱한 위치에
                # 선을 그리느니 안 그리는 게 낫다 (사용자 방향).
                from ..core.rotcam import field_to_pixel
                cur_f = int(getattr(self, "_cur_frame_idx",
                                    self.slider.value()))
                pose = self._rc_current_pose(cur_f)   # 프레임별 회전
                if pose is not None:
                    Kc, Rc = pose["K"], pose["R"]
                    tc = -Rc @ np.asarray(rc["cam_pos"])
                    lines_iter = field_outline(*self.field_size,
                                               circle_r=self.field_circle_r)
                else:
                    lines_iter = []           # 자세 못 풀면 라인 안 그림
                for line in lines_iter:
                    q = field_to_pixel(Kc, Rc, tc, line)
                    seg = []
                    for qx, qy in list(q) + [(np.nan, np.nan)]:
                        ok = (np.isfinite(qx)
                              and -0.2 * self.pano_w <= qx <= 1.2 * self.pano_w
                              and -0.2 * self.pano_h <= qy <= 1.2 * self.pano_h)
                        if ok:
                            seg.append((qx, qy))
                            continue
                        if len(seg) >= 2:
                            arr = np.array([[int(a * sc), int(b * sc)]
                                            for a, b in seg], np.int32)
                            cv2.polylines(frame, [arr], False,
                                          (60, 220, 255), max(1, int(2 * sc)))
                        seg = []
            elif self._field_calib is not None:
                for line in field_outline(*self.field_size,
                                          circle_r=self.field_circle_r):
                    q = field_to_pano(self._field_calib, line)
                    seg = []
                    for qx, qy in list(q) + [(np.nan, np.nan)]:
                        ok = (np.isfinite(qx)
                              and -0.2 * self.pano_w <= qx <= 1.2 * self.pano_w
                              and -0.2 * self.pano_h <= qy <= 1.2 * self.pano_h
                              and not (seg and abs(qx - seg[-1][0])
                                       > self.pano_w / 4))   # 랩어라운드 컷
                        if ok:
                            seg.append((qx, qy))
                            continue
                        if len(seg) >= 2:
                            arr = np.array([[int(a * sc), int(b * sc)]
                                            for a, b in seg], np.int32)
                            cv2.polylines(frame, [arr], False, (255, 255, 255),
                                          max(1, int(3 * sc)))
                        seg = [(qx, qy)] if np.isfinite(qx) else []
            for lx, ly in self.line_points:      # 흰 선 검출 샘플 (녹색 점)
                cv2.circle(frame, (int(lx * sc), int(ly * sc)),
                           max(2, int(5 * sc)), (0, 255, 0), -1)
            rotcam = getattr(self, "_is_rotcam", False)
            for k, pt in self.field_points.items():
                kf = self.field_point_frames.get(k)
                # 회전 카메라: 찍은 프레임과 다르면 현재 프레임으로 이송한
                # 위치에 그린다 (팬 따라 이동). 이송 실패는 회색 원위치.
                tp = self._lm_transferred.get(k) if rotcam else None
                transferred = tp is not None and kf is not None and int(kf) != int(f)
                px = tp if tp is not None else pt
                q = (int(px[0] * sc), int(px[1] * sc))
                if rotcam:
                    invalid = self._lm_invalid_here(k, f)
                    lost = (kf is not None and int(kf) != int(f)
                            and tp is None and not invalid)
                    if invalid:                    # 이 프레임 수동 무효 (주황 X)
                        col = (0, 140, 255)
                        tag = LANDMARK_TAGS[k] + " X"
                    else:
                        col = (150, 150, 150) if lost else (255, 255, 0)
                        tag = LANDMARK_TAGS[k] + ("~" if transferred else "")
                else:
                    # 파노라마는 고정 뷰 — 어느 프레임에 찍었든 랜드마크
                    # 위치가 동일하다. 프레임-스테일 색칠(회색/청록)은
                    # rotcam 잔재라 혼란만 주므로 전부 동일하게 그린다.
                    col = (255, 255, 0)
                    tag = LANDMARK_TAGS[k]
                cv2.drawMarker(frame, q, col,
                               cv2.MARKER_TILTED_CROSS,
                               max(14, int(36 * sc)), max(2, int(6 * sc)))
                cv2.putText(frame, tag,
                            (q[0] + 12, q[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, 1.0 * sc), col,
                            max(1, int(3 * sc)))
            # 라인 점(터치라인 등) — 패밀리 색 작은 점 (rotcam 이송 반영)
            for key, pts in self.field_lines.items():
                lcol = LINE_FAMILIES[key][2]
                for lx, ly, lfr in pts:
                    q2 = (lx, ly)
                    if rotcam and int(lfr) != int(f):
                        H = self._rc_H(int(lfr), int(f))
                        if H is None:
                            continue
                        from ..core.rotcam import transfer_points
                        q2 = transfer_points(H, [[lx, ly]])[0]
                    cv2.circle(frame, (int(q2[0] * sc), int(q2[1] * sc)),
                               max(3, int(7 * sc)), lcol, -1)
                    cv2.circle(frame, (int(q2[0] * sc), int(q2[1] * sc)),
                               max(3, int(7 * sc)), (30, 30, 30),
                               max(1, int(2 * sc)))
        # 러버밴드 사각형 (드래그 다중 선택 중)
        if self._rb is not None and "x1" in self._rb:
            rx1, rx2 = sorted((self._rb["x0"], self._rb["x1"]))
            ry1, ry2 = sorted((self._rb["y0"], self._rb["y1"]))
            cv2.rectangle(frame, (int(rx1 * sc), int(ry1 * sc)),
                          (int(rx2 * sc), int(ry2 * sc)),
                          (255, 255, 0), max(1, int(2 * sc)))
        # 좌하단 소스 배지: 원본(풀해상도) vs 스크럽 프록시 (OpenCV 한글
        # 미지원 → ASCII). 프록시 표시 직후 백그라운드 승격되면 ORIG 로 바뀜.
        if getattr(self, "_cur_frame", None) is not None:
            bh, bw = frame.shape[:2]
            is_orig = getattr(self, "_cur_scale", 1.0) >= 1.0
            loading = (not is_orig
                       and getattr(self, "_native_pending", None)
                       == getattr(self, "_cur_frame_idx", -1))
            txt = ("ORIG" if is_orig
                   else "SCRUB -> ORIG..." if loading else "SCRUB")
            col = (0, 220, 0) if is_orig else (0, 170, 255)
            fs = max(0.5, bw / 1920.0 * 0.9)
            th = max(1, int(bw / 1920.0 * 2))
            (tw, tht), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                            fs, th)
            m = max(6, int(bw * 0.006))
            x0, y0 = m, bh - m
            cv2.rectangle(frame, (x0 - 4, y0 - tht - bl - 4),
                          (x0 + tw + 4, y0 + 4), (0, 0, 0), -1)
            cv2.putText(frame, txt, (x0, y0 - bl),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
        self.pane.set_frame(frame)

    # ------------------------------------------------------ 화면 오브젝트 조작
    def _pane_clicked(self, fx, fy):
        """좌클릭: 오브젝트 상태별 기본 동작 / 빈 곳은 키프레임 추가."""
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        if (self._field_tab_active() or self.btn_field_pick.isChecked()) \
                and self._landmark_at(x, y) is not None:
            return              # 랜드마크 위 클릭 = 드래그 미수 — 무동작
        if self.btn_field_pick.isChecked():   # 랜드마크 찍기 모드가 우선
            key = self._match_landmark(x, y)
            if key is None:
                self.log("[field] 수평선 위 — 지면 지점을 클릭하세요")
            else:
                self._field_set_point(key, x, y)
            return
        o = self._hit(x, y)
        if o is None:
            # 사람 위 클릭 = 선택 (드래그 러버밴드와 같은 _pane_sel 공유).
            # Shift = 추가/삭제 토글, 없으면 그 사람만 선택.
            tid = self._player_at(x, y)
            if tid is not None:
                shift = bool(QApplication.keyboardModifiers()
                             & Qt.KeyboardModifier.ShiftModifier)
                self._toggle_pane_sel(tid, shift)
                return
            if self._box_hit(x, y) is not None:
                return                      # 박스 테두리 클릭 = 드래그 미수 — 무동작
            # 빈 곳 = 모드에 따라 수동 공(기본) 또는 크롭 키프레임 추가
            w = None
            if self.btn_mode_kf.isChecked() and self.plan is not None \
                    and int(f) < len(self.plan["crop_w"]):
                w = float(self.plan["crop_w"][int(f)])
            self._add_keyframe(f, x, y, width=w)
            return
        if o["kind"] == "kf":               # 키프레임 = 삭제
            self._delete_keyframe_idx(o["i"])
            return
        st = o["state"]                     # 공 후보 = 상태별 기본 동작
        if st == "promoted":
            self._unpromote_at(f, o["x"], o["y"])
        elif st == "ignored":
            self._restore_at(f, o["x"], o["y"])
        else:                               # gray/green → 경기 공 승격
            self._promote_ball(f, o["si"], o["x"], o["y"])

    def _pane_context(self, fx, fy, gpos):
        """우클릭: 오브젝트별 전체 동작 메뉴."""
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self.cap is None or self.analysis is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        menu = QMenu(self)

        # 드래그로 여러 명 선택돼 있으면 전원 대상 액션을 맨 위에
        if self._pane_sel:
            n = len(self._pane_sel)
            sub = menu.addMenu(f"선택 {n}명 — 역할/팀 지정")
            for r in (0, 1, 5, 6, 3, 4):
                sub.addAction(self._role_name(r),
                              lambda _=False, rr=r: self._sel_apply_role(rr))
            sub.addSeparator()
            sub.addAction("자동 분류로 되돌리기",
                          lambda _=False: self._sel_apply_role(None))
            if n >= 2:
                menu.addAction(f"선택 {n}명 병합 (같은 사람)",
                               lambda _=False: self._merge_tracklets(
                                   sorted(self._pane_sel)))
            menu.addAction(f"선택 {n}명 숨기기 (관중·오인식)",
                           lambda _=False: self._sel_hide())
            menu.addAction("선택 해제",
                           lambda _=False: (setattr(self, "_pane_sel", set()),
                                            self._redraw()))
            menu.addSeparator()

        def _add_kf_here():
            w = (float(self.plan["crop_w"][int(f)])
                 if self.plan is not None and int(f) < len(self.plan["crop_w"])
                 else None)
            self._add_keyframe(f, x, y, width=w)

        if o is None:
            menu.addAction("여기 수동 공 추가",
                           lambda: self._add_keyframe(f, x, y))
            menu.addAction("여기 크롭 키프레임 추가", _add_kf_here)
        elif o["kind"] == "kf":
            is_kf = len(self.keyframes[o["i"]]) > 3
            if not is_kf:
                k = self.keyframes[o["i"]]
                menu.addAction("앞뒤로 추적 확장 (±4s)",
                               lambda: self._propagate(
                                   int(k[0]), float(k[1]), float(k[2]),
                                   "ball"))
            menu.addAction("공으로 전환 (줌 자동)" if is_kf
                           else "크롭 키프레임으로 전환 (현재 폭 고정)",
                           lambda: self._toggle_kf_type(o["i"]))
            menu.addAction("삭제",
                           lambda: self._delete_keyframe_idx(o["i"]))
        else:
            st, ox, oy, si = o["state"], o["x"], o["y"], o["si"]
            if st == "promoted":
                menu.addAction("승격 취소",
                               lambda: self._unpromote_at(f, ox, oy))
            else:
                menu.addAction("이 공을 경기 공으로 승격",
                               lambda: self._promote_ball(f, si, ox, oy))
            if st == "ignored":
                menu.addAction("무시 해제(복원)",
                               lambda: self._restore_at(f, ox, oy))
            else:
                menu.addAction("이 검출 무시(오인식)",
                               lambda: self._ignore_track_at(f, si, ox, oy))
            menu.addSeparator()
            menu.addAction("여기 수동 공 추가",
                           lambda: self._add_keyframe(f, x, y))
        tid = self._player_at(x, y)
        hid = self._hidden_player_at(x, y) if tid is None else None
        if hid is not None:
            menu.addSeparator()
            menu.addAction(f"#{hid} 복원 (숨김 해제)",
                           lambda _=False, r=hid: self._unhide_one(r))
        if tid is None and hid is None and self._injected_person_at(x, y):
            menu.addSeparator()
            a = menu.addAction("ID 없는 주입 검출 (갭필) — 역할을 붙이려면 "
                               "\"이 주변 사람 재검출\"로 ID 부여")
            a.setEnabled(False)
        if tid is not None:
            menu.addSeparator()
            hdr = menu.addAction(
                f"선수 #{tid} — {self._disp_role(tid, self._role_of(tid))}")
            hdr.setEnabled(False)
            # 최상위에 펼침: 팀1 → 팀1 번호들, 팀2 → 팀2 번호들 (서브메뉴 X)
            for r in (0, 1):
                menu.addSeparator()
                menu.addAction(self._role_name(r),
                               lambda _=False, rr=r, t=tid:
                               self._set_role(t, rr))
                self._add_num_items(menu, tid, r)   # 그 팀 번호들 바로 밑
            menu.addSeparator()
            # 심판·선심·GK 는 덜 흔하므로 작은 서브메뉴로
            osub = menu.addMenu("기타 역할 (심판·선심·GK)")
            for r in (5, 6, 3, 4):
                osub.addAction(self._role_name(r),
                               lambda _=False, rr=r, t=tid:
                               self._set_role(t, rr))
            if tid in self.roles:
                osub.addSeparator()
                osub.addAction("자동 분류로 되돌리기",
                               lambda _=False, t=tid: self._set_role(t, None))
            # 오인식 무시 (최상위, 직접)
            menu.addAction("오인식 무시 — 이 사람 숨기기 (선수/심판 아님)",
                           lambda _=False, t=tid: self._hide_player(t))
            if tid >= 900001:
                menu.addSeparator()
                row = next((p for si_ in self.extra_players
                            for p in self.extra_players[si_]
                            if p[4] == tid), None)
                if row is not None:
                    si_seed = next(si_ for si_ in self.extra_players
                                   if any(p[4] == tid for p in
                                          self.extra_players[si_]))
                    f_seed = int(self.analysis["frames"][si_seed])
                    menu.addAction("앞뒤로 추적 확장 (±4s)",
                                   lambda _=False, ff=f_seed, rr=row, t=tid:
                                   self._propagate(ff, rr[0], rr[1],
                                                   "person", ctx=t))
                menu.addAction("이 수동 검출 삭제",
                               lambda _=False, t=tid: self._delete_extra(t))
        if self.analysis is not None and self._current_sample() is not None:
            menu.addSeparator()
            menu.addAction("이 주변 사람 재검출",
                           lambda _=False: self._detect_here(f, x, y, gpos))
        if self._field_tab_active():
            menu.addSeparator()
            sub = menu.addMenu("이 위치를 랜드마크로 지정")
            for key, name, req in LANDMARKS:
                mark = "★ " if req else ""
                cur = " ✓" if key in self.field_points else ""
                sub.addAction(f"[{LANDMARK_TAGS[key]}] {mark}{name}{cur}",
                              lambda _=False, kk=key:
                              self._field_set_point(kk, x, y))
            # 라인 위 다점 (터치라인 등) — point-to-line 캘리브로 f 안정화
            subl = menu.addMenu("이 점을 라인에 추가 (다점 → f 안정화)")
            for key, (fam, name, col) in LINE_FAMILIES.items():
                n = len(self.field_lines.get(key, []))
                subl.addAction(f"{name}" + (f" ({n}점)" if n else ""),
                               lambda _=False, kk=key: self._line_add_point(
                                   kk, x, y))
            lp = self._line_point_at(x, y, r=40.0)
            if lp is not None:
                menu.addAction(
                    f"라인 점 삭제 [{LINE_FAMILIES[lp[0]][1]}]",
                    lambda _=False, kk=lp[0], ii=lp[1]:
                    self._line_del_point(kk, ii))
            for key in list(self.field_lines):
                menu.addAction(f"{LINE_FAMILIES[key][1]} 점 전체 삭제 "
                               f"({len(self.field_lines[key])}개)",
                               lambda _=False, kk=key: self._line_clear(kk))
            near = self._landmark_at(x, y, r=100.0)
            if near is not None:
                near_name = next(n for k, n, _ in LANDMARKS if k == near)
                sub = menu.addMenu(f"마커 [{LANDMARK_TAGS[near]}] "
                                   f"{near_name} → 다른 랜드마크로 변경")
                for key, name, req in LANDMARKS:
                    if key == near:
                        continue
                    mark = "★ " if req else ""
                    cur = " ✓(교환)" if key in self.field_points else ""
                    sub.addAction(f"[{LANDMARK_TAGS[key]}] {mark}{name}{cur}",
                                  lambda _=False, o=near, n=key:
                                  self._field_reassign(o, n))
                menu.addAction(f"마커 [{LANDMARK_TAGS[near]}] 지정 해제",
                               lambda _=False, kk=near:
                               self._field_remove_point(kk))
            # rotcam: 다른 프레임에서 이송된 마커가 여기서 엉뚱하면
            # "이 프레임(구간) 이송 무효" 지정 — 원 프레임에선 유효 유지
            if near is not None and getattr(self, "_is_rotcam", False):
                f = int(getattr(self, "_cur_frame_idx", self.slider.value()))
                kf = self.field_point_frames.get(near)
                tag = LANDMARK_TAGS[near]
                if not (kf is None or int(kf) == f):    # 이송된 마커만
                    menu.addSeparator()
                    win = int(round(3.0 * self.fps))
                    if self._lm_invalid_here(near, f):
                        menu.addAction(
                            f"[{tag}] 이 프레임 이송 무효 해제",
                            lambda _=False, kk=near, ff=f:
                            self._lm_remove_invalid_at(kk, ff))
                    else:
                        menu.addAction(
                            f"[{tag}] 이 부근 이송 무효 (±3초)",
                            lambda _=False, kk=near, a=f - win, b=f + win:
                            self._lm_add_invalid(kk, a, b))
                        menu.addAction(
                            f"[{tag}] 여기부터 끝까지 이송 무효",
                            lambda _=False, kk=near, a=f:
                            self._lm_add_invalid(kk, a, self.total - 1))
                        menu.addAction(
                            f"[{tag}] 처음부터 여기까지 이송 무효",
                            lambda _=False, kk=near, b=f:
                            self._lm_add_invalid(kk, 0, b))
                    if self.field_point_invalid.get(near):
                        menu.addAction(
                            f"[{tag}] 이송 무효 구간 전체 해제",
                            lambda _=False, kk=near:
                            self._lm_clear_invalid(kk))
            # rotcam: 이 프레임 이송이 실제 라인과 맞으면 '확정'해 앵커로
            # 승격 → 이후 먼 프레임 이송의 최단 소스 (프레임 일괄).
            if getattr(self, "_is_rotcam", False):
                ff = int(getattr(self, "_cur_frame_idx", self.slider.value()))
                n_tr = sum(
                    1 for k in self._lm_transferred
                    if not (self.field_point_frames.get(k) is not None
                            and int(self.field_point_frames[k]) == ff))
                n_conf = sum(1 for k in self.field_point_anchors
                             for a in self.field_point_anchors[k]
                             if int(a[2]) == ff)
                if n_tr or n_conf:
                    menu.addSeparator()
                    if n_tr:
                        menu.addAction(
                            f"✓ 이 프레임 이송 {n_tr}개 확정 (앵커 승격)",
                            lambda _=False: self._rc_confirm_frame())
                    if n_conf:
                        menu.addAction(
                            f"이 프레임 확정 {n_conf}개 해제",
                            lambda _=False: self._rc_unconfirm_frame())
        menu.exec(gpos)

    def _pane_hover(self, fx, fy):
        """커서 근처 오브젝트/크롭 박스를 하이라이트 (바뀔 때만 리드로우)."""
        if self.mc is not None and self.mc.alt_on_main:
            return                        # 전환 모드: alt 표시 중 편집 입력 차단
        if self.analysis is None or getattr(self, "_cur_frame", None) is None:
            return
        if self._box_edit is not None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        zone = None if o is not None else self._box_hit(x, y)
        key = ((None if o is None
                else (o["kind"], round(o["x"]), round(o["y"]))), zone)
        if key != self._hover_key:
            self._hover_key = key
            self._hover = o
            self._box_hover = zone
            if o is not None:
                cur = Qt.CursorShape.PointingHandCursor
            elif zone is None:
                cur = Qt.CursorShape.OpenHandCursor
            elif zone[0] == "border":
                cur = Qt.CursorShape.SizeAllCursor
            else:                            # 모서리: 대각 리사이즈 커서
                cur = (Qt.CursorShape.SizeFDiagCursor if zone[1] in (0, 3)
                       else Qt.CursorShape.SizeBDiagCursor)
            self.pane.setCursor(cur)
            self._redraw()
        # 무시된 공(=공 아님)에 커서를 올리면 사유를 툴팁으로
        from PyQt6.QtWidgets import QToolTip
        from PyQt6.QtGui import QCursor
        if o is not None and o.get("kind") == "ball" \
                and o.get("state") == "ignored":
            QToolTip.showText(QCursor.pos(),
                              "무시됨 — 이 물체는 공이 아님", self.pane)
        else:
            QToolTip.hideText()

    def _refresh_lists(self):
        """위: 공(자동 트랙+수동 키프레임) / 아래: 오인식(무시 구간)."""
        # 위 목록 — 시간순 병합
        # 재구성 전 선택을 (종류, 시작 프레임) 으로 기억 — 승격/무시로
        # 목록이 다시 만들어져도 보던 항목을 따라간다 (점프 방지)
        old_top = getattr(self, "_top", [])
        cur = self.track_list.currentRow()
        sel_key = None
        if 0 <= cur < len(old_top):
            k, i = old_top[cur]
            if k == "kf" and i < len(self.keyframes):
                sel_key = ("kf", self.keyframes[i][0])
            elif k == "track":
                sel_key = ("track", None)   # 트랙 인덱스는 재계산됨 — 아래서 위치로
        self._top = ([("track", i) for i in range(len(self.track_spans))]
                     + [("kf", i) for i in range(len(self.keyframes))])
        self._top.sort(key=lambda e: (self.track_spans[e[1]][0] if e[0] == "track"
                                      else self.keyframes[e[1]][0]))
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for kind, i in self._top:
            if kind == "track":
                f0, f1 = self.track_spans[i]
                t0, t1 = f0 / self.fps, f1 / self.fps
                self.track_list.addItem(
                    f"{int(t0//60):02d}:{t0%60:04.1f} ~ "
                    f"{int(t1//60):02d}:{t1%60:04.1f}  ({t1-t0:.1f}s) 자동")
            else:
                k = self.keyframes[i]
                kf, kx, ky = k[0], k[1], k[2]
                t = kf / self.fps
                label = (f"◆ 키프레임 ({kx:.0f}, {ky:.0f}, 폭{k[3]:.0f})"
                         if len(k) > 3 else f"● 수동 공 ({kx:.0f}, {ky:.0f})")
                self.track_list.addItem(
                    f"{int(t//60):02d}:{t%60:04.1f}  {label}")
        # 선택 복원 (시그널 차단 상태 — 복원이 시크를 유발하지 않게)
        if sel_key is not None:
            pos = self.slider.value()
            for row, (k, i) in enumerate(self._top):
                if sel_key[0] == "kf" and k == "kf" \
                        and self.keyframes[i][0] == sel_key[1]:
                    self.track_list.setCurrentRow(row)
                    break
                if sel_key[0] == "track" and k == "track" \
                        and self.track_spans[i][0] <= pos <= self.track_spans[i][1]:
                    self.track_list.setCurrentRow(row)
                    break
        self.track_list.blockSignals(False)
        # 아래 목록 — 오인식만
        self.kf_list.clear()
        for r in self.ignores:
            f0, f1 = r[0], r[1]
            pos = f"  @({r[2]:.0f},{r[3]:.0f})" if len(r) >= 4 else ""
            self.kf_list.addItem(
                f"{int(f0/self.fps//60):02d}:{f0/self.fps%60:04.1f}~"
                f"{int(f1/self.fps//60):02d}:{f1/self.fps%60:04.1f}  공 아님{pos}")

    # 기존 호출부 호환 별칭
    _refresh_kf_list = _refresh_lists
    _refresh_track_list = _refresh_lists

    def _goto_kf(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            self.slider.setValue(int(self.ignores[row][0]))

    def _delete_kf(self):
        """선택 복원 — 오인식 마킹을 철회해 트랙을 되살린다."""
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            del self.ignores[row]
            self._save_keyframes()
            self._recompute_tracks()
            self._redraw()

    def _save_keyframes(self):
        """마킹 변경 저장 (1.5s 디바운스 — 분석 포함 파일이라 통으로 씀)."""
        self._save_timer.start()


    def _far_zoom_changed(self, v):
        QSettings("PyStitch360", "PyStitch360").setValue("ptz_far_zoom", v)
        self._plan_dirty()

    def _plan_dirty(self):
        if self.analysis is not None:
            self._plan_timer.start()

    def _run_plan(self):
        if self.analysis is None:
            return
        if self._plan_worker is not None and self._plan_worker.isRunning():
            self._plan_timer.start()      # 진행 중이면 잠시 뒤 재시도
            return
        excl = self._person_excluded()    # 수동 숨김 + 자동 장외
        hidden = {t for t in self._pspans
                  if self._rep(t) in excl} if excl else set()
        w = PlanWorker(self.analysis, [tuple(k) for k in self.keyframes],
                       [tuple(r) for r in self.ignores],
                       self.combo_mode.currentIndex() == 1, linked=self._linked,
                       far_zoom=self.spin_far_zoom.value(),
                       promotes=[tuple(p) for p in self.promotes],
                       exclude_tids=hidden,
                       top_margin=getattr(self, "_top_black", None))
        w.done.connect(self._plan_done)
        self._plan_worker = w
        w.start()

    def _plan_done(self, plan, out):
        self.plan = plan
        self.plan_out = out
        self._box_commit = None          # 새 계획 도착 — 낙관적 박스 해제
        self._redraw()

    def _start_link(self):
        """트랙 연결(느린 단계)을 백그라운드로 1회 계산."""
        if self.analysis is None:
            return
        ap = (self.pano_path.with_suffix(".analysis.json")
              if self.pano_path else None)
        # 상단 검은 경계는 파노라마 고정값 — 사이드카에 캐시돼 있으면
        # 재측정 생략(영상 시크 비용 회피). 없을 때만 워커에 video_path
        # 를 넘겨 1회 측정하고 _link_done 이 캐시에 저장한다.
        need_top = getattr(self, "_top_black", None) is None
        if not need_top:
            self.log(f"[plan] 상단 검은 경계 캐시: {self._top_black}px (측정 생략)")
        w = LinkWorker(self.analysis, analysis_path=ap,
                       video_path=(self.pano_path if need_top else None))
        w._analysis_id = id(self.analysis)    # 결과 도착 시 최신인지 확인
        w.done.connect(lambda linked, aid=id(self.analysis):
                       self._link_done(linked, aid))
        w.log.connect(self.log)               # 워커 단계 로그 → 로그 탭
        self._link_worker = w
        self.log("[ptz] 트랙 연결 계산 중... (완료 후 클릭 반응이 빨라짐)")
        # 비동기 워커라도 완료까지 대기 커서 — measure_top_black 의 원본
        # 시크가 수 초 걸려 '멈춘 듯' 보이던 걸 '작업 중'으로 표시.
        if not getattr(self, "_link_cursor", False):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._link_cursor = True
        w.start()

    def _airborne_key(self):
        """공중볼 캐시 무효화 키 — 수락 궤적에 영향 주는 입력의 해시."""
        import hashlib
        n_cands = sum(len(c) for c in (self.analysis.get("ball_cands")
                                       or []))
        payload = json.dumps([len(self.analysis["frames"]), n_cands,
                              self.ignores, self.promotes],
                             sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def _recompute_airborne(self):
        """공중볼 보정 재계산 — 파일 캐시(.events.json) 우선, 백그라운드."""
        self._air = {}
        if self.analysis is None or self._field_calib is None \
                or self._accepted_ball is None:
            return
        self._air_gen = getattr(self, "_air_gen", 0) + 1
        key = self._airborne_key()
        self._air_key = key
        segments = None
        try:                              # 저장된 구간이 유효하면 스캔 생략
            from ..core.events import events_json_path
            p = events_json_path(self.pano_path)
            if p.exists():
                doc = json.loads(p.read_text())
                air = doc.get("airborne")
                if air and air.get("key") == key:
                    segments = [(int(a), int(b), f)
                                for a, b, f in air["segments"]]
        except Exception as e:  # noqa: BLE001
            self.log(f"[air] 캐시 무시: {e}")
        w = AirborneWorker(self._air_gen,
                           np.asarray(self.analysis["frames"], dtype=float),
                           float(self.fps),
                           np.array(self._accepted_ball, copy=True),
                           dict(self._field_calib), segments=segments)
        w.done.connect(self._airborne_done)
        self._air_worker = w
        w.start()

    def _airborne_done(self, gen, cache, segments, msg):
        if gen != getattr(self, "_air_gen", 0):
            return                        # 그 사이 재계산됨 — 낡은 결과 폐기
        self._air = cache
        if msg:
            self.log(msg)
        if "캐시 재사용" not in msg:      # 새 스캔 결과만 저장
            try:
                from ..core.events import save_events
                save_events(self.pano_path,
                            airborne={"key": getattr(self, "_air_key", ""),
                                      "segments": segments})
            except Exception as e:  # noqa: BLE001
                self.log(f"[air] 캐시 저장 실패: {e}")
        # 타임라인 '뜬 공' 레인 갱신
        frames = np.asarray(self.analysis["frames"]) \
            if self.analysis is not None else None
        if frames is not None:
            self.trackbar.set_airborne(
                [(int(frames[i0]), int(frames[i1]),
                  float(f["apex_z"])) for i0, i1, f in segments])
        if cache:
            self._redraw()

    def _link_done(self, linked, analysis_id=None):
        # 워커 완료 → 대기 커서 해제 (레이스로 버려지는 경우도 반드시 해제)
        if getattr(self, "_link_cursor", False):
            QApplication.restoreOverrideCursor()
            self._link_cursor = False
        if "tracks" not in linked:            # 워커 오류(_error) — 커서만 풀고 종료
            return
        # 영상 전환 레이스: 이 결과가 계산될 때의 analysis 가 아직 현재
        # 것인지 확인 — 아니면 버린다 (옛 트랙 인덱스가 새 analysis
        # 범위를 벗어나 IndexError 나던 문제)
        if analysis_id is not None and analysis_id != id(self.analysis):
            return
        # 레인 계산(트랙 요약·목록·트랙바)은 여기서 — 커서 표시 대상
        with self._wait():
            self._linked = linked
            tb = linked.pop("top_black", None)
            if tb is not None:
                self._top_black = int(tb)
                self.log(f"[plan] 상단 검은 경계 실측: {tb}px"
                         + ("" if tb else " — 상단 클램프 해제"))
                self._save_keyframes()        # 캐시 — 다음 열기엔 측정 생략
            self._teams = linked.pop("teams", {}) or {}
            if self.roles and self.analysis is not None:
                self._teams = classify_teams(self.analysis, roles=self.roles,
                                             feats=self._team_feats())
            self._update_outfield()       # 병합 대표 확정 후 장외 판정
            if self._outfield:
                self.log(f"[ptz] 장외 트랙릿 {len(self._outfield)}개 자동 "
                         "숨김 (경기장 캘리브 기반 — '장외 숨김' 체크 해제로 표시)")
            n_team = sum(1 for v in self._teams.values() if v < 2)
            self.log(f"[ptz] 트랙 연결 완료: {len(linked['tracks'])}개"
                     + (f", 팀 분류 선수 ID {n_team}개" if self._teams else
                        " (팀 분류: ID 포함 재분석 필요)"))
            self._refresh_team_label(log_colors=True)
            self._refresh_player_list()
            self._recompute_tracks()
            self._plan_dirty()

    # ------------------------------------------------------ 선수(역할) 검수
    def _player_cache(self):
        """트랙릿 요약({tid: [f0, f1, n]})·대표색 캐시 — 분석 객체당 1회.

        숨김(hidden_players)은 raw 캐시 위의 가벼운 필터 — 숨기기/복원이
        전체 검출 재스캔(수 초, UI 프리즈)을 유발하지 않는다. 대표색은
        분석에만 의존하므로 별도 키로 캐시.
        """
        if self.analysis is None:
            return {}, {}
        if self._pcache_id != id(self.analysis):
            # 파생 요약은 디스크 캐시 경유 (.analysis.cache.json) —
            # 재열기 시 검출 수십만 행 재스캔·대표색 재계산 생략
            summ = analysis_summary(
                self.pano_path.with_suffix(".analysis.json")
                if self.pano_path else "", self.analysis, log=self.log)
            spans = dict(summ["spans"])
            frames = self.analysis["frames"]
            for si, rows in self.extra_players.items():   # 수동 검출 포함
                if si < len(frames):
                    f = int(frames[si])
                    for p in rows:
                        spans[int(p[4])] = [f, f, 1]
            self._pspans = spans
            if self._pcolors_id != id(self.analysis):
                self._pcolors = summ["colors"]
                self._pcolors_id = id(self.analysis)
            self._footmed = summ["foot_med"]
            self._footmed_id = id(self.analysis)
            self._pcache_id = id(self.analysis)
        if self.hidden_players:
            m = self.merges.get                # 지역 바인딩 — 핫 루프
            hp = self.hidden_players
            spans = {t: v for t, v in self._pspans.items()
                     if m(t, t) not in hp}
            return spans, self._pcolors
        return self._pspans, self._pcolors

    def _tid_bgr(self):
        """트랙릿 대표색 BGR 캐시 — cvtColor 1회(벡터화)로 전 트랙릿 변환.

        목록 아이콘·범례가 트랙릿마다 cvtColor 를 부르던 것을 대체
        (선수 목록 재구성 프리즈의 주범).
        """
        _, cols = self._player_cache()
        if self._pbgr_id != self._pcolors_id:
            ids = list(cols)
            if ids:
                hsv = np.array([[(int(cols[t][0]) % 180,
                                  min(cols[t][1], 255.0),
                                  min(cols[t][2], 255.0)) for t in ids]])
                # 그림자 보정 (_boost_bgr 과 동일) — 벡터화
                hsv[..., 1] = np.minimum(hsv[..., 1] * 1.35, 255)
                hsv[..., 2] = np.minimum(
                    np.maximum(hsv[..., 2] * 1.55, 190), 255)
                bgr = cv2.cvtColor(hsv.astype(np.uint8),
                                   cv2.COLOR_HSV2BGR)[0]
                self._pbgr = {t: tuple(int(v) for v in bgr[i])
                              for i, t in enumerate(ids)}
            else:
                self._pbgr = {}
            self._pbgr_id = self._pcolors_id
        return self._pbgr

    def _team_feats(self):
        """classify_teams 전처리(전 검출 스캔) 캐시 — 분석당 1회.

        역할/번호 지정마다 재추출하면 수 초 UI 프리즈가 생긴다.
        """
        from ..core.ptz import team_features
        if self._tfeat_id != id(self.analysis):
            self._tfeat = team_features(self.analysis)
            self._tfeat_id = id(self.analysis)
        return self._tfeat

    def _role_name(self, r):
        """역할 표시명 — 팀1/팀2 자리에 사용자 입력 팀 이름."""
        t1, t2 = self.team_names
        return {0: t1, 1: t2, 2: "기타",
                3: f"{t1} GK", 4: f"{t2} GK",
                5: "주심", 6: "선심"}.get(r, "기타")

    def _ref_tag(self, tid, role=5):
        """심판 트랙릿 태그 — 선심은 근/원측 (ARN/ARF, referee.py 분류).

        병합 그룹의 어느 멤버든 분류에 있으면 그룹 전체에 적용.
        분류 정보가 없으면 역할 기본값 (주심=REF, 선심=AR).
        """
        rep = self._rep(tid)
        grp = {int(tid), rep} | {t for t, r in self.merges.items()
                                 if r == rep}
        if grp & {int(t) for t in self._referees.get("ar_near") or []}:
            return "ARN"
        if grp & {int(t) for t in self._referees.get("ar_far") or []}:
            return "ARF"
        if role == 6:
            return self._ar_side(rep, grp) or "AR"
        return "REF"

    def _foot_med(self):
        """트랙릿별 발 위치 중앙값 캐시 — 분석당 1회 (프로파일: 재스캔이
        _ar_side 를 통해 역할 지정마다 수백 ms~수 초를 먹던 주범)."""
        if self._footmed_id != id(self.analysis):
            pts: dict[int, list] = {}
            for prow in self.analysis["players"]:
                for p in prow:
                    if len(p) >= 5 and p[4] >= 0:
                        pts.setdefault(int(p[4]), []).append(
                            (p[0], p[1] + p[3] / 2.0))
            self._footmed = {t: (float(np.median([q[0] for q in v])),
                                 float(np.median([q[1] for q in v])))
                             for t, v in pts.items()}
            self._footmed_id = id(self.analysis)
        return self._footmed

    def _ar_side(self, rep, grp):
        """선심 근/원측 자동 판정 — 그룹 발 위치 중앙값의 필드 Y 부호.

        역할 6 지정만으로 ARN/ARF 가 붙도록 (referee.py 분류 없이도).
        트랙릿별 중앙값 캐시(_foot_med) 위에서 변환 몇 번뿐이라 싸다.
        """
        if rep in self._ar_side_cache:
            return self._ar_side_cache[rep]
        if self.analysis is None:
            return None
        fm = self._foot_med()
        feet = [fm[t] for t in grp if t in fm]
        if not feet:
            return None
        ys = []
        if self._field_calib is not None:
            for gx, gy in pano_to_field(self._field_calib, feet):
                if np.isfinite(gy):
                    ys.append(float(gy))
        else:
            for X, Y, _t, _j in ground_positions(
                    [[fx, fy, 0.0, 0.0] for fx, fy in feet],
                    self.pano_w, self.pano_h):
                # 캘리브레이션 전: 카메라 거리로 근사 (near = 가까움)
                ys.append(float(Y) - (self.field_size[1] / 2.0 + 5.0))
        if not ys:
            return None
        side = "ARN" if float(np.median(ys)) < 0.0 else "ARF"
        self._ar_side_cache[rep] = side
        return side

    def _disp_role(self, tid, r):
        """목록용 역할명 — 선심은 ARN(근측)/ARF(원측) 표기 (+등번호)."""
        if r in (5, 6):
            return {"ARN": "선심 ARN", "ARF": "선심 ARF",
                    "AR": "선심"}.get(self._ref_tag(tid, r), "주심")
        name = self._role_name(r)
        num = self._num_of(tid)
        return f"{name} {num}번" if num else name

    def _hide_player(self, tid):
        """관중·오인식 트랙릿 숨김 (비파괴) — 병합 그룹 단위."""
        rep = self._rep(tid)
        self.hidden_players.add(rep)
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._plan_dirty()                # 크롭 계획도 재계산 (선수 추종 제외)
        self._redraw()
        self.log(f"[ptz] #{rep} 숨김 (관중·오인식) — 선수 목록 우클릭 "
                 f"\"숨긴 사람 복원\" 또는 역할 초기화로 되돌림")

    def _unhide_players(self):
        n = len(self.hidden_players)
        self.hidden_players = set()
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] 숨긴 사람 {n}명 복원")

    def _unhide_one(self, rep):
        self.hidden_players.discard(int(rep))
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] #{rep} 복원 (숨김 해제)")

    # -------------------------------------- 장외 자동 숨김 (경기장 안만 인식)
    _INFIELD_MARGIN = 2.0                 # 피치 경계 여유 (m) — 스로인 등

    def _update_outfield(self):
        """트랙릿별 발밑 필드 좌표 과반이 피치(+여유) 밖이면 장외 판정.

        파노라마 전용 (고정 캘리브 → 전 샘플 일괄 투영, 표본 상한
        ~4천 샘플). rotcam 은 프레임별 자세가 필요해 _redraw 가 현재
        프레임만 거른다. 비파괴 — hidden_players 처럼 표시·목록·레이더·
        크롭 계획·OCR 후보에서만 제외, 분석 원본은 그대로."""
        self._outfield = set()
        if (self.analysis is None or self._field_calib is None
                or getattr(self, "_is_rotcam", False)
                or not getattr(self, "check_infield", None)
                or not self.check_infield.isChecked()):
            return
        cache = getattr(self, "_foot_cache", None)
        if cache is None or cache[0] != id(self.analysis):
            tids, pts = [], []
            players = self.analysis["players"]
            step = max(1, len(players) // 4000)
            for si in range(0, len(players), step):
                for p in players[si]:
                    if len(p) >= 5 and p[4] >= 0:
                        tids.append(int(p[4]))
                        pts.append((float(p[0]),
                                    float(p[1]) + float(p[3]) / 2.0))
            cache = self._foot_cache = (
                id(self.analysis), np.asarray(tids, int),
                np.asarray(pts, float).reshape(-1, 2))
        _aid, tids, pts = cache
        if not len(tids):
            return
        uniq, inv = np.unique(tids, return_inverse=True)
        reps = np.asarray([self._rep(int(t)) for t in uniq])[inv]
        fxy = pano_to_field(self._field_calib, pts)
        hl = self.field_size[0] / 2.0 + self._INFIELD_MARGIN
        hw = self.field_size[1] / 2.0 + self._INFIELD_MARGIN
        inside = (np.isfinite(fxy).all(axis=1)
                  & (np.abs(fxy[:, 0]) <= hl) & (np.abs(fxy[:, 1]) <= hw))
        for r in np.unique(reps):
            m = reps == r
            # 과반이 밖(수평선 위 포함)이면 장외 — 경계 선수는 살아남는다
            if np.count_nonzero(inside[m]) * 2 < np.count_nonzero(m):
                self._outfield.add(int(r))

    def _person_excluded(self):
        """표시·목록·계획에서 제외할 대표 tid 집합 (수동 숨김 + 자동 장외)."""
        return (self.hidden_players | self._outfield
                if self._outfield else self.hidden_players)

    def _infield_toggled(self, on):
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_infield_only", "true" if on else "false")
        self._update_outfield()
        self._refresh_team_label()
        self._refresh_player_list()
        self._plan_dirty()                # 크롭 계획도 장외 제외 반영
        self._redraw()
        if self.analysis is not None:
            self.log(f"[ptz] 장외 숨김 {'켬' if on else '끔'}"
                     + (f" — 장외 트랙릿 {len(self._outfield)}개"
                        if on and self._outfield else ""))

    # ------------------------------------------------------ 등번호
    def _num_of(self, tid):
        """유효 등번호 — 병합 대표 기준 (없으면 None)."""
        rep = self._rep(tid)
        return self.player_nums.get(rep, self.player_nums.get(int(tid)))

    def _cooccur_samples(self, rep_a, rep_b):
        """두 대표(병합 그룹)가 같은 샘플에 동시 등장하는 수 — 같은 팀
        같은 번호 두 명은 물리적으로 불가하다는 제약의 판정 근거."""
        ga = {t for t in self._pspans if self._rep(t) == rep_a}
        gb = {t for t in self._pspans if self._rep(t) == rep_b}
        n = 0
        for prow in self.analysis["players"]:
            ia = ib = False
            for p in prow:
                if len(p) >= 5 and p[4] >= 0:
                    t = int(p[4])
                    ia = ia or t in ga
                    ib = ib or t in gb
                if ia and ib:
                    n += 1
                    break
        return n

    def _set_player_num(self, tid, team, num):
        """등번호 지정 — 번호는 팀 소속을 함의하므로 역할도 맞춘다.

        같은 팀에 같은 번호가 이미 있으면:
        - 동시 등장(같은 프레임) = 물리적으로 다른 사람 → **방금 지정한
          쪽이 우선**, 기존 소유자에게서 번호를 회수한다 (사용자 방향:
          수동 지정이 최우선). 오탐/뒤바뀐 배정을 클릭 한 번으로 교정.
        - 동시 등장 없음 = 같은 사람 → 자동 병합.
        """
        rep = self._rep(tid)
        num = str(num)
        other = next((self._rep(t) for t, n in self.player_nums.items()
                      if n == num and self._rep(t) != rep
                      and self._role_of(t) in (team, team + 3)), None)
        if other is not None and self.analysis is not None:
            co = self._cooccur_samples(rep, other)
            if co > 2:                        # 검출 지터 여유
                # 다른 사람인데 번호 충돌 — 수동 지정 우선. 회수는 최소한:
                # 그 번호를 실제로 보유한 '한' 키만 지운다 (병합 그룹의
                # 다른 트랙릿 번호나 병합 자체는 건드리지 않음).
                owner = next((k for k, v in self.player_nums.items()
                              if v == num and self._rep(k) == other
                              and self._role_of(k) in (team, team + 3)), None)
                if owner is not None:
                    self.player_nums.pop(owner, None)
                self.log(f"[ptz] {self.team_names[team]} {num}번 회수: "
                         f"#{other} → #{rep} (수동 지정 우선, 동시 {co}회)")
            else:
                # 동시 등장이 없으면 같은 사람 — 묻지 않고 바로 병합.
                # (해제는 선수 목록에서 그룹 우클릭 → 병합 해제)
                self.merges[rep] = other
                self._merges_changed()
                self.log(f"[ptz] #{rep} → #{other} 자동 병합 "
                         f"(등번호 {num} 동일)")
                return
        self.player_nums[rep] = num
        if self._role_of(rep) not in (team, team + 3):   # GK 는 유지
            self.roles[rep] = team
        self._roles_changed()
        self.log(f"[ptz] #{rep} → {self.team_names[team]} {num}번")

    def _clear_player_num(self, tid):
        rep = self._rep(tid)
        n = (self.player_nums.pop(rep, None)
             or self.player_nums.pop(int(tid), None))
        if n:
            self._roles_changed()
            self.log(f"[ptz] #{rep} 등번호 {n} 해제")

    def _input_player_num(self, tid, team):
        from PyQt6.QtWidgets import QInputDialog
        num, ok = QInputDialog.getText(
            self, "등번호",
            f"#{self._rep(tid)} ({self.team_names[team]}) 등번호:",
            text=self._num_of(tid) or "")
        if ok and num.strip():
            self._set_player_num(tid, team, num.strip().split()[0])

    def _edit_roster(self, team):
        """팀 명단 입력 — 한 줄에 하나, "번호" 또는 "번호 이름"."""
        from PyQt6.QtWidgets import QInputDialog
        txt, ok = QInputDialog.getMultiLineText(
            self, f"{self.team_names[team]} 명단",
            "한 줄에 한 명 — \"7\" 또는 \"7 홍길동\":",
            "\n".join(self.rosters.get(team, [])))
        if not ok:
            return
        self.rosters[team] = [ln.strip() for ln in txt.splitlines()
                              if ln.strip()]
        self._save_keyframes()
        self.log(f"[ptz] {self.team_names[team]} 명단 "
                 f"{len(self.rosters[team])}명 저장")

    def _add_num_items(self, sub, tid, team):
        """팀 이름 바로 아래 들여쓴 번호/이름 항목 (서브메뉴 없이 선택).

        선택 = 그 팀 + 그 번호 지정. 직접 입력·명단 편집·해제는 맨 뒤
        작은 서브메뉴 하나로.
        """
        cur = self._num_of(tid)
        used = {}                        # 번호 → 이미 쓴 대표 tid
        for t, n in self.player_nums.items():
            if self._role_of(t) in (team, team + 3):
                used.setdefault(n, self._rep(t))
        entries = [(e.split()[0], e) for e in self.rosters.get(team, [])]
        roster_nums = {n for n, _ in entries}
        # 명단에 없어도 이미 입력된 번호는 바로 선택 가능하게
        entries += [(n, n) for n in sorted(
            (n for n in used if n not in roster_nums),
            key=lambda n: (not n.isdigit(),
                           int(n) if n.isdigit() else 0, n))]
        # OCR 제안이 있으면 무엇보다 위 — 검수 중 가장 유력한 선택지
        # (사용자 요청: 선택이 한 번에 되게)
        ocr = self._ocr_nums.get(self._rep(int(tid)))
        if ocr and ocr.get("num") and ocr["num"] != cur:
            sub.addAction(
                f"   ★ OCR 제안: {ocr['num']}번  "
                f"(지분 {float(ocr.get('share', 0)):.0%})",
                lambda _=False, n=str(ocr["num"]):
                self._set_player_num(tid, team, n))
        for num, label in entries:
            mark = " ✓" if cur == num else (
                f"  (#{used[num]})" if num in used
                and used[num] != self._rep(tid) else "")
            sub.addAction("      " + label + mark,
                          lambda _=False, n=num:
                          self._set_player_num(tid, team, n))
        if cur:
            sub.addAction(f"      등번호 해제 ({cur}번)",
                          lambda: self._clear_player_num(tid))
        more = sub.addMenu("      번호 입력/명단...")
        more.addAction("직접 입력...",
                       lambda: self._input_player_num(tid, team))
        more.addAction(f"{self.team_names[team]} 명단 입력/수정...",
                       lambda: self._edit_roster(team))

    def _edit_team_names(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("팀 이름")
        form = QFormLayout(dlg)
        e1, e2 = QLineEdit(self.team_names[0]), QLineEdit(self.team_names[1])
        form.addRow("팀 1 (홈)", e1)
        form.addRow("팀 2 (원정)", e2)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec():
            return
        self.team_names = [e1.text().strip() or "팀1",
                           e2.text().strip() or "팀2"]
        self._save_keyframes()
        self._apply_team_names()

    def _apply_team_names(self):
        """팀 이름 변경을 역할 버튼·범례·목록·타임라인에 반영."""
        for r, b in getattr(self, "_role_btns", {}).items():
            b.setText(self._role_name(r))
        self.trackbar.set_lane_names(*self.team_names)
        self._refresh_team_label()
        self._refresh_player_list()

    def show_match_stats(self):
        """분석 메뉴: 경기 지표 통계 창 (P08-3) — 계산은 core.metrics."""
        if self.analysis is None:
            QMessageBox.information(self, "경기 지표", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(
                self, "경기 지표",
                "경기장 캘리브레이션이 필요합니다 — 영상 우클릭 메뉴에서 "
                "랜드마크를 지정하세요 (지표는 필드 좌표 기준).")
            return
        self._auto_merge_if_needed(reason=" (지표 계산 전)")
        from ..core.metrics import match_metrics
        r = self._norm_export_range()
        t_range = (r[0] / self.fps, r[1] / self.fps) if r else None
        if t_range:
            self.log(f"[지표] 구간 한정: {t_range[0]/60:.1f}~"
                     f"{t_range[1]/60:.1f}분 (IN/OUT 마커)")
        with self._busy("경기 지표 계산"):
            m = match_metrics(self.analysis, self._field_calib,
                              self._role_of, self._rep,
                              pauses=(self.match_info or {}).get("pauses"),
                              t_range=t_range)
        if m is None or m["summary"] is None:
            QMessageBox.information(self, "경기 지표", "계산 실패 — 로그 확인")
            return
        from ..core.metrics import mean_positions, render_passmap
        from .stats import StatsDialog
        nums = {r: n for r, n in self.player_nums.items() if n}
        maps = []
        for team in (0, 1):
            pos = mean_positions(self.analysis, self._field_calib,
                                 self._role_of, self._rep, team)
            tp = [p for p in m["passes"] if p["team"] == team]
            if pos:
                maps.append(render_passmap(tp, pos, numbers=nums,
                                           title=self.team_names[team]))
        # 뛴 거리 표 (P08-3): 기존 리포트 계산 재사용 + 관측 비율 병기
        from ..core.report import movement_stats, player_field_tracks
        dur = ((t_range[1] - t_range[0]) if t_range
               else max(self.total / self.fps, 1e-6))
        dist_rows = []
        tracks = player_field_tracks(self.analysis, self._field_calib,
                                     self.merges, t_range=t_range)
        for rep, tr in tracks.items():
            role = self._role_of(rep)
            if role not in (0, 1, 3, 4):
                continue
            st = movement_stats(tr)
            if st["time_s"] < 60:
                continue
            dist_rows.append((self.team_names[0 if role in (0, 3) else 1],
                              nums.get(rep, str(rep)), st["dist_m"],
                              st["avg_mps"], st["max_mps"],
                              st["time_s"] / dur))
        dist_rows.sort(key=lambda r: (r[0], -r[2]))
        self.trackbar.set_possession(
            [(int(sp["t0"] * self.fps), int(sp["t1"] * self.fps),
              sp["team"]) for sp in m["spans"]])
        dlg = StatsDialog(self, m, team_names=tuple(self.team_names),
                          numbers=nums, passmaps=maps,
                          dist_rows=dist_rows,
                          save_dir=str(self.pano_path.parent
                                       / f"{self.pano_path.stem}_report"))
        dlg.show()
        self._stats_dlg = dlg             # GC 방지

    def _rep(self, tid):
        """트랙릿의 병합 대표 tid (병합 없으면 자기 자신)."""
        return self.merges.get(int(tid), int(tid))

    def _role_of(self, tid):
        """유효 역할 — 병합 그룹은 대표 기준 (역할 데이터는 비파괴).

        대표의 역할이 우선이라 팀 분류가 틀린 조각도 그룹에 넣으면
        표시가 바로잡히고, 분리하면 원래 분류로 돌아간다.
        """
        rep = self.merges.get(tid, tid)
        if rep in self.roles:
            return self.roles[rep]
        if tid in self.roles:
            return self.roles[tid]
        return self._teams.get(rep, self._teams.get(tid, 2))

    def _role_color(self, role):
        """역할의 표시색(BGR) — 사용자 지정 > 측정 대표색 > 기본 팔레트."""
        if role in self.kit_colors:
            return tuple(self.kit_colors[role])
        return self._role_colors.get(
            role, TEAM_COLORS[min(max(role, 0), len(TEAM_COLORS) - 1)])

    def _refresh_team_label(self, log_colors=False):
        """팀/GK/심판별 대표 유니폼 색 범례 (분석 후 '이 팀이 이 색').

        스와치 버튼 색 = 표시색(사용자 지정 우선, 없으면 측정 대표색).
        """
        spans, cols = self._player_cache()
        for r in self._kit_btns:
            self._kit_btns[r].hide()
            self._kit_lbls[r].setText("")
        if not self._teams or not cols:
            self.lbl_team_colors.setText("팀 색: 분석 후 표시")
            return
        logs = []
        self._role_colors = {}
        shown = 0
        for r in (0, 1, 3, 4, 5, 6):
            member = [t for t in cols
                      if self._role_of(t) == r and spans.get(t, [0, 0, 0])[2] >= 5]
            if not member:
                continue
            # 대표색: 검출 수 가중 없이 BGR 중앙값 (스와치 용도라 충분)
            pb = self._tid_bgr()
            bgr = np.median(np.array([pb[t] for t in member]), axis=0)
            self._role_colors[r] = _boost_bgr(
                (int(bgr[0]), int(bgr[1]), int(bgr[2])))
            b_, g_, r_ = self._role_color(r)          # 사용자 지정 우선
            hexc = f"#{r_:02x}{g_:02x}{b_:02x}"
            self._kit_btns[r].setStyleSheet(
                f"background-color: {hexc}; border: 1px solid #666;")
            self._kit_btns[r].show()
            mark = "✎" if r in self.kit_colors else ""
            self._kit_lbls[r].setText(f"{self._role_name(r)}{mark} {len(member)}명")
            shown += 1
            logs.append(f"{self._role_name(r)} {hexc} ({len(member)}트랙릿)")
        self.lbl_team_colors.setText("유니폼 색:" if shown
                                     else "팀 색: 분석 후 표시")
        self._radar_palette = {r: self._role_color(r) for r in range(7)}
        self.trackbar.set_role_palette(self._radar_palette)
        if log_colors and logs:
            self.log("[ptz] 팀 색 분류: " + ", ".join(logs))

    def _pick_kit_color(self, role):
        """범례 스와치 클릭 → 컬러피커로 역할 표시 색 지정."""
        b, g, r = self._role_color(role)
        c = QColorDialog.getColor(QColor(r, g, b), self,
                                  f"{self._role_name(role)} 표시 색")
        if not c.isValid():
            return
        self.kit_colors[role] = [c.blue(), c.green(), c.red()]
        self._kit_colors_changed()
        self.log(f"[ptz] {self._role_name(role)} 표시 색 지정: {c.name()}")

    def _reset_kit_color(self, role):
        if self.kit_colors.pop(role, None) is not None:
            self._kit_colors_changed()
            self.log(f"[ptz] {self._role_name(role)} 표시 색 → 측정색으로 복귀")

    def _kit_colors_changed(self):
        self._save_keyframes()
        self._refresh_team_label()
        self._redraw()

    def _refresh_player_list(self):
        """선수 트랙릿 목록: 유니폼 색 스와치 + 역할 + 구간 (검출 수 순)."""
        spans, cols = self._player_cache()
        # 재구성 전 선택을 tid 로 기억해 복원 — 역할 지정 등으로 목록이
        # 다시 만들어질 때 선택이 리셋되면, 이후 포커스 이동이 첫 행을
        # 선택하며 엉뚱한 프레임으로 점프하는 문제가 있었다.
        old_rows = getattr(self, "_player_rows", [])
        cur = self.player_list.currentRow()
        prev_cur = old_rows[cur] if 0 <= cur < len(old_rows) else None
        prev_sel = {old_rows[i.row()] for i in self.player_list.selectedIndexes()
                    if i.row() < len(old_rows)}
        # 병합 그룹: 대표 아래 멤버를 들여쓰기로 (해제/분리를 위해 표시 유지)
        groups: dict[int, list] = {}
        for t in spans:
            groups.setdefault(self._rep(t), []).append(t)
        agg = {rep: [min(spans[t][0] for t in ms),
                     max(spans[t][1] for t in ms),
                     sum(spans[t][2] for t in ms)]
               for rep, ms in groups.items()}
        # GK 는 목록 하단 그룹으로 (그 안에선 검출 수 순)
        reps = sorted(groups, key=lambda t: (self._role_of(t) in (3, 4),
                                             -agg[t][2]))
        self._player_rows = []
        for rep in reps:
            self._player_rows.append(rep)
            self._player_rows += sorted(
                (t for t in groups[rep] if t != rep),
                key=lambda t: spans[t][0])
        self.player_list.blockSignals(True)
        self.player_list.setUpdatesEnabled(False)
        self.player_list.clear()
        # 아이템 루프 밖에서 1회 — 루프 안 _tid_bgr() 는 _player_cache 를
        # 다시 부르고, 숨김 필터가 있으면 매번 전 트랙릿 재스캔이라
        # O(n²) (4천 트랙릿 실측 24s → 이 호이스팅으로 ~1s)
        tid_bgr = self._tid_bgr()
        for tid in self._player_rows:
            rep = self._rep(tid)
            k = len(groups.get(tid, ()))
            f0, f1, n = agg[tid] if tid == rep else spans[tid]
            r = self._role_of(tid)
            t0, t1 = f0 / self.fps, f1 / self.fps
            mark = "● " if (tid in self.roles or rep in self.roles) else ""
            head = (f"  ↳ #{tid}" if tid != rep
                    else f"#{tid}" + (f" (+{k - 1})" if k > 1 else ""))
            it = QListWidgetItem(
                f"{head}  {mark}{self._disp_role(tid, r)}  "
                f"{int(t0//60):02d}:{t0%60:04.1f}~"
                f"{int(t1//60):02d}:{t1%60:04.1f}  ({n}회)")
            c = tid_bgr.get(tid)
            if c is not None:
                px = QPixmap(14, 14)
                px.fill(QColor(c[2], c[1], c[0]))
                it.setIcon(QIcon(px))
            self.player_list.addItem(it)
        for row, tid in enumerate(self._player_rows):     # 선택 복원
            if tid in prev_sel:
                self.player_list.item(row).setSelected(True)
            if tid == prev_cur:
                self.player_list.setCurrentRow(row)
        self.player_list.setUpdatesEnabled(True)
        self.player_list.blockSignals(False)
        self.trackbar.set_players(
            {t: (spans[t][0], spans[t][1], self._role_of(t)) for t in spans},
            numbers={self._rep(t): self.player_nums[self._rep(t)]
                     for t in spans if self._rep(t) in self.player_nums},
            reps={t: self._rep(t) for t in spans})

    def _nearest_det_frame(self, tid, fclick=None):
        """트랙릿(대표 그룹)의 검출 중 fclick 에 가장 가까운 프레임.
        fclick 없으면 트랙릿 시작."""
        spans, _ = self._player_cache()
        if fclick is None or self.analysis is None:
            return int(spans[tid][0]) if tid in spans else self.slider.value()
        grp = {t for t in spans if self._rep(t) == self._rep(tid)}
        frames = self.analysis["frames"]
        best, bd = int(spans.get(tid, [fclick])[0]), 1 << 62
        for si, prow in enumerate(self.analysis["players"]):
            if any(len(p) >= 5 and int(p[4]) in grp for p in prow):
                d = abs(frames[si] - fclick)
                if d < bd:
                    best, bd = int(frames[si]), d
        return best

    def _goto_player(self):
        row = self.player_list.currentRow()
        rows = getattr(self, "_player_rows", [])
        if 0 <= row < len(rows):
            spans, _ = self._player_cache()
            self.slider.setValue(int(spans[rows[row]][0]))
            self.trackbar.set_selection("player", rows[row])
            self._redraw()                    # 선택 bbox 강조 즉시 반영

    def start_gapfill(self):
        """분석 메뉴: 갭필 2차 패스 — 트랙 갭을 저문턱 검출로 메꿈."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "갭필", "먼저 분석이 필요합니다.")
            return
        if self._gapfill_worker is not None and self._gapfill_worker.isRunning():
            self._gapfill_worker.cancel()
            self.log("[gapfill] 취소 요청")
            return
        targets = gapfill_targets(
            self.analysis,
            ignore_ranges=[tuple(r) for r in self.ignores],
            force_ranges=[tuple(p) for p in self.promotes],
            linked=self._linked)
        if not targets:
            QMessageBox.information(self, "갭필",
                                    "메꿀 갭이 없습니다 (≤4초 갭 기준).")
            return
        est = len(targets) * 0.17 / 60
        if QMessageBox.question(
                self, "갭필 2차 패스",
                f"트랙 갭 {len(targets)}지점을 저문턱 재검출합니다 "
                f"(예상 ~{est:.0f}분). 진행할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        w = GapfillWorker(str(self.pano_path), self.analysis, targets,
                          weights=self._model_weights())
        w.progress.connect(lambda i, t, f: (
            self.progress.setRange(0, t), self.progress.setValue(i),
            self.progress.setFormat(f"갭필 %p% ({f:.1f}/s)")))
        w.log.connect(self.log)
        w.done.connect(self._gapfill_done)
        w.failed.connect(lambda e: self.log(f"[gapfill] 실패: {e}"))
        self._gapfill_worker = w
        self.log(f"[gapfill] 시작: {len(targets)}지점")
        w.start()

    def _gapfill_done(self, analysis, nb, np_):
        self.analysis = analysis
        self._write_analysis()            # 주입 반영해 .analysis.json 갱신
        self._pcache_id = None            # 선수 캐시 무효화 (주입분 반영)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("갭필 완료")
        self.log(f"[gapfill] 완료: 공 {nb}개, 선수 {np_}명 주입 — 트랙 재연결")
        self._start_link()                # 링크·수락 재계산 → 목록/타임라인 갱신

    def detect_events(self):
        """분석 메뉴: 호각 × 대형 → 킥오프 검출 → .events.json + 타임라인."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "이벤트", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(self, "이벤트",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(대형 판정에 필드 좌표 사용).")
            return
        from ..core.audio import load_whistle_track
        from ..core.events import detect_kickoffs, formation_track, \
            save_events
        _, whistles = load_whistle_track(self.pano_path)
        if not whistles:
            QMessageBox.information(
                self, "이벤트",
                "호각 트랙이 없습니다 — scripts/whistle.py 를 먼저 "
                "실행하세요 (오디오 추출, ~20초).")
            return
        with self._busy("킥오프 검출 (대형 × 호각 융합)"):
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            tr = formation_track(self.analysis, teams, self._field_calib)
            ks = detect_kickoffs(tr, whistles)
            save_events(self.pano_path, ks)
        self._refresh_events()
        times = ", ".join(self._hms(t) for t, _, _ in ks) or "없음"
        self.log(f"[events] 킥오프 {len(ks)}개: {times}")

    def detect_highlights(self):
        """분석 메뉴: 이벤트 융합 → 하이라이트 후보 (.events.json, 비파괴)."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "하이라이트", "먼저 분석이 필요합니다.")
            return
        from ..core.audio import load_whistle_track
        from ..core.events import load_events_doc, save_events
        from ..core.highlights import (
            airborne_box_events, ball_speed_events, build_highlights,
            carry_states,
        )
        with self._busy("하이라이트 후보 생성 (이벤트 융합)"):
            doc = load_events_doc(self.pano_path)
            _, whistles = load_whistle_track(self.pano_path)
            air_ev, speed_ev = [], []
            if self._field_calib is not None:
                calib = self._field_calib
                air = doc.get("airborne") or {}
                air_ev = airborne_box_events(air.get("segments") or [],
                                             calib["length"], calib["width"])
                if self._accepted_ball is not None:
                    acc = np.asarray(self._accepted_ball, dtype=float)
                    fin = np.isfinite(acc[:, 0])
                    g = np.full((len(acc), 2), np.nan)
                    if fin.any():
                        g[fin] = pano_to_field(calib, acc[fin])
                    t = np.asarray(self.analysis["frames"], float) / self.fps
                    speed_ev = ball_speed_events(t, g)
            segs = build_highlights(
                self.total / self.fps,
                kickoffs=doc.get("kickoffs") or [],
                whistles=whistles or [],
                signals=doc.get("linesman_signals") or [],
                air_events=air_ev, speed_events=speed_ev,
                user_events=[(f / self.fps, lb) for f, lb in self.user_events])
            self.highlights = carry_states(segs, self.highlights)
            save_events(self.pano_path, highlights=self.highlights)
        self._refresh_highlight_lane()
        n_acc = sum(1 for h in self.highlights if h.get("state") == "accept")
        srcs = (f"킥오프 {len(doc.get('kickoffs') or [])}, 기 신호 "
                f"{sum(1 for s in doc.get('linesman_signals') or [] if (s.get('near') or {}).get('signal') in ('foul', 'offside'))}, "
                f"공중볼→박스 {len(air_ev)}, 속도 급증 {len(speed_ev)}, "
                f"사용자 {len(self.user_events)}")
        self.log(f"[hl] 하이라이트 후보 {len(self.highlights)}개 "
                 f"(수락 {n_acc}) — 소스: {srcs}")
        self.log("[hl] 이벤트 레인의 호박색 바 우클릭 → 수락/제외/경계 조정")
        if self._field_calib is None:
            self.log("[hl] 경기장 캘리브레이션 없음 — 공중볼/속도 규칙 생략됨")

    # ------------------------------------------------------ 하이라이트 검수
    def _add_manual_highlight(self, f=None):
        """수동 하이라이트 추가 — 자동이 놓친 장면 (빗나간 슛 등).

        f 기준 ±8초, f=None 이면 현재 IN/OUT 마커 구간. 수동 추가는
        검수가 끝난 것이므로 바로 수락 상태 — 재생성해도 보존된다.
        """
        if f is None:
            r = self._norm_export_range()
            if r is None:
                return
            t0, t1 = r[0] / self.fps, r[1] / self.fps
        else:
            t0 = max(0.0, f / self.fps - 8.0)
            t1 = min(self.total / self.fps, f / self.fps + 8.0)
        from PyQt6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getText(
            self, "하이라이트 추가",
            f"{self._hms(t0)} ~ {self._hms(t1)} 라벨 (예: 빗나간 슛):")
        if not ok or not label.strip():
            return
        self.highlights.append({"t0": round(t0, 2), "t1": round(t1, 2),
                                "kinds": ["user"], "label": label.strip(),
                                "score": 5.0, "state": "accept"})
        self.highlights.sort(key=lambda h: h["t0"])
        self._save_highlights()
        self.log(f"[hl] 수동 하이라이트 '{label.strip()}' "
                 f"{self._hms(t0)}~{self._hms(t1)} (수락 상태)")

    def _set_hl_state(self, i, state):
        if not 0 <= i < len(self.highlights):
            return
        h = self.highlights[i]
        h["state"] = state
        self._save_highlights()
        name = {"accept": "수락", "reject": "제외", "cand": "후보로 되돌림"}
        self.log(f"[hl] {name[state]}: {h.get('label', '')} "
                 f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def _hl_to_marks(self, i):
        """하이라이트 경계 → IN/OUT 마커 (미리보기·경계 조정용)."""
        h = self.highlights[i]
        self._set_export_mark("in", int(h["t0"] * self.fps))
        self._set_export_mark("out", int(h["t1"] * self.fps))
        self.slider.setValue(int(h["t0"] * self.fps))

    def _marks_to_hl(self, i):
        """현재 IN/OUT 마커 → 하이라이트 경계 (조정 커밋)."""
        r = self._norm_export_range()
        if r is None:
            return
        h = self.highlights[i]
        h["t0"], h["t1"] = round(r[0] / self.fps, 2), round(r[1] / self.fps, 2)
        self._save_highlights()
        self.log(f"[hl] 경계 갱신: {h.get('label', '')} "
                 f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def _del_highlight(self, i):
        if 0 <= i < len(self.highlights):
            h = self.highlights.pop(i)
            self._save_highlights()
            self.log(f"[hl] 삭제: {h.get('label', '')} "
                     f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def export_highlights(self):
        """분석 메뉴: 검수된 하이라이트 구간들을 개별 클립으로 일괄 렌더."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "하이라이트", "먼저 분석이 필요합니다.")
            return
        if self._render_worker is not None and self._render_worker.isRunning():
            QMessageBox.information(self, "하이라이트",
                                    "내보내기가 이미 진행 중입니다.")
            return
        cands = [h for h in self.highlights if h.get("state") != "reject"]
        if not cands:
            QMessageBox.information(
                self, "하이라이트",
                "내보낼 하이라이트가 없습니다 — 분석 메뉴에서 "
                "\"하이라이트 후보 생성\"을 먼저 실행하세요.")
            return
        self._stop_play()
        st = QSettings("PyStitch360", "PyStitch360")
        clock_avail = self._clock_config() is not None
        # 동기화된 다른 카메라 (sync_cams.py 결과, P06)
        alt = None
        try:
            from ..core.events import load_events_doc
            sync = load_events_doc(self.pano_path).get("sync")
            if sync and Path(sync.get("other", "")).exists():
                import re as _re
                label = _re.sub(r"[^0-9A-Za-z가-힣]+", "_",
                                Path(sync["other"]).stem)[:16] or "alt"
                alt = {"path": sync["other"],
                       "offset": float(sync["offset"]),
                       "drift": float(sync.get("drift", 1.0)),
                       "label": label}
                acap = cv2.VideoCapture(sync["other"])
                if acap.isOpened():        # 커버 구간 (앵글 뱃지용)
                    adur = (acap.get(cv2.CAP_PROP_FRAME_COUNT)
                            / (acap.get(cv2.CAP_PROP_FPS) or 30.0))
                    alt["span"] = (alt["offset"],
                                   alt["offset"] + alt["drift"] * adur)
                acap.release()
        except Exception as e:  # noqa: BLE001
            self.log(f"[hl] 동기화 정보 무시: {e}")
        dlg = HighlightExportDialog(
            self, cands, self.fps, self.encoders,
            int(st.value("ptz_export_crf", 20)),
            st.value("ptz_export_radar", "true") == "true",
            str(self.pano_path.parent),
            clock_on=(st.value("ptz_export_clock", "true") == "true"
                      if clock_avail else None),
            alt_label=alt["label"] if alt else None,
            alt_span=alt.get("span") if alt else None)
        if not dlg.exec():
            return
        cfg = dlg.config()
        if not cfg["indices"] or not cfg["dir"]:
            return
        st.setValue("ptz_export_crf", cfg["crf"])
        st.setValue("ptz_export_radar", "true" if cfg["radar"] else "false")
        if clock_avail:
            st.setValue("ptz_export_clock", "true" if cfg["clock"] else "false")
        import re
        segs = []
        for i in cfg["indices"]:
            h = cands[i]
            name = re.sub(r'[\\/:*?"<>|\s]+', "_",
                          h.get("label", "")).strip("_") or "장면"
            segs.append((int(h["t0"] * self.fps), int(h["t1"] * self.fps),
                         name))
        radar = None
        if cfg["radar"]:
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            radar = build_radar_data(
                self.analysis, teams, calib=self._field_calib,
                field_size=tuple(self.field_size),
                extra_players=self.extra_players,
                palette={r: self._role_color(r) for r in range(7)})
        dur = sum(e - s for s, e, _ in segs) / self.fps
        self.log(f"[hl] 일괄 내보내기 시작: {len(segs)}개 구간, "
                 f"총 {dur/60:.1f}분 → {cfg['dir']}")
        w = HighlightBatchWorker(
            str(self.pano_path), cfg["dir"], self.pano_path.stem,
            self.analysis, [tuple(k) for k in self.keyframes],
            self.encoders[cfg["codec_name"]], cfg["crf"],
            ignores=[tuple(r) for r in self.ignores],
            far_zoom=self.spin_far_zoom.value(),
            promotes=[tuple(p) for p in self.promotes],
            radar=radar, segments=segs,
            clock=self._clock_config() if cfg["clock"] else None,
            alt=alt if cfg["alt"] else None)
        w.log.connect(self.log)
        w.progress.connect(self._render_progress)
        w.finished_ok.connect(self._batch_done)
        w.failed.connect(self._render_failed)
        self._render_worker = w
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("준비 중...")
        w.start()

    def generate_report(self):
        """분석 메뉴: 팀/선수 히트맵 + 활동량 리포트 (PNG + markdown)."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "리포트", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(self, "리포트",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(히트맵은 필드 좌표 기준).")
            return
        self._auto_merge_if_needed(reason=" (리포트 생성 전)")
        from ..core.report import generate_report
        out_dir = self.pano_path.with_name(self.pano_path.stem + "_report")
        spans, _ = self._player_cache()
        roles_of = {self._rep(t): self._role_of(t) for t in spans}
        rr = self._norm_export_range()
        t_range = (rr[0] / self.fps, rr[1] / self.fps) if rr else None
        if t_range:
            self.log(f"[report] 구간 한정: {t_range[0]/60:.1f}~"
                     f"{t_range[1]/60:.1f}분 (IN/OUT 마커)")
        with self._busy("리포트 생성 (히트맵 + 활동량)"):
            r = generate_report(
                self.analysis, self._field_calib, roles_of, out_dir,
                merges=dict(self.merges),
                team_names=tuple(self.team_names), t_range=t_range,
                log=self.log)
        QMessageBox.information(
            self, "리포트",
            f"{len(r['files'])}개 파일 생성:\n{r['dir']}\n\n"
            f"선수 {len(r['rows'])}명 (players.md 요약표 포함)")

    def run_jersey_ocr(self):
        """분석 메뉴: 등번호 OCR — 근측(큰 박스) 선수 (devlog 040).

        rotcam(AX700 등)은 파노라마 필드 캘리브(_field_calib)가 없다 —
        calib=None 으로 넘기면 collect_ocr_candidates 가 필드 근/원 게이트
        대신 박스 높이 게이트만 쓴다(원경 선수는 작아서 대부분 걸러짐)."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "등번호 OCR", "먼저 분석이 필요합니다.")
            return
        rotcam = getattr(self, "_is_rotcam", False)
        calib = self._field_calib
        if calib is None and not rotcam:
            QMessageBox.information(self, "등번호 OCR",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(근측 게이트에 필드 좌표 사용).")
            return
        w_old = getattr(self, "_ocr_worker", None)
        if w_old is not None and w_old.isRunning():
            w_old.cancel()
            self.log("[ocr] 취소 요청")
            return
        from ..core.ocr import collect_ocr_candidates
        # 장외 트랙릿(관중 등)은 OCR 후보에서도 제외 — 역할을 -1 로 위장
        role_of = ((lambda t: -1 if self._rep(t) in self._outfield
                    else self._role_of(t))
                   if self._outfield else self._role_of)
        picked = collect_ocr_candidates(self.analysis, calib,
                                        role_of, self._rep)
        if not picked:
            gate = ("박스 높이 ≥90px" if calib is None
                    else "필드 Y<0, 박스 높이 ≥90px")
            QMessageBox.information(
                self, "등번호 OCR", f"근측 후보가 없습니다 ({gate}).")
            return
        n_rep = len({r for _, _, r in picked})
        if QMessageBox.question(
                self, "등번호 OCR",
                f"근측 트랙릿 {n_rep}개, 크롭 {len(picked)}장을 인식합니다 "
                f"(easyocr, 수 분 예상). 진행할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        w = OcrWorker(str(self.pano_path), self.analysis, picked)
        w.progress.connect(lambda d, t, f: (
            self.progress.setRange(0, t), self.progress.setValue(d),
            self.progress.setFormat(f"OCR %p% ({f:.1f}장/s)")))
        w.log.connect(self.log)
        w.done.connect(self._ocr_done)
        w.failed.connect(lambda m: self.log(f"[ocr] 실패: {m}"))
        self._ocr_worker = w
        self.log(f"[ocr] 시작: 근측 트랙릿 {n_rep}개, 크롭 {len(picked)}장")
        w.start()

    def _ocr_done(self, out):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("OCR 완료")
        try:
            from ..core.events import save_events
            save_events(self.pano_path, ocr_numbers=out)
        except Exception as e:  # noqa: BLE001
            self.log(f"[ocr] 저장 실패: {e}")
        self._ocr_nums = {int(k): v for k, v in out.items()}
        agree = sum(1 for k, v in self._ocr_nums.items()
                    if self._num_of(k) == v["num"])
        conflict = sum(1 for k, v in self._ocr_nums.items()
                       if self._num_of(k) and self._num_of(k) != v["num"])
        self.log(f"[ocr] 제안 {len(out)}건 (기존 지정과 일치 {agree}, "
                 f"충돌 {conflict}) — 선수 우클릭 번호 메뉴에 'OCR 제안'")

    # ------------------------------------------------------ 경기 정보/시계
    def edit_match_info(self):
        """분석 메뉴: 경기 정보 (시계 앵커·하프·중단 구간) 입력."""
        if self.pano_path is None:
            QMessageBox.information(self, "경기 정보", "열린 파노라마가 없습니다.")
            return
        kicks = []
        try:
            from ..core.events import load_events
            kicks = load_events(self.pano_path)
        except Exception as e:  # noqa: BLE001
            self.log(f"[clock] 킥오프 목록 무시: {e}")
        dlg = MatchInfoDialog(self, self.fps, self.total, kicks,
                              self.match_info, int(self.slider.value()))
        if not dlg.exec():
            return
        self.match_info = dlg.config()
        self._save_keyframes()
        mi = self.match_info
        anchor = ("미지정" if mi["anchor_f"] is None
                  else self._hms(mi["anchor_f"] / self.fps))
        self.log(f"[clock] 경기 정보: {'후반' if mi['half'] == 2 else '전반'} "
                 f"{mi['half_len_min']:.0f}분, 앵커 {anchor}, "
                 f"중단 {len(mi.get('pauses') or [])}개")

    def _clock_config(self):
        """render_plan 용 시계 설정 — 앵커 미지정이면 None.

        표기는 분:초 누적 (후반 +하프길이, 연장 90/120분도 분이 계속
        커짐). cv2 폰트 제약으로 태그는 1H/2H, 비ASCII 팀 이름은 T1/T2.
        """
        mi = self.match_info or {}
        if mi.get("anchor_f") is None:
            return None
        half = 2 if int(mi.get("half", 1)) == 2 else 1
        base = (float(mi.get("half_len_min", 45.0)) * 60.0
                if half == 2 and mi.get("cumulative", True) else 0.0)
        goals = []
        for f, lb in self.user_events:
            lb = str(lb).replace(" ", "")
            if lb == "골1":
                goals.append([int(f), 1])
            elif lb == "골2":
                goals.append([int(f), 2])
        score = None
        if goals:
            names = [n.strip() if n.strip() and n.isascii() else f"T{i + 1}"
                     for i, n in enumerate(self.team_names)]
            score = (names[0], names[1], goals)
        return {"anchor_f": int(mi["anchor_f"]), "fps": self.fps,
                "base_s": base, "tag": "2H" if half == 2 else "1H",
                "pauses": [[int(a), int(b)]
                           for a, b in mi.get("pauses") or []],
                "score": score}

    def _add_pause_range(self):
        """IN/OUT 마커 구간 → 경기 중단 (시계 정지, hydration break 등)."""
        r = self._norm_export_range()
        if r is None:
            return
        mi = self.match_info or {"half": 1, "half_len_min": 45.0,
                                 "anchor_f": None, "cumulative": True,
                                 "pauses": []}
        mi.setdefault("pauses", []).append([int(r[0]), int(r[1])])
        mi["pauses"].sort()
        self.match_info = mi
        self._save_keyframes()
        self.trackbar.set_pauses(mi["pauses"])
        self.log(f"[clock] 경기 중단 구간 추가: "
                 f"{self._hms(r[0] / self.fps)}~{self._hms(r[1] / self.fps)}"
                 f" (시계 정지, 총 {len(mi['pauses'])}개)")

    def _del_pause(self, i):
        pauses = (self.match_info or {}).get("pauses") or []
        if 0 <= i < len(pauses):
            p0, p1 = pauses.pop(i)
            self._save_keyframes()
            self.trackbar.set_pauses(pauses)
            self.log(f"[clock] 경기 중단 구간 삭제: "
                     f"{self._hms(p0 / self.fps)}~{self._hms(p1 / self.fps)}")

    def _batch_done(self, n, out_dir):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("완료")
        self.log(f"[hl] 일괄 내보내기 완료: {n}개 클립 → {out_dir}")
        QMessageBox.information(self, "하이라이트",
                                f"{n}개 클립 저장 완료:\n{out_dir}")

    def reset_edits(self, scope="all"):
        """사용자 편집 무효화 → 순수 분석 원본 상태 (분석 메뉴에서 호출).

        분석(.analysis.json)은 검수로 안 바뀌므로 에디트만 지우면 원본이다.
        """
        names = {"ball": "공 트랙 편집 (키프레임·무시·승격)",
                 "roles": "선수 역할 지정",
                 "field": "경기장 캘리브레이션",
                 "all": "모든 사용자 편집"}
        if self.pano_path is None:
            QMessageBox.information(self, "초기화", "열린 파노라마가 없습니다.")
            return
        n_extra = sum(len(v) for v in self.extra_players.values())
        detail = (f"키프레임 {len(self.keyframes)} / 무시 {len(self.ignores)}"
                  f" / 승격 {len(self.promotes)} / 역할 {len(self.roles)}"
                  f" / 병합 {len(self.merges)}"
                  f" / 랜드마크 {len(self.field_points)}"
                  f" / 수동 검출 {n_extra}")
        if QMessageBox.question(
                self, "분석 원본으로 되돌리기",
                f"{names[scope]}을(를) 삭제하고 분석 결과 원본으로 "
                f"되돌립니다.\n(현재: {detail})\n계속할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        if scope in ("ball", "all"):
            self.keyframes, self.ignores, self.promotes = [], [], []
            self._box_commit = None
        if scope in ("roles", "all"):
            self.roles = {}
            self.merges = {}
            self.player_nums = {}
            self.hidden_players = set()
            self._plan_dirty()
            self.extra_players = {}
            self.kit_colors = {}
            self._pcache_id = None
        if scope in ("field", "all"):
            self.field_points = {}
            self.field_point_frames = {}
            self.field_point_invalid = {}
            self.field_lines = {}
            self.field_point_anchors = {}
            self.line_points = []
        if self.analysis is not None:
            self._teams = classify_teams(self.analysis, roles=self.roles,
                                         feats=self._team_feats())
        self._write_sidecar()
        self._refit_field()
        self._refresh_field_list()
        self._refresh_team_label()
        self._refresh_player_list()
        self._refresh_lists()
        self._recompute_tracks()
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] {names[scope]} 초기화 — 분석 원본 상태로 되돌림")

    # ------------------------------------------------------ 내보내기 구간 마커
    def _norm_export_range(self):
        """정규화된 (f0, f1) — 마커가 하나도 없으면 None."""
        fi, fo = self.export_range
        if fi is None and fo is None:
            return None
        f0 = 0 if fi is None else int(fi)
        f1 = self.total if fo is None else int(fo)
        return (min(f0, f1), max(f0, f1))

    def _set_export_mark(self, kind, f, clear=False):
        if clear:
            self.export_range = [None, None]
            self.log("[ptz] 내보내기 구간 해제")
        else:
            i = 0 if kind == "in" else 1
            self.export_range[i] = int(f)
            self.log(f"[ptz] 내보내기 {'시작' if kind == 'in' else '끝'} "
                     f"마커 = {self._hms(f/self.fps, tenth=True)}")
        r = self._norm_export_range()
        self.trackbar.set_range(*(r if r else (None, None)))
        self._save_keyframes()

    def _timeline_menu(self, f, gpos):
        menu = QMenu(self)
        # 선수 레인에서 트랙릿 우클릭 → 그 선수 전용 메뉴만 (병합/분리/해체).
        # 내보내기/이벤트/하이라이트 등 일반 타임라인 메뉴는 띄우지 않는다.
        hit = getattr(self.trackbar, "_menu_hit", None)
        if hit is not None and hit[0] == "player":
            tid = int(hit[1])
            rep = self._rep(tid)
            self.trackbar.selected = hit          # 빨간 테두리 = 우클릭 대상
            self.trackbar.update()
            role = self._role_of(tid)
            menu.addAction(f"#{tid} — {self._disp_role(tid, role)}"
                           + (f"  {self._num_of(tid)}번"
                              if self._num_of(tid) else "")).setEnabled(False)
            menu.addSeparator()
            sel = getattr(self, "_tl_pair_sel", None)
            if sel is not None and sel != tid and self._rep(sel) != rep:
                snum = self._num_of(sel)
                menu.addAction(
                    f"#{sel}" + (f" ({snum}번)" if snum else "")
                    + f" 과(와) 병합 (같은 사람)",
                    lambda _=False, a=sel, b=tid: self._tl_merge_pair(a, b))
            menu.addAction("이 트랙릿을 병합 대상으로 선택",
                           lambda _=False, t=tid: self._tl_mark_pair(t))
            group = [t for t in self._pspans if self._rep(t) == rep]
            if tid != rep:
                menu.addAction(f"#{tid} 그룹에서 분리 (원래 분류로)",
                               lambda _=False, t=tid: self._split_tracklet(t))
            if len(group) > 1:
                menu.addAction(
                    f"그룹 해체 — #{rep} 외 {len(group) - 1}개 전부 분리",
                    lambda _=False, r=rep: self._dissolve_group(r))
            menu.exec(gpos)
            return
        lane = getattr(self.trackbar, "_menu_lane", -1)
        if 0 <= lane < len(self.trackbar.LANES):
            name = self.trackbar.lanes[lane]
            folded = lane in self.trackbar.collapsed
            solo = (self.trackbar.collapsed
                    == set(range(len(self.trackbar.lane_h))) - {lane})
            menu.addAction(("단독 확대 해제 (전체 펼치기)" if solo else
                            f"\"{name}\" 만 확대 — 나머지 접기"),
                           lambda _=False, ln=lane:
                           self.trackbar.solo_lane(ln))
            menu.addAction(("펼치기: " if folded else "접기: ")
                           + f"\"{name}\" 레인"
                           + ("" if folded else " (얇은 한 줄로)"),
                           lambda _=False, ln=lane:
                           self.trackbar.toggle_collapse(ln))
            if self.trackbar.collapsed and not solo:
                menu.addAction("접힌 레인 모두 펼치기",
                               self.trackbar.expand_all)
            menu.addSeparator()
        menu.addAction(f"내보내기 시작 지점 (I) — {self._hms(f/self.fps)}",
                       lambda: self._set_export_mark("in", f))
        menu.addAction(f"내보내기 끝 지점 (O) — {self._hms(f/self.fps)}",
                       lambda: self._set_export_mark("out", f))
        if self._norm_export_range():
            menu.addSeparator()
            r = self._norm_export_range()
            menu.addAction("마커 구간 재생 (OUT 에서 정지)",
                           lambda: self._play_segment(*r))
            menu.addAction("내보내기 구간 해제",
                           lambda: self._set_export_mark("", 0, clear=True))
        menu.addSeparator()
        menu.addAction("여기에 이벤트 추가...",
                       lambda: self._add_user_event(f))
        near = [i for i, (uf, _l) in enumerate(self.user_events)
                if abs(uf - f) <= 3 * self.fps]
        for i in near:
            menu.addAction(f"이벤트 '{self.user_events[i][1]}' 이름 바꾸기...",
                           lambda _=False, ii=i: self._rename_user_event(ii))
            menu.addAction(f"이벤트 '{self.user_events[i][1]}' 삭제",
                           lambda _=False, ii=i: self._del_user_event(ii))
        menu.addSeparator()
        menu.addAction("여기에 하이라이트 추가 (±8초)...",
                       lambda: self._add_manual_highlight(f))
        if self._norm_export_range():
            menu.addAction("IN/OUT 마커 구간 → 새 하이라이트...",
                           lambda: self._add_manual_highlight(None))
            menu.addAction("IN/OUT → 경기 중단 구간 (시계 정지)",
                           self._add_pause_range)
        for i, (p0, p1) in enumerate(
                (self.match_info or {}).get("pauses") or []):
            if p0 <= f <= p1:
                menu.addAction(
                    f"경기 중단 구간 삭제 — {self._hms(p0 / self.fps)}~"
                    f"{self._hms(p1 / self.fps)}",
                    lambda _=False, ii=i: self._del_pause(ii))
        hl = [i for i, h in enumerate(self.highlights)
              if h["t0"] * self.fps <= f <= h["t1"] * self.fps]
        for i in hl:
            h = self.highlights[i]
            tag = (f"'{h.get('label', '')}' "
                   f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")
            menu.addSeparator()
            menu.addAction(f"하이라이트 구간 재생 — {tag}",
                           lambda _=False, ii=i: self._play_segment(
                               self.highlights[ii]["t0"] * self.fps,
                               self.highlights[ii]["t1"] * self.fps))
            st = h.get("state", "cand")
            if st != "accept":
                menu.addAction(f"하이라이트 수락 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "accept"))
            if st != "reject":
                menu.addAction(f"하이라이트 제외 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "reject"))
            if st != "cand":
                menu.addAction(f"하이라이트 후보로 되돌림 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "cand"))
            menu.addAction("IN/OUT 마커 ← 하이라이트 경계 (조정 시작)",
                           lambda _=False, ii=i: self._hl_to_marks(ii))
            if self._norm_export_range():
                menu.addAction("하이라이트 경계 ← 현재 IN/OUT 마커 (조정 커밋)",
                               lambda _=False, ii=i: self._marks_to_hl(ii))
            menu.addAction(f"하이라이트 삭제 — {tag}",
                           lambda _=False, ii=i: self._del_highlight(ii))
        menu.addSeparator()
        menu.addAction("여기로 이동", lambda: self.slider.setValue(int(f)))
        menu.exec(gpos)

    def _add_user_event(self, f):
        from PyQt6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getText(
            self, "이벤트 추가",
            f"{self._hms(f/self.fps)} 이벤트 이름 (예: 골, 코너킥):")
        if not ok or not label.strip():
            return
        self.user_events.append([int(f), label.strip()])
        self.user_events.sort()
        self._save_keyframes()
        self._refresh_events()
        self.log(f"[ptz] 이벤트 '{label.strip()}' @ {self._hms(f/self.fps)}")

    def _del_user_event(self, i):
        if 0 <= i < len(self.user_events):
            f, label = self.user_events.pop(i)
            self._save_keyframes()
            self._refresh_events()
            self.log(f"[ptz] 이벤트 '{label}' 삭제")

    def _rename_user_event(self, i):
        """이벤트 이름 변경 — '골?' 검수 후 골1/골2 확정 (스코어 반영)."""
        if not 0 <= i < len(self.user_events):
            return
        from PyQt6.QtWidgets import QInputDialog
        f, old = self.user_events[i]
        label, ok = QInputDialog.getText(
            self, "이벤트 이름",
            f"{self._hms(f / self.fps)} 이벤트 이름\n"
            "(골1/골2 = 팀1/팀2 득점 — 시계 스코어에 집계):", text=old)
        if not ok or not label.strip() or label.strip() == old:
            return
        self.user_events[i][1] = label.strip()
        self._save_keyframes()
        self._refresh_events()
        self.log(f"[ptz] 이벤트 '{old}' → '{label.strip()}'")

    def suggest_goals(self):
        """분석 메뉴: 득점 역추론 — 경기 중 킥오프 = 직전 골 (P03-6 보조).

        첫 킥오프(하프 시작) 이후의 각 킥오프 앞에 '골?' 이벤트(−45s)와
        '골 추정' 하이라이트 후보([−70s, −5s])를 제안한다. 구간 재생으로
        확인 후 이벤트 이름을 골1/골2 로 바꾸면 스코어에 반영, 오탐이면
        삭제. 이미 근처에 골 이벤트가 있으면 건너뛴다 (재실행 안전).
        """
        if self.pano_path is None:
            QMessageBox.information(self, "득점 역추론",
                                    "열린 파노라마가 없습니다.")
            return
        from ..core.events import load_events
        ks = load_events(self.pano_path)
        if len(ks) < 2:
            QMessageBox.information(
                self, "득점 역추론",
                f"경기 중 킥오프가 없습니다 (검출 {len(ks)}개 — 첫 킥오프는 "
                "하프 시작). 먼저 \"킥오프 검출\"을 실행하세요.")
            return
        n_ev = n_hl = 0
        for k in ks[1:]:
            t = float(k["t"])
            near = [(uf, lb) for uf, lb in self.user_events
                    if (t - 120.0) * self.fps <= uf <= t * self.fps
                    and "골" in lb]
            if not near:
                self.user_events.append(
                    [int(max(0.0, t - 45.0) * self.fps), "골?"])
                n_ev += 1
            t0, t1 = max(0.0, t - 70.0), t - 5.0
            if not any("goal" in (h.get("kinds") or [])
                       and min(h["t1"], t1) - max(h["t0"], t0) > 0
                       for h in self.highlights):
                self.highlights.append(
                    {"t0": round(t0, 2), "t1": round(t1, 2),
                     "kinds": ["goal"], "label": "골 추정", "score": 4.0,
                     "state": "cand"})
                n_hl += 1
        if n_ev or n_hl:
            self.user_events.sort()
            self.highlights.sort(key=lambda h: h["t0"])
            self._save_keyframes()
            self._save_highlights()
            self._refresh_events()
        times = ", ".join(self._hms(float(k["t"])) for k in ks[1:])
        self.log(f"[goal] 경기 중 킥오프 {len(ks) - 1}개 ({times}) → "
                 f"'골?' 이벤트 {n_ev}개, '골 추정' 구간 {n_hl}개 제안")
        self.log("[goal] 검수: 구간 재생으로 확인 → 이벤트 이름을 "
                 "골1/골2 로 변경 (스코어 반영) 또는 삭제")

    def _tl_view_changed(self, t0, vis, total):
        """타임라인 줌/팬 → 스크롤바 동기화 (전체 보기면 숨김)."""
        if vis >= total - 1 or total <= 1:
            self.tl_scroll.hide()
            return
        self.tl_scroll.blockSignals(True)
        self.tl_scroll.setRange(0, int(total - vis))
        self.tl_scroll.setPageStep(max(1, int(vis)))
        self.tl_scroll.setSingleStep(max(1, int(vis / 20)))
        self.tl_scroll.setValue(int(t0))
        self.tl_scroll.blockSignals(False)
        self.tl_scroll.show()

    def _timeline_pick(self, kind, key, fclick=None):
        """타임라인 바 클릭 → 해당 목록 선택(→ 프레임 이동).

        선수 트랙릿은 시작이 아니라 **클릭 지점 최근접 검출**로 이동 —
        희소 트랙릿(검출 몇 개가 수십 분에 흩어짐)에서 시작으로 튀어
        아무도 안 잡히던 문제 수정.
        """
        if kind in ("kf", "ball"):
            want = ("kf" if kind == "kf" else "track", key)
            for row, e in enumerate(getattr(self, "_top", [])):
                if e == want:
                    self.track_list.setCurrentRow(row)   # currentRowChanged=이동
                    return
            if kind == "kf" and 0 <= key < len(self.keyframes):
                self.slider.setValue(int(self.keyframes[key][0]))
        elif kind == "ignore":
            if 0 <= key < len(self.ignores):
                self.kf_list.setCurrentRow(key)
                self.slider.setValue(int(self.ignores[key][0]))
        elif kind == "player":
            self.slider.setValue(self._nearest_det_frame(key, fclick))
            self.trackbar.set_selection("player", key)
            rows = getattr(self, "_player_rows", [])
            if key in rows:
                self.player_list.blockSignals(True)
                self.player_list.setCurrentRow(rows.index(key))
                self.player_list.blockSignals(False)
            self._redraw()
        elif kind == "event":
            evs = self.trackbar.events
            if 0 <= key < len(evs):
                self.slider.setValue(int(evs[key][0]))
        elif kind == "hl":
            if 0 <= key < len(self.highlights):
                self.slider.setValue(
                    int(self.highlights[key]["t0"] * self.fps))
        elif kind == "landmark":
            lm = self.trackbar.landmarks
            if 0 <= key < len(lm):
                self.trackbar.set_selection("landmark", key)
                self.slider.setValue(int(lm[key][0]))

    def _assign_selected_role(self, role):
        rows = getattr(self, "_player_rows", [])
        # 병합 멤버 선택은 대표에 지정 — 그룹(같은 사람) 전체에 적용
        sel = {self._rep(rows[i.row()])
               for i in self.player_list.selectedIndexes()
               if i.row() < len(rows)}
        for tid in sel:
            self._maybe_clear_num(
                tid, role if role is not None else self._teams.get(tid, 2))
            if role is None:
                self.roles.pop(tid, None)
            else:
                self.roles[tid] = role
        if sel:
            self._roles_changed()

    def _maybe_clear_num(self, tid, new_role):
        """팀이 바뀌면 등번호 제거 — 번호는 팀 소속 (GK 승격/강등은 유지)."""
        rep = self._rep(int(tid))
        num = self.player_nums.get(rep)
        if num is None:
            return
        old = self._role_of(rep)
        old_team = old % 3 if old in (0, 1, 3, 4) else None
        new_team = new_role % 3 if new_role in (0, 1, 3, 4) else None
        if old_team != new_team:
            self.player_nums.pop(rep, None)
            self.log(f"[ptz] #{rep} 팀 변경 — 등번호 {num} 제거")

    def _set_role(self, tid, role):
        """역할 지정 — 병합 대표에 저장 (그룹 = 한 사람).

        멤버 tid 에 저장하면 대표의 기존 역할이 우선돼(_role_of 순서)
        지정이 가려지는 버그가 있었다 — "팀을 바꿨는데 안 바뀜".
        """
        rep = self._rep(int(tid))
        eff = int(role) if role is not None else self._teams.get(rep, 2)
        self._maybe_clear_num(tid, eff)
        if role is None:
            self.roles.pop(rep, None)
            self.roles.pop(int(tid), None)
        else:
            self.roles[rep] = int(role)
            if int(tid) != rep:              # 멤버의 낡은 개별 역할 제거
                self.roles.pop(int(tid), None)
        self._roles_changed()

    def _roles_changed(self):
        self._ar_side_cache = {}
        if self.analysis is not None:
            self._teams = classify_teams(self.analysis, roles=self.roles,
                                         feats=self._team_feats())
        self._refresh_team_label()
        self._refresh_player_list()
        self._save_keyframes()
        self._redraw()

    # ------------------------------------------------------ 트랙릿 병합
    def _apply_merge_suggestions(self):
        """병합 제안 계산·적용 (공용) — 반환 (링크 수, 그룹 전/후)."""
        from ..core.tracklets import (
            merge_map, suggest_links, tracklet_summaries,
        )
        spans, _ = self._player_cache()
        n_before = len({self._rep(t) for t in spans})
        summ = tracklet_summaries(self.analysis, self._field_calib)
        roles_eff = {t: self._role_of(t) for t in summ}
        nums_eff = {t: self._num_of(t) for t in summ}
        links = suggest_links(summ, roles_eff, nums=nums_eff)
        all_links = ([(a, b) for a, b, _ in links]
                     + list(self.merges.items()))
        self.merges = merge_map(all_links,
                                {t: spans[t][2] for t in spans})
        self._merges_changed()
        n_after = len({self._rep(t) for t in spans})
        return len(links), n_before, n_after

    def _auto_merge_if_needed(self, reason=""):
        """병합이 하나도 없으면 제안을 자동 적용 (사용자 방향, 2026-07-22).

        제안은 보수적 게이트(시공간×색×역할×등번호)라 자동 적용해도
        비파괴 — 선수 목록에서 그룹 해체/분리로 언제든 되돌린다.
        수동/기존 병합이 있으면 검수 존중 — 건드리지 않는다.
        """
        if (self.analysis is None or self._field_calib is None
                or self.merges):
            return
        with self._busy("트랙릿 자동 병합 (첫 실행)"):
            n_links, nb, na = self._apply_merge_suggestions()
        self.log(f"[merge] 자동 병합{reason}: 링크 {n_links}개, "
                 f"그룹 {nb}→{na}개 — 선수 목록에서 검수/해체 가능")

    def suggest_tracklet_merges(self):
        """분석 메뉴: 트랙릿 병합 제안 (시공간 × 유니폼색 × 역할).

        기존 병합(수동 포함)은 유지하고 새 링크를 합친다 — 비파괴라
        목록 우클릭으로 언제든 그룹 해체/멤버 분리/추가 가능.
        """
        if self.analysis is None:
            QMessageBox.information(self, "병합", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(self, "병합",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(시공간 근접 판단에 필드 좌표 사용).")
            return
        with self._busy("트랙릿 병합 제안 (시공간 × 유니폼색 × 역할)"):
            n_links, n_before, n_after = self._apply_merge_suggestions()
        self.log(f"[merge] 링크 {n_links}개 제안 — "
                 f"그룹 {n_before}→{n_after}개 "
                 f"(선수 목록 우클릭으로 해제/분리/추가)")

    def _merges_changed(self):
        self._ar_side_cache = {}
        # 등번호를 새 대표로 이관 (충돌 시 대표 기존 번호 우선)
        for t in list(self.player_nums):
            rep = self._rep(t)
            if rep != t:
                self.player_nums.setdefault(rep, self.player_nums.pop(t))
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._redraw()

    def _player_menu(self, pos):
        """선수 목록 우클릭 — 병합/분리/해체 (전부 되돌리기 가능)."""
        rows = getattr(self, "_player_rows", [])
        sel = sorted({rows[i.row()] for i in self.player_list.selectedIndexes()
                      if i.row() < len(rows)})
        menu = QMenu(self)
        if len(sel) >= 2:
            menu.addAction(
                f"선택 {len(sel)}개 트랙릿 병합 (같은 사람)",
                lambda: self._merge_tracklets(sel))
        it = self.player_list.itemAt(pos)
        tid = (rows[self.player_list.row(it)]
               if it is not None and self.player_list.row(it) < len(rows)
               else None)
        if tid is not None:
            rep = self._rep(tid)
            group = [t for t in rows if self._rep(t) == rep]
            if tid != rep:
                menu.addAction(f"#{tid} 그룹에서 분리 (원래 분류로)",
                               lambda: self._split_tracklet(tid))
            if len(group) > 1:
                menu.addAction(
                    f"그룹 해체 — #{rep} 외 {len(group) - 1}개 전부 분리",
                    lambda: self._dissolve_group(rep))
        if self.hidden_players:
            menu.addSeparator()
            menu.addAction(f"숨긴 사람 {len(self.hidden_players)}명 전부 복원",
                           self._unhide_players)
        if not menu.isEmpty():
            menu.exec(self.player_list.mapToGlobal(pos))

    def _tl_mark_pair(self, tid):
        """타임라인에서 병합 대상 1개를 찍어둔다 (다음 우클릭에서 짝지음)."""
        self._tl_pair_sel = int(tid)
        self.log(f"[merge] #{tid} 병합 대상 선택 — 다른 트랙릿 우클릭으로 병합")

    def _tl_merge_pair(self, a, b):
        self._tl_pair_sel = None
        self._merge_tracklets([int(a), int(b)])

    def _merge_tracklets(self, tids):
        """선택 트랙릿 수동 병합 — 기존 그룹과 union (빠진 조각 추가 포함)."""
        nums = {self._num_of(t) for t in tids} - {None}
        if len(nums) > 1:
            QMessageBox.warning(
                self, "병합",
                "등번호가 다른 선수는 병합할 수 없습니다: "
                + ", ".join(sorted(nums)) + "번")
            return
        from ..core.tracklets import merge_map
        spans, _ = self._player_cache()
        links = ([(tids[0], t) for t in tids[1:]]
                 + list(self.merges.items()))
        self.merges = merge_map(links, {t: spans[t][2] for t in spans})
        self._merges_changed()
        self.log(f"[merge] 수동 병합: {', '.join(f'#{t}' for t in tids)}"
                 f" → 대표 #{self._rep(tids[0])}")

    def _split_tracklet(self, tid):
        """멤버 하나만 그룹에서 분리 — 역할은 원래 분류로 돌아간다."""
        rep = self.merges.pop(int(tid), None)
        if rep is None:
            return
        self._merges_changed()
        self.log(f"[merge] #{tid} 를 그룹 #{rep} 에서 분리")

    def _dissolve_group(self, rep):
        n = sum(1 for r in self.merges.values() if r == rep)
        self.merges = {t: r for t, r in self.merges.items() if r != rep}
        self._merges_changed()
        self.log(f"[merge] 그룹 #{rep} 해체 ({n}개 분리)")

    # ------------------------------------------------------ 경기장 캘리브레이션
    def _review_tab_changed(self, idx):
        if idx != 2:                      # 경기장 탭을 떠나면 찍기 모드 해제
            self.btn_field_pick.setChecked(False)
        self._redraw()

    def _field_tab_active(self):
        return self.tabs_review.currentIndex() == 2

    def _cam_field_pos(self):
        """카메라의 필드 좌표 (캘리브레이션 전엔 기본 가정값)."""
        if self._field_calib is not None:
            return (self._field_calib["ex"], self._field_calib["ey"])
        return (0.0, -(self.field_size[1] / 2.0 + 5.0))

    def _click_to_field(self, x, y):
        """클릭 픽셀 → 필드 좌표. 캘리브레이션 전엔 기본 카메라 모델."""
        if self._field_calib is not None:
            fx, fy = pano_to_field(self._field_calib, [[x, y]])[0]
            return (fx, fy) if np.isfinite(fx) else None
        g = ground_positions([[x, y, 0.0, 0.0]], self.pano_w, self.pano_h)
        if not g:
            return None
        cx, cy = self._cam_field_pos()
        return (cx + g[0][0], cy + g[0][1])

    # 캘리브레이션 전 자동 매칭 후보: 간격이 넓은 최외곽 점들만.
    # 페널티박스·센터서클은 기본 카메라 모델의 거리 오차(수십 m)보다
    # 이웃 간격이 좁아 오배정되기 쉬움 — 피팅이 잡힌 뒤(픽셀 매칭)에만.
    _PRECALIB_KEYS = ("corner_far_l", "corner_far_r", "corner_near_l",
                      "corner_near_r", "half_far", "half_near",
                      "sideline_near_l", "sideline_near_r")

    def _match_positions(self):
        pos = dict(landmark_positions(*self.field_size))
        # 선 위 점(사이드라인)은 대표 위치로 매칭: 카메라 좌우 1/4 지점
        hl, hw = self.field_size[0] / 2.0, self.field_size[1] / 2.0
        pos["sideline_near_l"] = (-hl / 2.0, -hw)
        pos["sideline_near_r"] = (hl / 2.0, -hw)
        pos["center_near"] = (0.0, -hw + 5.0)   # 중앙선 가까운 끝 대표점
        return pos

    def _match_landmark(self, x, y):
        """클릭 위치를 가장 그럴듯한 랜드마크에 휴리스틱 매칭.

        피팅 후: 전체 랜드마크를 화면에 투영해 픽셀 최근접 (정확).
        피팅 전: 외곽 8종만, 카메라 기준 방향(요)을 강하게·거리는
        로그 비율로 느슨하게 비교 — 기본 모델의 높이/거리 오차가
        모든 점을 같은 배율로 밀기 때문에 방향이 훨씬 믿을 만하다.
        미지정 랜드마크 우선, 전부 지정됐으면 최근접 이동(재클릭=수정).
        """
        pos = self._match_positions()
        if self._field_calib is not None:
            keys = list(pos)
            px = field_to_pano(self._field_calib, [pos[k] for k in keys])
            order = sorted(
                range(len(keys)),
                key=lambda i: (px[i][0] - x) ** 2 + (px[i][1] - y) ** 2
                if np.isfinite(px[i][0]) else np.inf)
            for i in order:
                if keys[i] not in self.field_points:
                    return keys[i]
            return keys[order[0]]
        f = self._click_to_field(x, y)
        if f is None:
            return None
        cx, cy = self._cam_field_pos()
        a_est = np.arctan2(f[0] - cx, f[1] - cy)
        d_est = max(np.hypot(f[0] - cx, f[1] - cy), 1e-6)

        def score(k):
            lx, ly = pos[k]
            da = a_est - np.arctan2(lx - cx, ly - cy)
            dd = np.log(d_est / max(np.hypot(lx - cx, ly - cy), 1e-6))
            return (da / 0.10) ** 2 + (dd / 0.4) ** 2

        order = sorted(self._PRECALIB_KEYS, key=score)
        for k in order:
            if k not in self.field_points:
                return k
        return order[0]

    def _field_set_point(self, key, x, y):
        self.field_points[key] = [round(float(x), 1), round(float(y), 1)]
        self.field_point_frames[key] = int(self.slider.value())
        self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_reassign(self, old, new):
        """찍힌 마커의 랜드마크 종류 변경 — 대상이 이미 있으면 서로 교환."""
        pt = self.field_points.pop(old, None)
        if pt is None or old == new:
            return
        if new in self.field_points:
            self.field_points[old] = self.field_points[new]
        self.field_points[new] = pt
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_remove_point(self, key):
        if self.field_points.pop(key, None) is not None:
            # 그 키의 프레임/무효구간/확정앵커도 함께 제거 — 안 그러면
            # 재지정 시 옛 프레임에서 이송/무효처리돼 엉뚱해진다.
            self.field_point_frames.pop(key, None)
            self.field_point_invalid.pop(key, None)
            self.field_point_anchors.pop(key, None)
            self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
            self._refit_field()
            self._save_keyframes()
            self._refresh_field_list()
            self._redraw()

    def _field_clear_selected(self):
        row = self.field_list.currentRow()
        if 0 <= row < len(LANDMARKS):
            self._field_remove_point(LANDMARKS[row][0])

    def _field_clear_all(self):
        if not (self.field_points or self.field_lines or self.line_points):
            return
        n = len(self.field_points) + len(self.field_lines)
        if QMessageBox.question(
                self, "랜드마크 전체 삭제",
                f"찍은 랜드마크/라인점 {n}개를 모두 "
                "삭제할까요?") != QMessageBox.StandardButton.Yes:
            return
        # 완전 초기화 — field_points 만 비우고 frames/invalid/lines 를
        # 남기면 클리어 후 재작업 시 이전 프레임/이전 라인점으로 이송돼
        # 엉뚱해진다 (reset_edits('field') 와 동일 집합).
        self.field_points = {}
        self.field_point_frames = {}
        self.field_point_invalid = {}
        self.field_lines = {}
        self.field_point_anchors = {}
        self.line_points = []
        self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
        self._rc_calib = None
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_pick_toggled(self, on):
        self._refit_field()          # 상태 라벨의 '다음 찍을 점' 안내 갱신
        self._refresh_field_list()   # 다음 항목 자동 선택
        self._redraw()

    def _field_next_key(self):
        """권장 순서상 다음 미지정 랜드마크 (없으면 None)."""
        for key, name, req in LANDMARKS:
            if key not in self.field_points:
                return key
        return None

    def _field_size_changed(self, _v=None):
        self.field_size = [self.spin_field_len.value(),
                           self.spin_field_w.value()]
        self._refit_field()
        self._save_keyframes()
        self._redraw()

    def _circle_r_changed(self, v):
        # 사용자가 직접 만지면 수동 모드로 (자동 추정 끔)
        if self.chk_circle_auto.isChecked():
            self.chk_circle_auto.blockSignals(True)
            self.chk_circle_auto.setChecked(False)
            self.chk_circle_auto.blockSignals(False)
            self.field_circle_auto = False
        self.field_circle_r = float(v)
        self._refit_field()
        self._save_keyframes()
        self._redraw()

    def _circle_auto_toggled(self, on):
        self.field_circle_auto = bool(on)
        if on:                            # 켜면 즉시 재추정
            self._refit_field()
            self.spin_circle_r.blockSignals(True)
            self.spin_circle_r.setValue(self.field_circle_r)
            self.spin_circle_r.blockSignals(False)
        self._save_keyframes()
        self._redraw()

    # ------------------------------------------------- venue 프리셋 (경기장 규격)
    def _venues_path(self):
        from pathlib import Path
        return Path(__file__).resolve().parent.parent.parent \
            / "presets" / "venues.json"

    def _load_venues(self):
        p = self._venues_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _refresh_venue_combo(self):
        self.combo_venue.blockSignals(True)
        self.combo_venue.clear()
        self.combo_venue.addItem("— venue 선택 —")
        for name in sorted(self._load_venues()):
            self.combo_venue.addItem(name)
        self.combo_venue.blockSignals(False)

    def _apply_venue_idx(self, idx):
        if idx <= 0:
            return
        name = self.combo_venue.itemText(idx)
        v = self._load_venues().get(name)
        if not v:
            return
        self.field_size = [float(v["length"]), float(v["width"])]
        self.field_circle_r = float(v.get("circle_r", CENTER_CIRCLE_R))
        self.field_circle_auto = False
        for sp, val in ((self.spin_field_len, self.field_size[0]),
                        (self.spin_field_w, self.field_size[1]),
                        (self.spin_circle_r, self.field_circle_r)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.chk_circle_auto.blockSignals(True)
        self.chk_circle_auto.setChecked(False)
        self.chk_circle_auto.blockSignals(False)
        self._refit_field()
        self._save_keyframes()
        self._redraw()
        self.log(f"[field] venue '{name}' 적용: {self.field_size[0]}×"
                 f"{self.field_size[1]}m, 서클R {self.field_circle_r}m")

    def _save_venue_dialog(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "venue 저장",
            f"경기장 이름 ({self.field_size[0]:.0f}×{self.field_size[1]:.0f}m, "
            f"서클R {self.field_circle_r}m):")
        if not ok or not name.strip():
            return
        name = name.strip()
        venues = self._load_venues()
        venues[name] = {"length": self.field_size[0],
                        "width": self.field_size[1],
                        "circle_r": self.field_circle_r}
        p = self._venues_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(venues, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self.log(f"[field] venue 저장 실패: {e}")
            return
        self._refresh_venue_combo()
        i = self.combo_venue.findText(name)
        if i >= 0:
            self.combo_venue.setCurrentIndex(i)
        self.log(f"[field] venue '{name}' 저장 → {p.name}")

    def _refresh_landmark_lane(self):
        """찍은 랜드마크(점+라인) → 타임라인 랜드마크 레인 마커."""
        marks = []
        for k, fr in self.field_point_frames.items():
            if k in self.field_points:
                marks.append((int(fr), LANDMARK_TAGS.get(k, k[:3]),
                              (0, 255, 255)))
        for key, pts in self.field_lines.items():
            col = LINE_FAMILIES[key][2]
            for _lx, _ly, fr in pts:
                marks.append((int(fr), "", col))
        # 확정 앵커 — 초록 틱 (안 찍었지만 이송을 눈으로 확인·확정한 프레임)
        for _k, pts in self.field_point_anchors.items():
            for _ax, _ay, fr in pts:
                marks.append((int(fr), "", (0, 200, 0)))
        self.trackbar.set_landmarks(marks)

    def _refresh_field_list(self):
        self._refresh_landmark_lane()
        row = self.field_list.currentRow()
        self.field_list.clear()
        for i, (key, name, req) in enumerate(LANDMARKS):
            p = self.field_points.get(key)
            mark = "★" if req else "☆"
            where = f"({p[0]:.0f}, {p[1]:.0f})" if p else "— 미지정"
            self.field_list.addItem(
                f"{i+1:2d}. {mark} [{LANDMARK_TAGS[key]}] {name}  {where}")
        nk = self._field_next_key() if self.btn_field_pick.isChecked() else None
        if nk is not None:                   # 찍기 모드: 다음 권장 점 하이라이트
            self.field_list.setCurrentRow(
                [k for k, _, _ in LANDMARKS].index(nk))
        elif row >= 0:
            self.field_list.setCurrentRow(row)

    def _refine_sideline(self):
        """예측 사이드라인 주변 흰 픽셀 검출 → 라인 샘플로 재피팅."""
        if self._field_calib is None:
            self.log("[field] 캘리브레이션 먼저 (랜드마크 4점 이상)")
            return
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        with self._busy("흰 선 검출 (사이드라인 정밀화)"):
            frame = self._native_frame(f)
            pts = (detect_sideline_points(self._field_calib, frame)
                   if frame is not None else [])
        if frame is None:
            return
        if len(pts) < 8:
            self.log(f"[field] 흰 선 샘플 부족 ({len(pts)}개) — "
                     "선이 프레임에 잘 보이는 프레임에서 다시 시도")
            return
        self.line_points = [[round(float(a), 1), round(float(b), 1)]
                            for a, b in pts]
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._redraw()
        self.log(f"[field] 사이드라인 흰 선 샘플 {len(pts)}개 반영")

    def _clear_line_points(self):
        """흰 선 정밀화 취소 — 샘플 제거 후 랜드마크만으로 재피팅."""
        if not self.line_points:
            self.log("[field] 지울 흰 선 샘플이 없습니다")
            return
        n = len(self.line_points)
        self.line_points = []
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._redraw()
        self.log(f"[field] 흰 선 샘플 {n}개 제거 — 랜드마크만으로 재피팅")

    def _lm_pos_at(self, k, ref):
        """랜드마크 k 를 프레임 ref 로 이송한 위치 (파노라마면 원위치)."""
        if k not in self.field_points:
            return None
        p = self.field_points[k]
        if not getattr(self, "_is_rotcam", False):
            return np.array(p, float)
        src = self.field_point_frames.get(k)
        if src is None or int(src) == int(ref):
            return np.array(p, float)
        H = self._rc_H(int(src), int(ref))
        if H is None:
            return None
        from ..core.rotcam import transfer_points
        return np.array(transfer_points(H, [p])[0], float)

    def _estimate_circle_r(self):
        """중앙선 위 4점(half_far/near, circle_far/near)의 사영 교차비로
        센터서클 반지름 R 을 역산 (경기장별로 R 이 다름 — 유소년은 작음).

        네 점은 월드에서 (0, +hw), (0, +R), (0, -R), (0, -hw) 로 한 직선
        위에 있고, 교차비는 사영 불변이라 이미지에서 R 을 닫힌형으로 푼다:
          CR = (hw+R)² / (4·hw·R)  →  R = hw·(k − √(k²−1)), k = 2·CR−1.
        4점이 없거나 퇴화면 None."""
        need = ["half_far", "circle_far", "circle_near", "half_near"]
        if not all(k in self.field_points for k in need):
            return None
        ref = (int(self.field_point_frames.get("circle_far",
                                               self.slider.value()))
               if getattr(self, "_is_rotcam", False) else 0)
        P = [self._lm_pos_at(k, ref) for k in need]
        if any(p is None for p in P):
            return None
        P = np.array(P)
        c = P.mean(0)
        u, s, vt = np.linalg.svd(P - c)
        t = (P - c) @ vt[0]                 # 직선 위 1D 좌표
        a, b, cc, dd = t
        den = (a - dd) * (b - cc)
        if abs(den) < 1e-9:
            return None
        cr = ((a - cc) * (b - dd)) / den
        hw = self.field_size[1] / 2.0
        k = 2.0 * cr - 1.0
        disc = k * k - 1.0
        if disc <= 0:
            return None
        R = hw * (k - np.sqrt(disc))        # (0, hw) 근
        if not (0.5 < R < hw - 0.5):
            return None
        return float(R)

    def _refit_field_rotcam(self):
        """회전 카메라 필드 캘리브레이션 — 여러 프레임의 랜드마크를 기준
        프레임으로 이송해 합치고 원근 모델(rotcam) 피팅 (P06-3).

        위치를 아는 랜드마크만 사용(선 위 점 제외). 이 캘리브레이션은
        이 영상 자체 좌표계 — 다른 영상과의 정렬(same/opposite side)은
        별도 단계(P06 대칭 판별)."""
        from ..core.field import LINE_LANDMARKS, VLINE_LANDMARKS, \
            landmark_positions
        from ..core.rotcam import (calibrate_reference,
                                   calibrate_reference_lines, transfer_points)
        # 센터서클 반지름 자동 추정 (중앙선 4점 있으면) — 경기장별 R
        if self.field_circle_auto:
            r = self._estimate_circle_r()
            if r is not None and abs(r - self.field_circle_r) > 0.05:
                self.field_circle_r = round(r, 2)
                if hasattr(self, "spin_circle_r"):
                    self.spin_circle_r.blockSignals(True)
                    self.spin_circle_r.setValue(self.field_circle_r)
                    self.spin_circle_r.blockSignals(False)
                self.log(f"[field] 센터서클 반지름 자동 추정: "
                         f"R = {self.field_circle_r}m (교차비)")
        pos = landmark_positions(self.field_size[0], self.field_size[1],
                                 circle_r=self.field_circle_r)
        keys = [k for k in self.field_points
                if k not in LINE_LANDMARKS and k not in VLINE_LANDMARKS
                and k in pos]
        self._rc_calib = None
        if len(keys) < 4:
            return
        # 기준 프레임 = 배치+확정앵커 프레임 최빈 (이송 비용 최소)
        rec = []
        for k in keys:
            f = self.field_point_frames.get(k)
            if f is not None:
                rec.append(int(f))
            rec += [int(a[2]) for a in self.field_point_anchors.get(k, ())]
        ref = int(max(set(rec), key=rec.count)) if rec else \
            int(self.slider.value())
        px, fld, skipped = [], [], 0
        for k in keys:
            got = self._rc_anchor_for(k, ref)      # 무효 체크 + 최단 앵커 이송
            if got is None:
                skipped += 1
                continue
            p, _src = got
            px.append(p)
            fld.append([float(pos[k][0]), float(pos[k][1])])
        if len(px) < 4:
            self._rc_skip = skipped
            return
        # 라인 점(터치라인 등) → 기준 프레임 이송 + 패밀리 (point-to-line)
        line_px, line_fam = [], []
        for key, pts in self.field_lines.items():
            fam = LINE_FAMILIES[key][0]
            for lx, ly, lfr in pts:
                p = [float(lx), float(ly)]
                if int(lfr) != ref:
                    H = self._rc_H(int(lfr), ref)
                    if H is None:
                        continue
                    p = [float(v) for v in transfer_points(H, [p])[0]]
                line_px.append(p)
                line_fam.append(fam)
        cal = calibrate_reference_lines(
            px, fld, line_px, line_fam, (self.pano_w, self.pano_h),
            length=self.field_size[0], width=self.field_size[1])
        if cal is not None:
            cal["ref_frame"] = ref
            self._rc_calib = cal
            self._rc_skip = skipped

    def _field_status(self):
        pass

    def _refit_field(self, log_result=False):
        self._field_calib = None
        if getattr(self, "_is_rotcam", False):
            self._refit_field_rotcam()
            # (이 아래 상태줄 갱신까지 흐르게 — 예전엔 여기서 early return
            #  이라 rotcam 은 lbl_field_status 가 이전 영상 값으로 고정됐다)
        elif self.pano_w and len(self.field_points) >= 4:
            self._field_calib = fit_field_calibration(
                self.field_points, self.pano_w, self.pano_h,
                length=self.field_size[0], width=self.field_size[1],
                line_points=self.line_points)
        self._update_outfield()           # 캘리브 변경 → 장외 재판정
        if getattr(self, "_is_rotcam", False):
            rc = getattr(self, "_rc_calib", None)
            npos = sum(1 for k in self.field_points
                       if k not in ("sideline_near_l", "sideline_near_r",
                                    "center_near"))
            if rc is not None:
                cp = rc["cam_pos"]
                sk = getattr(self, "_rc_skip", 0)
                rej = rc.get("n_rejected", 0)
                msg = (f"회전 카메라 캘리브레이션 OK — f {rc['f']:.0f}px, "
                       f"설치 ({cp[0]:+.0f},{cp[1]:+.0f},{cp[2]:.1f})m, "
                       f"잔차 {rc['rms_px']:.1f}px ({rc.get('n_points', 0)}점)"
                       + (f", 이상치 {rej}개 제외" if rej else "")
                       + (f", 이송실패 {sk}개" if sk else ""))
            elif npos >= 4:
                msg = ("회전 캘리브레이션 실패 — 프레임 간 이송 실패 "
                       "(랜드마크 프레임들이 특징 매칭 안 됨) 또는 배치 확인")
            else:
                msg = (f"{npos}점 찍음 — 위치 랜드마크 4개부터 풀림. "
                       "팬 카메라라 여러 프레임에 나눠 찍어도 됨 "
                       "(프레임 간 자동 이송)")
            c = None
        else:
            c = self._field_calib
        if c is not None:
            tilt = (f", 기울기 {np.degrees(c['pitch']):+.1f}°/"
                    f"{np.degrees(c['roll']):+.1f}°"
                    if c.get("pitch") or c.get("roll") else "")
            msg = (f"캘리브레이션 OK — {c['n_points']}점, 모델 잔차 "
                   f"{c['rms']:.1f}px (랜드마크는 워프로 고정), 높이 "
                   f"{c['h']:.1f}m, 터치라인 {-(c['ey'] + c['width']/2):.1f}m"
                   + tilt)
        elif not getattr(self, "_is_rotcam", False) and len(self.field_points) >= 4:
            msg = ("캘리브레이션 실패 — 점 위치 확인 "
                   "(위치 랜드마크 3개 이상 필요, 사이드라인 점은 보조)")
        elif not getattr(self, "_is_rotcam", False):
            msg = (f"{len(self.field_points)}점 찍음 — 위치 랜드마크 4개"
                   "(또는 3개+사이드라인 2점)부터 풀림, ★ 먼쪽 코너부터")
        if self.btn_field_pick.isChecked():
            nk = self._field_next_key()
            if nk is not None:
                i = [k for k, _, _ in LANDMARKS].index(nk)
                msg = (f"다음 찍을 점 → {i+1}. {LANDMARKS[i][1]} "
                       f"[{LANDMARK_TAGS[nk]}]   |   {msg}")
            else:
                msg = "모든 랜드마크 지정 완료   |   " + msg
        self.lbl_field_status.setText(msg)
        if log_result:
            self.log(f"[field] {msg}")

    def _players_row(self, si):
        """샘플 si 의 선수 행: 분석 + 수동 검출(extra), 숨김 제외.

        숨김(hidden_players, 관중·오인식)은 비파괴 — 분석 원본은 그대로,
        표시·목록·레이더에서만 빠진다. 병합 그룹 단위(대표 tid) 적용.
        """
        if self.analysis is None or si is None:
            return []
        rows = list(self.analysis["players"][si]) \
            + self.extra_players.get(int(si), [])
        excl = self._person_excluded()    # 수동 숨김 + 자동 장외
        if not excl:
            return rows
        return [p for p in rows
                if len(p) < 5 or p[4] < 0
                or self._rep(int(p[4])) not in excl]

    def _hidden_player_at(self, x, y):
        """(x, y) 아래 '숨긴' 사람의 대표 tid (없으면 None) — 회색 박스 복원용."""
        si = self._current_sample()
        if self.analysis is None or si is None or not self.hidden_players:
            return None
        raw = (list(self.analysis["players"][si])
               + self.extra_players.get(int(si), []))
        best, bestd = None, None
        for p in raw:
            if len(p) < 5 or p[4] < 0:
                continue
            rep = self._rep(int(p[4]))
            if rep not in self.hidden_players:
                continue
            if (abs(x - p[0]) <= p[2] / 2 + 10
                    and abs(y - p[1]) <= p[3] / 2 + 10):
                d = (x - p[0]) ** 2 + (y - p[1]) ** 2
                if best is None or d < bestd:
                    best, bestd = rep, d
        return best

    def _person_px_height(self, x, y):
        """(x, y) 지점에 선 사람(1.8m)의 예상 픽셀 키 — 캘리브레이션 기준.

        캘리브레이션 전이면 기본 카메라 모델로 대충 추정.
        """
        if self._field_calib is not None:
            c = self._field_calib
            g = pano_to_field(c, [[x, y]])[0]
            if not np.isfinite(g[0]):
                return None
            d = np.hypot(g[0] - c["ex"], g[1] - c["ey"])
            h, span = c["h"], c["t_top"] - c["t_bot"]
        else:
            gp = ground_positions([[x, y, 0.0, 0.0]], self.pano_w, self.pano_h)
            if not gp:
                return None
            d = np.hypot(gp[0][0], gp[0][1])
            h, span = 4.0, np.tan(np.deg2rad(10.0)) - np.tan(np.deg2rad(-38.0))
        return 1.8 / max(d, 1.0) / span * (self.pano_h - 1)

    _RC_DETW = 1600

    def _rc_gray(self, f):
        """프레임 f 의 det_w 그레이 (캐시) — 호모그래피 매칭용."""
        g = self._graycache.get(f)
        if g is None:
            fr = self._native_frame(f)
            if fr is None:
                return None
            sc = self._RC_DETW / fr.shape[1] if fr.shape[1] > self._RC_DETW else 1.0
            if sc < 1.0:
                fr = cv2.resize(fr, (self._RC_DETW, int(fr.shape[0] * sc)),
                                interpolation=cv2.INTER_AREA)
            g = (cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), sc)
            self._graycache[f] = g
        return g

    def _rc_H1(self, src, dst):
        """src → dst 직접 호모그래피 (원본 좌표계, 캐시). 실패 시 None."""
        if src == dst:
            return np.eye(3)
        key = (int(src), int(dst))
        if key in self._Hcache:
            return self._Hcache[key]
        from ..core.rotcam import match_frames
        ga, gb = self._rc_gray(src), self._rc_gray(dst)
        H = None
        if ga is not None and gb is not None:
            pa, pb = match_frames(ga[0], gb[0])
            if len(pa) >= 20:
                Hs, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
                # 인라이어 15 로 완화 — 약한 겹침(먼 팬)에서도 체인 연결
                if Hs is not None and mask is not None and mask.sum() >= 15:
                    S = np.diag([ga[1], ga[1], 1.0])
                    T = np.diag([gb[1], gb[1], 1.0])
                    H = np.linalg.inv(T) @ Hs @ S
        self._Hcache[key] = H
        return H

    def _rc_H(self, src, dst):
        """src → dst 이송 호모그래피. 직접 실패 시 랜드마크 프레임들을
        경유해 체인 (인접 프레임은 배경 특징으로 잘 매칭 — 고정 카메라).
        순수 회전이라 호모그래피가 정확히 합성된다. 실패 시 None."""
        H = self._rc_H1(src, dst)
        if H is not None:
            return H
        anchors = sorted(set(int(v) for v in self.field_point_frames.values())
                         | {int(src), int(dst)})
        if int(src) not in anchors or int(dst) not in anchors:
            return None
        i0, i1 = anchors.index(int(src)), anchors.index(int(dst))
        path = anchors[i0:i1 + 1] if i0 < i1 else anchors[i1:i0 + 1][::-1]
        Hm = np.eye(3)
        for a, b in zip(path, path[1:]):
            h = self._rc_H1(a, b)
            if h is None:
                return None
            Hm = h @ Hm
        return Hm

    def _rc_current_pose(self, cur=None):
        """현재 프레임의 카메라 자세 (R, f) — 위치·f 는 캘리브레이션 고정,
        회전만 이 프레임 랜드마크로 재추정 (rotcam.anchor_rotation).

        회전 카메라라 프레임마다 R 만 바뀐다: 기준 프레임과 특징 매칭이
        안 되는 프레임이라도 그 프레임에 랜드마크 3개만 있으면(찍었거나
        이송됐거나) 자세를 풀 수 있다 — 매칭 안 되는 프레임에 랜드마크
        찍어 라인을 되살리는 워크플로우 (사용자 방향). None = 불가."""
        rc = getattr(self, "_rc_calib", None)
        if rc is None:
            return None
        cur = int(self.slider.value() if cur is None else cur)
        cache = getattr(self, "_rc_pose_cache", None)
        if cache is None:
            cache = self._rc_pose_cache = {}
        # 랜드마크 지정/이동 시 무효화 키
        gen = getattr(self, "_rc_pose_gen", 0)
        if cache.get("_gen") != gen:
            cache.clear()
            cache["_gen"] = gen
        if cur in cache:
            return cache[cur]
        from ..core.field import (LINE_LANDMARKS, VLINE_LANDMARKS,
                                  landmark_positions)
        from ..core.rotcam import anchor_rotation
        pos = landmark_positions(self.field_size[0], self.field_size[1],
                                 circle_r=self.field_circle_r)
        cam = np.asarray(rc["cam_pos"])
        px, fld = [], []
        for k in self.field_points:
            if k in LINE_LANDMARKS or k in VLINE_LANDMARKS or k not in pos:
                continue
            got = self._rc_anchor_for(k, cur)     # 무효 체크 + 최단 앵커 이송
            if got is None:
                continue
            p, _src = got
            px.append(p)
            fld.append([float(pos[k][0]), float(pos[k][1])])
        got = None
        if len(px) >= 3:
            # f_span 넓게 — 팬 뿐 아니라 줌도 프레임마다 흡수 (AX700 은
            # 드물지만 줌 함, 사용자 지적). 회전+줌 동시 추정.
            got = anchor_rotation(cam, px, fld, rc["f"],
                                  (self.pano_w, self.pano_h), f_span=0.35)
        cache[cur] = got
        return got

    def _line_add_point(self, key, x, y):
        """터치라인 등 라인 위 점 추가 (다점 → point-to-line 캘리브)."""
        self.field_lines.setdefault(key, []).append(
            [round(float(x), 1), round(float(y), 1),
             int(getattr(self, "_cur_frame_idx", self.slider.value()))])
        n = len(self.field_lines[key])
        self._save_keyframes()
        self._refit_field(log_result=True)
        self._refresh_field_list()
        self._redraw()
        self.log(f"[field] {LINE_FAMILIES[key][1]} 점 추가 ({n}개)")

    def _line_clear(self, key):
        if self.field_lines.pop(key, None) is not None:
            self._save_keyframes()
            self._refit_field()
            self._refresh_field_list()
            self._redraw()
            self.log(f"[field] {LINE_FAMILIES[key][1]} 점 전체 삭제")

    def _line_point_at(self, x, y, r=40.0):
        """근처 라인 점 (key, index) — rotcam 이송 위치 기준 (삭제용)."""
        best, bestd = None, r * r
        rc = getattr(self, "_is_rotcam", False)
        cur = int(getattr(self, "_cur_frame_idx", self.slider.value()))
        for key, pts in self.field_lines.items():
            for i, (px, py, fr) in enumerate(pts):
                q = (px, py)
                if rc and int(fr) != cur:
                    H = self._rc_H(int(fr), cur)
                    if H is None:
                        continue
                    from ..core.rotcam import transfer_points
                    q = transfer_points(H, [[px, py]])[0]
                d = (q[0] - x) ** 2 + (q[1] - y) ** 2
                if d <= bestd:
                    best, bestd = (key, i), d
        return best

    def _line_del_point(self, key, i):
        pts = self.field_lines.get(key, [])
        if 0 <= i < len(pts):
            pts.pop(i)
            if not pts:
                self.field_lines.pop(key, None)
            self._save_keyframes()
            self._refit_field()
            self._refresh_field_list()
            self._redraw()

    def _lm_invalid_here(self, k, f=None):
        """랜드마크 k 가 프레임 f 에서 (이송이 엉뚱해) 무효로 표시됐는가."""
        if f is None:
            f = int(getattr(self, "_cur_frame_idx", self.slider.value()))
        for a, b in self.field_point_invalid.get(k, ()):
            if a <= f <= b:
                return True
        return False

    def _lm_add_invalid(self, k, f0, f1):
        """랜드마크 k 의 무효 구간 [f0,f1] 추가 (겹치면 병합)."""
        rngs = list(self.field_point_invalid.get(k, []))
        rngs.append([int(max(0, f0)), int(min(self.total - 1, f1))])
        rngs.sort()
        merged = []
        for a, b in rngs:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.field_point_invalid[k] = merged
        self._lm_invalid_changed()

    def _lm_clear_invalid(self, k):
        if self.field_point_invalid.pop(k, None) is not None:
            self._lm_invalid_changed()

    def _lm_remove_invalid_at(self, k, f):
        """프레임 f 를 덮는 무효 구간만 제거 (나머지 구간은 유지)."""
        rngs = [r for r in self.field_point_invalid.get(k, [])
                if not (r[0] <= f <= r[1])]
        if rngs:
            self.field_point_invalid[k] = rngs
        elif self.field_point_invalid.pop(k, None) is None:
            return
        self._lm_invalid_changed()

    def _lm_invalid_changed(self):
        """무효 구간 변경 → 프레임별 자세 캐시 무효화 + 재이송·재피팅·리드로."""
        self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
        self._save_keyframes()
        self._rc_transfer_landmarks()
        self._refit_field(log_result=True)
        self._refresh_field_list()
        self._redraw()

    def _rc_anchor_for(self, k, cur):
        """랜드마크 k 를 프레임 cur 로 이송할 최적 소스 → ([x,y], src_frame).
        원 배치 + 확정 앵커 중 cur 에 가장 가까운(=이송 체인 최단) 소스에서
        이송한다. cur 에 앵커/배치가 직접 있으면 그대로. 이 프레임 이송이
        무효이거나 이송 불가면 None. 팬 카메라라 먼 프레임끼리는 직접
        매칭이 안 되므로, 검증된 중간 앵커가 브리지 역할을 한다."""
        if self._lm_invalid_here(k, cur):
            return None
        cands = []                                # (frame, [x, y])
        p0 = self.field_points.get(k)
        if p0 is not None:
            f0 = self.field_point_frames.get(k)
            cands.append((int(f0) if f0 is not None else cur,
                          [float(p0[0]), float(p0[1])]))
        for ax, ay, af in self.field_point_anchors.get(k, ()):
            cands.append((int(af), [float(ax), float(ay)]))
        if not cands:
            return None
        cands.sort(key=lambda c: abs(c[0] - cur))     # cur 에 가까운 소스부터
        from ..core.rotcam import transfer_points
        for src, px in cands:
            if src == cur:
                return px, src
            H = self._rc_H(src, cur)
            if H is not None:
                q = transfer_points(H, [px])[0]
                return [float(q[0]), float(q[1])], src
        return None

    def _rc_transfer_landmarks(self):
        """찍은 랜드마크 + 확정 앵커를 현재 프레임으로 이송 → _lm_transferred.
        회전 카메라에서 팬을 따라 랜드마크가 움직이도록 (사용자 요청).
        각 랜드마크는 현재 프레임에 가장 가까운 소스에서 이송하며, 이
        프레임에서 무효인 것은 건너뛴다."""
        self._lm_transferred = {}
        if not getattr(self, "_is_rotcam", False):
            return
        cur = int(getattr(self, "_cur_frame_idx", self.slider.value()))
        for k in self.field_points:
            got = self._rc_anchor_for(k, cur)
            if got is not None:
                (x, y), _src = got
                self._lm_transferred[k] = (x, y)

    def _rc_confirm_frame(self):
        """현재 프레임의 이송 랜드마크를 확정 앵커로 일괄 승격.
        이송이 실제 라인과 맞는 걸 눈으로 확인한 뒤 호출 — 그 프레임을
        '찍은 것과 동등'한 앵커로 고정해 이후 이송의 최단 소스로 쓴다.
        원 배치 프레임(이미 앵커)인 것은 제외."""
        if not getattr(self, "_is_rotcam", False):
            return
        cur = int(getattr(self, "_cur_frame_idx", self.slider.value()))
        self._rc_transfer_landmarks()             # cur 기준 이송 최신화
        n = 0
        for k, (x, y) in list(self._lm_transferred.items()):
            src = self.field_point_frames.get(k)
            if src is not None and int(src) == cur:
                continue                          # 원 배치 프레임 — 스킵
            lst = self.field_point_anchors.setdefault(k, [])
            lst[:] = [a for a in lst if int(a[2]) != cur]   # 같은 프레임 갱신
            lst.append([round(float(x), 1), round(float(y), 1), cur])
            n += 1
        if not n:
            self.log(f"[field] 프레임 {cur}: 확정할 이송 랜드마크 없음")
            return
        self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
        self._save_keyframes()
        self._refit_field(log_result=True)
        self._refresh_field_list()
        self._redraw()
        self.log(f"[field] 프레임 {cur} 이송 {n}개 확정 (앵커 승격)")

    def _rc_unconfirm_frame(self):
        """현재 프레임의 확정 앵커 제거 (일괄)."""
        cur = int(getattr(self, "_cur_frame_idx", self.slider.value()))
        n = 0
        for k in list(self.field_point_anchors):
            keep = [a for a in self.field_point_anchors[k] if int(a[2]) != cur]
            if len(keep) != len(self.field_point_anchors[k]):
                n += 1
            if keep:
                self.field_point_anchors[k] = keep
            else:
                del self.field_point_anchors[k]
        if not n:
            return
        self._rc_pose_gen = getattr(self, "_rc_pose_gen", 0) + 1
        self._save_keyframes()
        self._refit_field(log_result=True)
        self._refresh_field_list()
        self._redraw()
        self.log(f"[field] 프레임 {cur} 확정 {n}개 해제")

    def _native_frame(self, f):
        """프레임 f 의 원본 해상도 이미지 (프록시 표시 중이면 원본을 읽음)."""
        # 이미 원본 해상도로 표시 중인 그 프레임이면 재디코드 없이 재사용
        # (원본 열기든 백그라운드 승격이든 _cur_scale 이 1.0 이면 원본).
        if getattr(self, "_cur_scale", 1.0) >= 1.0 \
                and getattr(self, "_cur_frame_idx", -1) == f \
                and getattr(self, "_cur_frame", None) is not None:
            return self._cur_frame
        # ffmpeg 빠른 시크 (긴 GOP·큰 프레임에서 OpenCV 시크는 수십 초)
        img = _ffmpeg_frame(self.pano_path, f / max(self.fps, 1e-9),
                            self.pano_w, self.pano_h)
        if img is not None:
            return img
        if self._native_cap is None:                # ffmpeg 실패 시 폴백
            self._native_cap = cv2.VideoCapture(str(self.pano_path))
        self._native_cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self._native_cap.read()
        return frame if ok else None

    def _detect_here(self, f, x, y, gpos):
        """빈 곳 우클릭 → 주변만 네이티브 해상도로 사람 재검출.

        크롭 크기는 캘리브레이션으로 예측한 그 자리 사람 키의 ~6배
        (마진 넉넉히). 검출된 사람은 수동 검출(extra)로 추가되고 바로
        역할 지정 메뉴를 띄운다.
        """
        si = self._current_sample()
        if si is None or not ptz_available():
            return
        ph = self._person_px_height(x, y) or 120.0
        half = int(np.clip(3.0 * ph, 160, 900))
        with self._busy("주변 사람 재검출 (YOLO 타일)"):
            frame = self._native_frame(f)
            if frame is None:
                return
            x0 = int(np.clip(x - half, 0, max(self.pano_w - 2 * half, 0)))
            y0 = int(np.clip(y - half, 0, max(self.pano_h - 2 * half, 0)))
            crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if self._adhoc is None:
                from ultralytics import YOLO
                w = self._model_weights()
                self._adhoc = YOLO(str(w) if w else "yolov8n.pt")
            imgsz = int(np.clip(2 * half, 320, 1280)) // 32 * 32
            r = self._adhoc.predict(crop, imgsz=imgsz, conf=0.1,
                                    classes=[0], verbose=False)[0]
        # 주변에 여럿 잡혀도 커서를 포함하는 박스 하나만 채택
        # (포함 박스가 여럿이면 가장 작은 것 = 가장 특정한 것).
        best, best_key = None, None
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            cx, cy = x0 + (x1 + x2) / 2, y0 + (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            inside = abs(x - cx) <= bw / 2 + 8 and abs(y - cy) <= bh / 2 + 8
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if not inside and d2 > max(150.0, ph) ** 2:
                continue                 # 커서에서 먼 박스는 무시
            key = (0, bw * bh) if inside else (1, d2)
            if best_key is None or key < best_key:
                best, best_key = (cx, cy, bw, bh, float(b.conf[0])), key
        if best is None:
            self.log(f"[ptz] 주변 재검출: 커서 위치에 사람 없음 "
                     f"(크롭 {2*half}px, 검출 {len(r.boxes)}건)")
            return
        cx, cy, bw, bh, conf = best
        # 기존(분석+수동) 박스와 겹치면 중복 추가하지 않음
        if any((p[0] - cx) ** 2 + (p[1] - cy) ** 2 < (p[3] / 2) ** 2
               for p in self._players_row(si) if len(p) >= 4):
            self.log("[ptz] 주변 재검출: 이미 있는 박스와 중복 — 추가 안 함")
            return
        tid = self._next_extra_id
        self._next_extra_id += 1
        self.extra_players.setdefault(int(si), []).append(
            [round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1), tid])
        skipped = len(r.boxes) - 1
        self.log(f"[ptz] 주변 재검출: 사람 1명 추가 (conf {conf:.2f}"
                 + (f", 주변 {skipped}건은 제외" if skipped > 0 else "")
                 + f", 크롭 {2*half}px)")
        self._save_keyframes()
        self._pcache_id = None           # 선수 목록 캐시 무효화
        self._refresh_player_list()
        self._redraw()
        # 바로 사람 지정 메뉴 — 팀 선택 아래 번호까지 한 번에 (사용자
        # 요청: 재검출 → 팀+번호 지정이 한 흐름). 기존 선수 우클릭
        # 메뉴와 동일 구조 (_add_num_items 재사용).
        menu = QMenu(self)
        for r in (0, 1):
            menu.addAction(self._role_name(r),
                           lambda _=False, rr=r, t=tid:
                           self._set_role(t, rr))
            self._add_num_items(menu, tid, r)
        menu.addSeparator()
        for rr in (3, 4, 5, 6):
            menu.addAction(f"{self._role_name(rr)} 지정",
                           lambda _=False, r_=rr, t=tid:
                           self._set_role(t, r_))
        menu.addSeparator()
        menu.addAction("역할 없이 두기", lambda: None)
        menu.exec(gpos)

    def _propagate(self, f, x, y, kind, ctx=None):
        """시드 전파 시작 — 수동 인식을 앞뒤 ±4s 로 확장."""
        if self.analysis is None:
            return
        if self._seed_worker is not None and self._seed_worker.isRunning():
            self.log("[seed] 이미 실행 중")
            return
        w = SeedWorker(str(self.pano_path), self.analysis, f, x, y, kind,
                       weights=self._model_weights(), ctx=ctx)
        w.log.connect(self.log)
        w.done.connect(self._seed_done)
        w.failed.connect(lambda e: self.log(f"[seed] 실패: {e}"))
        self._seed_worker = w
        self.log(f"[seed] {'공' if kind == 'ball' else '선수'} 추적 확장 "
                 f"시작 ({f/self.fps:.1f}s ±4s)...")
        w.start()

    def _seed_done(self, kind, matches, ctx):
        if not matches:
            self.log("[seed] 연결된 샘플 없음")
            return
        if kind == "ball":
            nb = 0
            for si, x, y, w_, h_, c in matches:
                cands = self.analysis["ball_cands"][si]
                if any(np.hypot(x - p[0], y - p[1]) <= 30 for p in cands):
                    continue
                cands.append([x, y, max(0.26, c), w_, h_, c])
                if self.analysis["balls"][si] is None:
                    self.analysis["balls"][si] = [x, y, max(0.26, c), w_, h_]
                nb += 1
            self._write_analysis()
            self.log(f"[seed] 공 {nb}샘플 주입 — 트랙 재연결")
            self._start_link()
        else:
            tid = int(ctx)
            np_ = 0
            for si, x, y, w_, h_, c in matches:
                rows = self.extra_players.setdefault(int(si), [])
                if any(p[4] == tid or np.hypot(x - p[0], y - p[1])
                       < max(w_ / 2, 20) for p in rows):
                    continue
                if any(len(p) >= 5 and
                       (p[0] - x) ** 2 + (p[1] - y) ** 2 < (p[3] / 2) ** 2
                       for p in self.analysis["players"][si]):
                    continue                 # 분석 검출과 중복
                rows.append([x, y, w_, h_, tid])
                np_ += 1
            self._save_keyframes()
            self._pcache_id = None
            self._refresh_player_list()
            self._redraw()
            self.log(f"[seed] 선수 #{tid - 900000} {np_}샘플로 확장")

    def _delete_extra(self, tid):
        for si, rows in list(self.extra_players.items()):
            self.extra_players[si] = [p for p in rows if p[4] != tid]
            if not self.extra_players[si]:
                del self.extra_players[si]
        self.roles.pop(tid, None)
        self._save_keyframes()
        self._pcache_id = None
        self._refresh_player_list()
        self._redraw()

    def _ball_rad(self, x, y, sc):
        """공 마커 반지름 — 그 자리 기대 공 크기에 비례 (원경 과대 방지).

        사람 키 추정(_person_px_height)의 ~9% ≈ 공 지름보다 약간 크게.
        """
        ph = self._person_px_height(x, y)
        if ph:
            return int(np.clip(ph * 0.09 * sc + 2, 4.0, max(9.0, 22 * sc)))
        return max(6, int(18 * sc))

    def _injected_person_at(self, x, y):
        """(x, y)가 ID 없는 주입 검출(갭필 id<0) 박스 안인지."""
        si = self._current_sample()
        if self.analysis is None or si is None:
            return False
        return any(len(p) >= 5 and p[4] < 0
                   and abs(x - p[0]) <= p[2] / 2 + 10
                   and abs(y - p[1]) <= p[3] / 2 + 10
                   for p in self._players_row(si))

    def _player_at(self, x, y):
        """(x, y)를 포함하는 현재 샘플의 선수 박스 track id (없으면 None)."""
        si = self._current_sample()
        if self.analysis is None or si is None:
            return None
        best, bestd = None, None
        for p in self._players_row(si):
            if len(p) < 5 or p[4] < 0:
                continue
            if (abs(x - p[0]) <= p[2] / 2 + 10
                    and abs(y - p[1]) <= p[3] / 2 + 10):
                d = (x - p[0]) ** 2 + (y - p[1]) ** 2
                if best is None or d < bestd:
                    best, bestd = int(p[4]), d
        return best

    def _sel_apply_role(self, role):
        """러버밴드 선택 전원에게 역할/팀 지정 (None = 자동 분류로)."""
        for rep in list(self._pane_sel):
            self._maybe_clear_num(
                rep, role if role is not None else self._teams.get(rep, 2))
            if role is None:
                self.roles.pop(rep, None)
            else:
                self.roles[rep] = role
        if self._pane_sel:
            self.log(f"[ptz] 선택 {len(self._pane_sel)}명 → "
                     + (self._role_name(role) if role is not None else "자동"))
            self._roles_changed()

    def _sel_hide(self):
        for rep in list(self._pane_sel):
            self.hidden_players.add(rep)
        n = len(self._pane_sel)
        self._pane_sel = set()
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._plan_dirty()                # 크롭 계획 재계산 (숨긴 사람 제외)
        self._redraw()
        self.log(f"[ptz] 선택 {n}명 숨김")

    def _players_in_rect(self, x1, y1, x2, y2):
        """사각형(파노라마 px) 안에 중심이 든 선수들의 대표 tid 집합."""
        si = self._current_sample()
        out: set[int] = set()
        if self.analysis is None or si is None:
            return out
        for p in self._players_row(si):
            if len(p) < 5 or p[4] < 0:
                continue
            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                out.add(self._rep(int(p[4])))
        return out

    def _toggle_pane_sel(self, tid, shift):
        """사람 클릭 선택 — shift=추가/삭제 토글, 아니면 그 사람만."""
        rep = self._rep(int(tid))
        if shift:
            if rep in self._pane_sel:
                self._pane_sel.discard(rep)
            else:
                self._pane_sel.add(rep)
        else:
            self._pane_sel = {rep}
        self._sync_list_to_pane_sel()
        self.log(f"[ptz] 선택 {len(self._pane_sel)}명")
        self._redraw()

    def _sync_list_to_pane_sel(self):
        """팬 러버밴드 선택 → 선수 목록 선택도 맞춤 (목록 우클릭 메뉴 재사용)."""
        rows = getattr(self, "_player_rows", [])
        if not rows:
            return
        sm = self.player_list.selectionModel()
        if sm is None:
            return
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel
        self.player_list.clearSelection()
        first = None
        for r, tid in enumerate(rows):
            if self._rep(tid) in self._pane_sel:
                idx = self.player_list.model().index(r, 0)
                sm.select(idx, QItemSelectionModel.SelectionFlag.Select)
                if first is None:
                    first = idx
        if first is not None:
            self.player_list.scrollTo(first)

    def _recompute_tracks(self):
        """수락 트랙 구간 재계산 → 트랙바 갱신 (분석/무시 구간 변경 시).

        linked 캐시가 있으면 수락 단계만 돌아 즉각 반응한다.
        """
        if self.analysis is None:
            self.track_spans = []
            self._accepted_ball = None
        else:
            _, self._accepted_ball, self.track_spans = accept_ball_tracks(
                self.analysis, ignore_ranges=[tuple(r) for r in self.ignores],
                force_ranges=[tuple(p) for p in self.promotes],
                linked=self._linked, log=self.log)
        self._recompute_airborne()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._refresh_track_list()
        self._plan_dirty()

    def _goto_track(self):
        row = self.track_list.currentRow()
        if 0 <= row < len(getattr(self, "_top", [])):
            kind, i = self._top[row]
            if kind == "track":
                f0, f1 = self.track_spans[i]
                if f0 <= self.slider.value() <= f1:
                    return          # 이미 그 트랙 안 — 시작으로 되감지 않음
                self.slider.setValue(int(f0))
            else:
                self.slider.setValue(int(self.keyframes[i][0]))

    def _jump_track(self, direction: int):
        """현재 위치 기준 이전(-1)/다음(+1) 수락 트랙 시작으로 이동."""
        self._stop_play()
        f = self.slider.value()
        if direction > 0:
            nxt = [sp for sp in self.track_spans if sp[0] > f]
            if nxt:
                self.slider.setValue(int(nxt[0][0]))
        else:
            prv = [sp for sp in self.track_spans if sp[0] < f - 1]
            if prv:
                self.slider.setValue(int(prv[-1][0]))

    def _to_ignore(self):
        """≫ 버튼: 목록 선택이 있으면 그 항목, 없으면 현재 시각 트랙."""
        if 0 <= self.track_list.currentRow() < len(getattr(self, "_top", [])):
            self._ignore_selected_track()
        else:
            self._ignore_current_track()

    def _ignore_selected_track(self, advance=False):
        """위 목록에서 Del — 자동 트랙은 오인식으로, 수동 지정은 삭제."""
        row = self.track_list.currentRow()
        if not (0 <= row < len(getattr(self, "_top", []))):
            return
        kind, i = self._top[row]
        if kind == "track":
            # 재생 위치는 건드리지 않고 그 트랙만 무시 — 정적 미끼는
            # 대개 영상 시작부터 있어서 시작으로 점프하면 처음으로 튄다
            self._ignore_current_track(anchor_f=int(self.track_spans[i][0]),
                                       advance=advance)
        else:
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_lists()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes,
                                   promotes=self.promotes)
            self._plan_dirty()
            self._redraw()

    def _ignore_current_track(self, anchor_f=None, advance=False):
        """anchor_f(기본: 현재 시각)를 덮는 수락 트랙을 통째로 무시.

        advance=True(키보드 검수 Del/→)일 때만 다음 항목으로 이동까지 —
        미리보기/버튼 경로에서는 보던 위치를 유지한다.
        """
        f = self.slider.value() if anchor_f is None else anchor_f
        for f0, f1 in self.track_spans:
            if f0 <= f <= f1:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    spans = [(int(f0), int(f1))]
                    if self._linked is not None:
                        # 같은 자리 반복 정적 트랙 일괄 수집 (낙엽·마킹).
                        # 위치 포함 4-요소 — 같은 시간대의 진짜 공 트랙은 보호
                        spans = same_spot_spans(self._linked, f0, f1) or spans
                    elif self._accepted_ball is not None:
                        # linked 전이라도 위치를 붙인다 — "이 트랙이 공이
                        # 아님"이지 시간대 전체 무시가 아니므로
                        si0 = int(np.searchsorted(self.analysis["frames"], f0))
                        si1 = int(np.searchsorted(self.analysis["frames"], f1))
                        pts = self._accepted_ball[si0:si1 + 1]
                        pts = pts[np.isfinite(pts).all(axis=1)] \
                            if len(pts) else pts
                        if len(pts):
                            mx, my = np.median(pts, axis=0)
                            spans = [(int(f0), int(f1),
                                      round(float(mx), 1),
                                      round(float(my), 1))]
                    added = 0
                    for sp in spans:
                        lo, hi = sp[0], sp[1]
                        if not any(a <= lo and hi <= b for a, b in
                                   ((r[0], r[1]) for r in self.ignores)):
                            self.ignores.append(list(sp))
                            added += 1
                    self.ignores.sort()
                    self._save_keyframes()
                    self._recompute_tracks()
                    self._refresh_kf_list()
                    self._redraw()
                finally:
                    QApplication.restoreOverrideCursor()
                extra = f" (같은 자리 반복 포함 {added}개 구간)" if added > 1 else ""
                self.log(f"[ptz] 트랙 무시: {f0/self.fps:.1f}s ~ "
                         f"{f1/self.fps:.1f}s{extra}")
                # 다음 항목 자동 선택 (시크 없이) — 이동은 키보드 검수만
                def _start(e):
                    return (self.track_spans[e[1]][0] if e[0] == "track"
                            else self.keyframes[e[1]][0])
                nxt = [r for r, e in enumerate(self._top) if _start(e) > f0]
                row = nxt[0] if nxt else len(self._top) - 1
                if row >= 0:
                    self.track_list.blockSignals(True)
                    self.track_list.setCurrentRow(row)
                    self.track_list.blockSignals(False)
                    if advance:
                        self._goto_track()
                return
        QMessageBox.information(self, "무시", "현재 시각을 덮는 공 트랙이 없습니다.")

    # ------------------------------------------------------------ 내보내기
    def _update_export_enabled(self):
        self.btn_export.setEnabled(self.analysis is not None)

    def _start_render(self):
        if self.analysis is None or self.pano_path is None:
            return
        self._stop_play()
        st = QSettings("PyStitch360", "PyStitch360")
        clock_avail = self._clock_config() is not None
        dlg = ExportDialog(
            self, self.total, self.fps, self._norm_export_range(),
            self.combo_mode.currentIndex(), self.encoders,
            int(st.value("ptz_export_crf", 20)),
            st.value("ptz_export_radar", "true") == "true",
            str(self.pano_path.parent), self.pano_path.stem,
            clock_on=(st.value("ptz_export_clock", "true") == "true"
                      if clock_avail else None))
        if not dlg.exec():
            return
        cfg = dlg.config()
        if not cfg["path"]:
            return
        st.setValue("ptz_export_crf", cfg["crf"])
        st.setValue("ptz_export_radar", "true" if cfg["radar"] else "false")
        if clock_avail:
            st.setValue("ptz_export_clock", "true" if cfg["clock"] else "false")
        wide = cfg["wide"]
        self.combo_mode.setCurrentIndex(1 if wide else 0)  # 미리보기 일치
        codec = self.encoders[cfg["codec_name"]]
        kfs = [tuple(k) for k in self.keyframes]
        radar = None
        if cfg["radar"]:
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            radar = build_radar_data(
                self.analysis, teams, calib=self._field_calib,
                field_size=tuple(self.field_size),
                extra_players=self.extra_players,
                palette={r: self._role_color(r) for r in range(7)})
            self.log("[ptz] 미니맵 오버레이 포함 "
                     + ("(경기장 절대 좌표)" if self._field_calib is not None
                        else "(캘리브레이션 없음 — 근사 좌표)"))
        dur = (cfg["end"] - cfg["start"]) / self.fps
        self.log(f"[ptz] 내보내기 시작: {'와이드' if wide else 'PTZ'} 모드, "
                 f"구간 {self._hms(cfg['start']/self.fps)}~"
                 f"{self._hms(cfg['end']/self.fps)} ({dur/60:.1f}분), "
                 f"키프레임 {len(kfs)}개 반영")
        clock = self._clock_config() if cfg["clock"] else None
        if clock is not None:
            self.log(f"[ptz] 경기 시계 포함 ({clock['tag']}, "
                     f"중단 {len(clock['pauses'])}개"
                     + (", 스코어" if clock["score"] else "") + ")")
        w = PtzRenderWorker(str(self.pano_path), cfg["path"], self.analysis,
                            kfs, codec, cfg["crf"], wide=wide,
                            ignores=[tuple(r) for r in self.ignores],
                            far_zoom=self.spin_far_zoom.value(),
                            promotes=[tuple(p) for p in self.promotes],
                            radar=radar, start=cfg["start"], end=cfg["end"],
                            clock=clock)
        w.log.connect(self.log)
        w.progress.connect(self._render_progress)
        w.finished_ok.connect(self._render_done)
        w.failed.connect(self._render_failed)
        self._render_worker = w
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("준비 중...")
        w.start()

    def _cancel_render(self):
        if self._render_worker is not None and self._render_worker.isRunning():
            self._render_worker.cancel()

    def _render_progress(self, done, total, fps):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _render_done(self, path):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("완료")
        self.log(f"[ptz] 저장: {path}")
        QMessageBox.information(self, "가상 PTZ", f"완료: {path}")

    def _render_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.log(f"[오류] PTZ 내보내기: {msg}")
