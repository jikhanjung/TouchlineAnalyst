"""멀티캠 뷰어 (P07-1): 동기화된 alt 카메라를 PtzTab 위에 얹는다.

구조는 "alt 뷰 목록 + 배치 전략(모드)" — 모드(PiP/좌우 분할/전환)는
선택된 alt 하나를 보여주는 최소 전략이고, 카메라가 늘면(4~8대) 그리드
등을 새 모드로 추가한다 (P07 확장 대비 항목).

- 편집은 항상 primary(파노라마) — alt 페인은 읽기 전용.
- 시간축은 primary 프레임 기준, 시계 모델(t_p = offset + drift·t_a)로
  환산해 읽기만 한다.
- 디코드는 전용 스레드에서 "최신 요청 우선" — 4K 랜덤 시크(수백 ms)가
  UI/재생을 막지 않고, 밀린 요청은 버린다.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from ..core.match import to_alt_time
from .settings import app_settings
from .widgets import FramePane


class AltDecodeWorker(QThread):
    """alt 프레임 디코더 — 카메라별 요청 슬롯, 슬롯마다 최신 요청만.

    v1 은 활성 alt 하나지만, v2 다중 페인(그리드 등)이 워커 수정 없이
    카메라 수만큼 request 를 걸 수 있는 구조로 둔다.
    """

    frame_ready = pyqtSignal(object, int, float)   # (BGR, cam_idx, t_alt)

    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()
        self._reqs: dict[int, tuple[str, float]] = {}   # cam_idx → 최신 요청
        self._stop = False
        self._caps: dict[str, cv2.VideoCapture] = {}   # 스레드 소유

    def request(self, path: str, t_alt: float, cam_idx: int):
        with self._cond:
            self._reqs[cam_idx] = (path, t_alt)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.wait(3000)

    def run(self):
        while True:
            with self._cond:
                while not self._reqs and not self._stop:
                    self._cond.wait()
                if self._stop:
                    break
                cam_idx, (path, t_alt) = self._reqs.popitem()
            cap = self._caps.get(path)
            if cap is None:
                cap = cv2.VideoCapture(path)
                self._caps[path] = cap
            if not cap.isOpened() or t_alt < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_alt * 1000.0))
            ok, frame = cap.read()
            if ok:
                self.frame_ready.emit(frame, cam_idx, t_alt)
        for c in self._caps.values():
            c.release()


class AltPane(FramePane):
    """읽기 전용 alt 페인 — PiP 모드에선 드래그 이동/우하단 리사이즈."""

    _HANDLE = 14                              # 우하단 리사이즈 존 (px)

    def __init__(self):
        super().__init__("동기화된 앵글", interactive=False)
        self.floating = False                 # True = PiP (부모 위 자식 위젯)
        self._drag = None                     # ("move"|"size", 시작 geo, 시작 pos)
        self.setStyleSheet(
            "background-color: #101010; color: #808080;"
            "border: 1px solid #505050;")
        from PyQt6.QtWidgets import QLabel
        self.lbl = QLabel("", self)           # 좌상단 제목 칩 (P07)
        self.lbl.setStyleSheet(
            "QLabel { color: white; background: rgba(20,20,20,170);"
            " padding: 1px 8px; border-radius: 4px; }")
        self.lbl.move(6, 6)
        self.lbl.hide()

    def set_title(self, text: str):
        self.lbl.setText(text)
        self.lbl.adjustSize()
        self.lbl.setVisible(bool(text))
        self.lbl.raise_()

    # PiP 전용 마우스 처리 — 분할 모드에선 스플리터가 담당하므로 무시
    def mousePressEvent(self, ev):
        if not self.floating:
            return
        p = ev.position()
        in_handle = (self.width() - p.x() <= self._HANDLE
                     and self.height() - p.y() <= self._HANDLE)
        self._drag = ("size" if in_handle else "move",
                      self.geometry(), ev.globalPosition())

    def mouseMoveEvent(self, ev):
        if not self.floating or self._drag is None:
            return
        kind, geo, p0 = self._drag
        d = ev.globalPosition() - p0
        if kind == "move":
            x = int(geo.x() + d.x())
            y = int(geo.y() + d.y())
            pw = self.parentWidget()
            x = max(0, min(x, pw.width() - self.width()))
            y = max(0, min(y, pw.height() - self.height()))
            self.move(x, y)
        else:
            w = max(160, int(geo.width() + d.x()))
            self.resize(w, max(90, int(w * 9 / 16)))

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            self._drag = None
            st = app_settings()
            st.setValue("mc_pip_geo", [self.x(), self.y(),
                                       self.width(), self.height()])


class MulticamViewer:
    """PtzTab 에 얹히는 컨트롤러 — 모드가 상위, 카메라는 모드 안의 배치.

    모드(사용자 확정 2026-07-21):
      - pip:   focus 카메라가 메인, 다른 카메라가 구석 PiP.
               focus=파노라마 → PiP 에 alt / focus=alt → PiP 에 파노라마.
      - split: 파노라마 | alt 좌우 동시 — focus 무관.
      - swap:  focus 카메라 하나만 (swap+1 = 순수 파노라마 화면).
    focus 는 숫자키/버튼: 0=파노라마(primary), 1..=alt.

    호출 계약 (PtzTab):
      - set_half(alts, redraw): 경기 열기/하프 전환 때 alt 목록 교체
      - update(t_primary, playing): alt 프레임 요청 (재생 중 스로틀)
      - primary_tick(frame): 현재 파노라마 프레임 공급 (PiP 용, 디코드 재사용)
      - main_frame(): 메인 페인에 파노라마 대신 그릴 프레임 (없으면 None)
      - alt_on_main: 메인이 alt 표시 중 (편집 입력 차단용)
      - close(): 디코드 스레드 정리
    """

    MODES = ("pip", "split", "swap")
    RATE_PLAY_S = 0.2                         # 재생 중 alt 갱신 주기 (~5fps)

    def __init__(self, pane, pane_split, log_fn):
        self._pane = pane                     # 메인 FramePane (PiP 부모)
        self._split = pane_split              # 가로 QSplitter (분할 모드)
        self._log = log_fn
        self.alts: list[dict] = []            # [{video, clock}]
        self.focus = 0                        # 0=primary, 1..=alt
        self.mode = str(app_settings()
                        .value("mc_mode", "pip"))
        if self.mode not in self.MODES:
            self.mode = "pip"
        self._last_t = None
        self._last_emit = 0.0
        self._main_frame = None               # focus>0 일 때 메인용 alt 프레임
        self._redraw = None                   # 메인 재그리기 콜백
        self.pane_alt = AltPane()
        self.pane_alt.hide()
        self._worker = AltDecodeWorker()
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.start()

    # ------------------------------------------------------------ 구성
    def set_half(self, alts: list[dict], redraw=None):
        self.alts = alts or []
        self.focus = 0
        self._main_frame = None
        self._redraw = redraw
        self._apply_layout()

    def set_focus(self, idx: int):
        """카메라 선택 (모드 안의 배치): 0=파노라마, 1..=alt."""
        if not (0 <= idx <= len(self.alts)):
            return
        self.focus = idx
        self._main_frame = None
        self._apply_layout()
        if self._redraw:
            self._redraw()
        if self._last_t is not None:
            self._request(self._last_t)

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self.mode = mode
            app_settings().setValue("mc_mode", mode)
            self._main_frame = None
            self._apply_layout()
            if self._redraw:
                self._redraw()
            if self._last_t is not None:
                self._request(self._last_t)

    def _apply_layout(self):
        p = self.pane_alt
        show_pane = bool(self.alts) and self.mode in ("pip", "split")
        if not show_pane:
            if p.parent() is self._split:
                p.setParent(None)
            p.hide()
            return
        if self.mode == "pip":
            if p.parent() is self._split:
                p.setParent(None)
            p.floating = True
            p.setParent(self._pane)
            geo = app_settings().value("mc_pip_geo")
            try:
                x, y, w, h = [int(v) for v in geo]
                p.setGeometry(x, y, w, h)
            except Exception:  # noqa: BLE001 — 첫 실행: 우상단 구석
                w = max(240, self._pane.width() // 4)
                p.setGeometry(self._pane.width() - w - 10, 10, w, w * 9 // 16)
            p.raise_()
            p.show()
        else:                                 # split — 오른쪽에 alt
            p.floating = False
            if p.parent() is not self._split:
                p.setParent(None)
                self._split.addWidget(p)
                self._split.setSizes([3, 2])
            p.show()

    # ------------------------------------------------------------ 배치 규칙
    def _shown_alt(self) -> int | None:
        """지금 표시해야 하는 alt 인덱스 (없으면 None).

        pip: focus=0 → 첫 alt 가 PiP / focus>0 → 그 alt 가 메인.
        split: 첫 alt (v1 — 다중 페인은 v2). swap: focus>0 인 alt.
        """
        if not self.alts:
            return None
        if self.mode == "split":
            return 0
        if self.focus > 0:
            return self.focus - 1
        return 0 if self.mode == "pip" else None

    @property
    def alt_on_main(self) -> bool:
        """메인 페인이 alt 를 표시 중 — 편집 입력 차단 조건."""
        return bool(self.alts) and self.mode != "split" and self.focus > 0

    # ------------------------------------------------------------ 갱신
    def update(self, t_primary: float, playing: bool = False):
        self._last_t = t_primary
        if self._shown_alt() is None:
            return
        now = time.monotonic()
        if playing and now - self._last_emit < self.RATE_PLAY_S:
            return
        self._last_emit = now
        self._request(t_primary)

    def primary_tick(self, frame_bgr, playing: bool = False):
        """현재 파노라마 프레임 — focus 가 alt 인 PiP 의 안쪽 그림.

        메인 디코드를 재사용하므로 공짜지만, 5900px 폭 변환이 아깝다 —
        PiP 크기에 맞춰 축소 + 재생 중 저주기.
        """
        if self.mode != "pip" or self.focus == 0 or frame_bgr is None:
            return
        now = time.monotonic()
        if playing and now - getattr(self, "_last_pip", 0.0) < self.RATE_PLAY_S:
            return
        self._last_pip = now
        w = frame_bgr.shape[1]
        tw = max(320, self.pane_alt.width())
        if w > tw * 1.5:
            h = int(frame_bgr.shape[0] * tw / w)
            frame_bgr = cv2.resize(frame_bgr, (tw, h),
                                   interpolation=cv2.INTER_AREA)
        self.pane_alt.set_frame(frame_bgr)

    def _request(self, t_primary: float):
        i = self._shown_alt()
        if i is None:
            return
        a = self.alts[i]
        self._worker.request(a["video"], to_alt_time(a["clock"], t_primary), i)

    def _alt_analysis(self, idx):
        """alt 의 .analysis.json 지연 로드 (없으면 False 캐시)."""
        cache = getattr(self, "_ana_cache", None)
        if cache is None:
            cache = self._ana_cache = {}
        if idx not in cache:
            cache[idx] = False
            try:
                import json
                p = Path(self.alts[idx]["video"]).with_suffix(
                    ".analysis.json")
                if p.exists():
                    d = json.loads(p.read_text())
                    import numpy as np
                    cache[idx] = {"t": np.asarray(d["frames"], float)
                                  / float(d["fps"]),
                                  "balls": d["balls"],
                                  "players": d["players"],
                                  "w": d.get("pano_w"), "h": d.get("pano_h")}
            except Exception as e:  # noqa: BLE001
                self._log(f"[mc] alt 분석 무시: {e}")
        return cache[idx]

    def _draw_overlay(self, frame, idx, t_alt):
        """alt 분석 사이드카의 공/선수 박스 (P07-2 일부, 픽셀 좌표)."""
        ana = self._alt_analysis(idx)
        if not ana:
            return frame
        import numpy as np
        si = int(np.argmin(np.abs(ana["t"] - t_alt)))
        if abs(ana["t"][si] - t_alt) > 0.5:
            return frame
        sx = frame.shape[1] / (ana["w"] or frame.shape[1])
        sy = frame.shape[0] / (ana["h"] or frame.shape[0])
        out = frame.copy()
        for pl in ana["players"][si]:
            cx, cy, w, h = pl[0] * sx, pl[1] * sy, pl[2] * sx, pl[3] * sy
            cv2.rectangle(out, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)),
                          (90, 220, 90), 2)
        b = ana["balls"][si]
        if b is not None:
            cv2.circle(out, (int(b[0] * sx), int(b[1] * sy)), 10,
                       (40, 180, 255), 3)
        return out

    def _on_frame(self, frame, cam_idx, t_alt=0.0):
        if cam_idx != self._shown_alt():
            return                            # 늦게 도착한 옛 카메라 프레임
        frame = self._draw_overlay(frame, cam_idx, t_alt)
        if self.alt_on_main:
            self._main_frame = frame          # 메인 페인으로 (pip/swap)
            if self._redraw:
                self._redraw()
        else:
            self.pane_alt.set_frame(frame)    # PiP/분할 페인으로

    def main_frame(self):
        """메인 페인에 파노라마 대신 그릴 alt 프레임 (아니면 None)."""
        return self._main_frame if self.alt_on_main else None

    def close(self):
        self._worker.stop()
        self.pane_alt.setParent(None)


class SyncRunWorker(QThread):
    """호각 추출(+없으면) → 거친 동기화 → .events.json "sync" 저장."""

    log = pyqtSignal(str)
    done = pyqtSignal(object)                 # sync dict 또는 None

    def __init__(self, primary: str, alt: str):
        super().__init__()
        self.primary, self.alt = str(primary), str(alt)

    def run(self):
        try:
            from ..core.audio import (
                extract_audio, load_whistle_track, save_whistle_track,
                whistle_events, whistle_track,
            )
            from ..core.events import save_events
            from ..core.sync_multi import sync_by_whistles
            evs = []
            for p in (self.primary, self.alt):
                _tr, ev = load_whistle_track(p)
                if not ev:
                    self.log.emit(f"[sync] 호각 추출: {Path(p).name} "
                                  "(오디오 전체 — 수 분)")
                    x = extract_audio(p)
                    tr = whistle_track(x)
                    ev = whistle_events(tr)
                    save_whistle_track(p, tr, ev)
                evs.append(ev)
            r = sync_by_whistles(evs[0], evs[1])
            if r is None:
                self.log.emit("[sync] 호각 매칭 실패")
                self.done.emit(None)
                return
            ppm = (r["drift"] - 1.0) * 1e6
            self.log.emit(f"[sync] {Path(self.primary).name} ↔ "
                          f"{Path(self.alt).name}: {r['n']}쌍, "
                          f"offset {r['offset']:+.2f}s, drift {ppm:+.1f}ppm, "
                          f"rms {r['rms_s'] * 1000:.0f}ms")
            sync = {"other": self.alt, "stage": "whistle",
                    "offset": round(r["offset"], 4), "drift": r["drift"],
                    "n_whistles": r["n"], "rms_s": round(r["rms_s"], 3)}
            save_events(self.primary, sync=sync)
            self.done.emit(sync)
        except Exception as e:  # noqa: BLE001 — 워커는 UI 로 보고만
            self.log.emit(f"[sync] 오류: {e}")
            self.done.emit(None)


class MatchBuildDialog(QDialog):
    """멀티캠 경기 만들기 — primary 하프들 + 하프별 alt 동기화.

    v1 흐름: 하프(파노라마) 추가 → alt 영상 추가(선택한 하프에 붙음,
    기존 sync 사이드카 있으면 재사용·없으면 여기서 실행) → 저장.
    """

    def __init__(self, parent, log_fn, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle("멀티캠 경기 만들기")
        self.resize(680, 420)
        self._log = log_fn
        self._dir = start_dir
        self.halves: list[dict] = []          # match.json "halves" 구조
        self._worker = None
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "하프(주 파노라마)를 순서대로 추가하고, 각 하프에 다른 카메라"
            " 영상을 붙이세요.\nalt 는 기존 동기화(.events.json)를 재사용하고,"
            " 없으면 호각 동기화를 즉시 실행합니다."))
        row = QHBoxLayout()
        self.list_halves = QListWidget()
        row.addWidget(self.list_halves, 1)
        col = QVBoxLayout()
        b_half = QPushButton("하프 추가 (파노라마)...")
        b_half.clicked.connect(self._add_half)
        b_alt = QPushButton("선택 하프에 앵글 추가...")
        b_alt.clicked.connect(self._add_alt)
        col.addWidget(b_half)
        col.addWidget(b_alt)
        col.addStretch(1)
        row.addLayout(col)
        v.addLayout(row)
        self.log_box = QPlainTextEdit(readOnly=True, maximumBlockCount=200)
        self.log_box.setMaximumHeight(110)
        v.addWidget(self.log_box)
        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        v.addWidget(self.bb)

    def _say(self, msg):
        self.log_box.appendPlainText(msg)
        self._log(msg)

    def _refresh(self):
        self.list_halves.clear()
        for h in self.halves:
            alts = ", ".join(Path(a["video"]).name for a in h["alts"]) or "—"
            self.list_halves.addItem(
                f"{h['label']}: {Path(h['primary']).name}  [앵글: {alts}]")

    def _add_half(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "주 파노라마 영상", self._dir, "영상 (*.mp4 *.MP4 *.mkv)")
        if not path:
            return
        self._dir = str(Path(path).parent)
        n = len(self.halves)
        label = "전반" if n == 0 else ("후반" if n == 1 else f"{n + 1}")
        # 기존 sync 사이드카가 있으면 alt 자동 채움
        from ..core.events import load_events_doc
        alts = []
        sync = load_events_doc(path).get("sync")
        if sync and Path(sync.get("other", "")).exists():
            alts.append({"video": sync["other"],
                         "clock": {"offset": sync["offset"],
                                   "drift": sync.get("drift", 1.0)},
                         "stage": sync.get("stage", "whistle")})
            self._say(f"[match] {Path(path).name}: 기존 동기화 재사용 → "
                      f"{Path(sync['other']).name}")
        self.halves.append({"label": label, "primary": path, "alts": alts})
        self._refresh()

    def _add_alt(self):
        i = self.list_halves.currentRow()
        if i < 0 or self._worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "다른 카메라 영상", self._dir, "영상 (*.mp4 *.MP4 *.mkv *.MOV)")
        if not path:
            return
        h = self.halves[i]
        self._say(f"[match] 동기화 실행: {Path(h['primary']).name} ↔ "
                  f"{Path(path).name}")
        self.bb.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        w = SyncRunWorker(h["primary"], path)
        w.log.connect(self._say)
        w.done.connect(lambda s, hh=h: self._alt_done(hh, s))
        self._worker = w
        w.start()

    def _alt_done(self, h, sync):
        self._worker = None
        self.bb.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
        if sync is None:
            self._say("[match] 앵글 추가 실패 — 로그 확인")
            return
        h["alts"].append({"video": sync["other"],
                          "clock": {"offset": sync["offset"],
                                    "drift": sync["drift"]},
                          "stage": sync["stage"]})
        self._refresh()

    def doc(self) -> dict | None:
        if not self.halves:
            return None
        return {"version": 1, "title": "", "halves": self.halves}
