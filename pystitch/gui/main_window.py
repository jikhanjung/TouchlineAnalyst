"""PyStitch360 메인 윈도우.

탭 구성:
  1. 동기화   — 좌/우 영상 나란히 표시, 챕터 통합 타임라인, 오디오 자동/수동 오프셋
  2. 정합     — 자동 정합 + pitch/roll/yaw 미세조정 + 파노라마 미리보기
  3. 내보내기 — 구간/코덱/CRF/해상도, 진행률
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..core.chapters import ChapteredVideo, find_chapters
from ..core.encoders import available_encoders
from ..core.lens import LensProfile, builtin_profiles
from ..core.project import load_project, save_project
from ..core.ptz import ptz_available
from .widgets import FramePane
from .workers import (
    AlignWorker, ExportWorker, GpmfWorker, PlaybackWorker, PreviewWorker,
    SyncWorker,
)


class MainWindow(QMainWindow):
    def __init__(self, with_ptz: bool = True):
        super().__init__()
        self._with_ptz = with_ptz         # False: PitchStitch — PTZ 탭/분석 메뉴 없음
        self._app_name = "PyStitch360"    # 런처가 덮어씀 (PitchStitch)
        self.setWindowTitle(self._app_name)
        self.resize(1500, 900)

        # 상태
        self.video_l: ChapteredVideo | None = None
        self.video_r: ChapteredVideo | None = None
        self.files_l: list[Path] = []
        self.files_r: list[Path] = []
        self.lens: LensProfile | None = None
        # 세그먼트: 방향이 (충격 등으로) 바뀐 구간마다 하나의 정합
        # [{"start_sec": float, "alignment": Alignment}, ...] 시작 시각 오름차순
        self.segments: list[dict] = []
        self.cur_imgs = (None, None)     # 현재 표시 중인 (L, R) 프레임
        # 워커 참조는 용도별로 분리 (실행 중 재할당 시 QThread 파괴 → 크래시)
        self._sync_worker = None
        self._align_worker = None
        self._preview_worker = None
        self._export_worker = None
        self._playback_worker = None
        self._playing = False

        tabs = QTabWidget()
        tabs.addTab(self._build_sync_tab(), "1. 영상·동기화")
        tabs.addTab(self._build_align_tab(), "2. 정합·미리보기")
        tabs.addTab(self._build_export_tab(), "3. 내보내기")
        if with_ptz:
            from .ptz_tab import PtzTab
            self.ptz_tab = PtzTab(self.log, self._last_video_dir,
                                  self._remember_video_dir)
            tabs.addTab(self.ptz_tab, "4. 가상 PTZ")
        else:
            self.ptz_tab = None
        tabs.currentChanged.connect(lambda _: self._stop_playback())
        self.tabs = tabs

        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumBlockCount(500)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(tabs)
        split.addWidget(self.log_box)
        split.setSizes([760, 140])
        self.setCentralWidget(split)
        # PTZ 탭은 자체 로그 탭이 있음 — 하단 로그는 1~3 탭에서만 표시
        tabs.currentChanged.connect(
            lambda i: self.log_box.setVisible(tabs.widget(i)
                                              is not self.ptz_tab))

        self.project_path: Path | None = None
        self._loaded_ptz_pano: str | None = None
        self._build_menu()
        self._load_profiles()

    # ------------------------------------------------------------ 메뉴/프로젝트

    def _build_menu(self):
        m = self.menuBar().addMenu("파일(&F)")
        m.addAction("새 프로젝트", self._new_project)
        m.addAction("프로젝트 열기...", self._open_project)
        self._recent_menu = m.addMenu("최근 프로젝트")
        m.addAction("프로젝트 저장", self._save_project)
        m.addAction("프로젝트 다른 이름으로 저장...", lambda: self._save_project(as_new=True))
        m.addSeparator()
        m.addAction("종료", self.close)
        if not self._with_ptz:            # PitchStitch — 분석 메뉴는 PitchWatch 로
            self._rebuild_recent_menu()
            return
        a = self.menuBar().addMenu("분석(&A)")
        a.addAction("갭필 2차 패스 (트랙 갭 재검출)...",
                    lambda: self.ptz_tab.start_gapfill())
        a.addAction("킥오프 검출 (호각 × 대형)",
                    lambda: self.ptz_tab.detect_events())
        a.addAction("트랙릿 병합 제안 (ReID 라이트)",
                    lambda: self.ptz_tab.suggest_tracklet_merges())
        a.addAction("등번호 OCR (파노라마는 근측만)...",
                    lambda: self.ptz_tab.run_jersey_ocr())
        a.addSeparator()
        a.addAction("하이라이트 후보 생성 (이벤트 융합)",
                    lambda: self.ptz_tab.detect_highlights())
        a.addAction("하이라이트 일괄 내보내기...",
                    lambda: self.ptz_tab.export_highlights())
        a.addAction("득점 역추론 (경기 중 킥오프 → 골 제안)",
                    lambda: self.ptz_tab.suggest_goals())
        a.addAction("경기 정보 (시계 앵커·중단 구간)...",
                    lambda: self.ptz_tab.edit_match_info())
        a.addAction("선수 히트맵/활동량 리포트",
                    lambda: self.ptz_tab.generate_report())
        a.addAction("경기 지표 (점유율·패스)...",
                    lambda: self.ptz_tab.show_match_stats())
        a.addSeparator()
        a.addAction("공/키프레임 편집 초기화",
                    lambda: self.ptz_tab.reset_edits("ball"))
        a.addAction("선수 역할 지정 초기화",
                    lambda: self.ptz_tab.reset_edits("roles"))
        a.addAction("경기장 캘리브레이션 초기화",
                    lambda: self.ptz_tab.reset_edits("field"))
        a.addSeparator()
        a.addAction("모든 사용자 편집 초기화 (분석 원본으로)",
                    lambda: self.ptz_tab.reset_edits("all"))
        self._rebuild_recent_menu()

    # ------------------------------------------------------------ 최근 프로젝트
    _MAX_RECENT = 10

    def _recent_projects(self) -> list[str]:
        v = QSettings("PyStitch360", "PyStitch360").value("recent_projects", [])
        if isinstance(v, str):          # QSettings 는 원소 1개 리스트를 str 로 돌려줄 수 있음
            v = [v]
        out, seen = [], set()
        for p in v or []:               # 기존에 쌓인 중복 표기도 정리
            c = self._canon(p)
            if c not in seen and Path(p).exists():
                seen.add(c)
                out.append(p)
        return out

    @staticmethod
    def _canon(path) -> str:
        """중복 판정용 정규화 — 슬래시/역슬래시·대소문자 표기 차이 흡수.

        Windows 에서 파일 대화상자는 D:/a/b, 내부 Path 는 D:\\a\\b 를
        돌려줘 같은 파일이 두 표기로 최근 목록에 쌓이는 문제가 있었다.
        """
        return os.path.normcase(os.path.normpath(str(path)))

    def _remember_recent(self, path):
        canon = self._canon(path)
        paths = [str(Path(path))] + [p for p in self._recent_projects()
                                     if self._canon(p) != canon]
        QSettings("PyStitch360", "PyStitch360").setValue(
            "recent_projects", paths[: self._MAX_RECENT])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        recent = self._recent_projects()
        self._recent_menu.setEnabled(bool(recent))
        for p in recent:
            self._recent_menu.addAction(
                f"{Path(p).name}  —  {Path(p).parent}",
                lambda checked=False, pp=p: self._open_recent(pp))
        if recent:
            self._recent_menu.addSeparator()
            self._recent_menu.addAction("목록 비우기", self._clear_recent)

    def _clear_recent(self):
        QSettings("PyStitch360", "PyStitch360").remove("recent_projects")
        self._rebuild_recent_menu()

    def _open_recent(self, path: str):
        self._busy(True, flush=True)
        try:
            d = load_project(path)
            self._apply_project(d)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "열기 실패", str(e))
            return
        finally:
            self._busy(False)
        self.project_path = Path(path)
        self.setWindowTitle(f"{self._app_name} — {self.project_path.name}")
        self.log(f"[project] 열기: {path}")
        self._remember_recent(path)

    def _busy(self, on: bool, flush: bool = False):
        """블로킹 작업/백그라운드 대기 중 wait cursor 표시 (스택 균형 유지할 것).

        flush=True 면 커서 변경을 즉시 화면에 반영 — 이어서 GUI 스레드가
        블로킹되는 경우(프로젝트/영상 열기) 필수. 단 이벤트 재진입 위험이
        있으니 슬라이더 디바운스 같은 고빈도 경로에서는 쓰지 말 것.
        """
        if on:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            if flush:
                QApplication.processEvents()
        else:
            QApplication.restoreOverrideCursor()

    def current_time(self) -> float:
        return self.slider.value() / self.video_l.fps if self.video_l else 0.0

    def current_alignment(self):
        """현재 슬라이더 시각을 담당하는 세그먼트의 정합."""
        t = self.current_time()
        chosen = None
        for s in self.segments:
            if s["start_sec"] <= t + 1e-6:
                chosen = s
        return (chosen or (self.segments[0] if self.segments else None) or {}).get("alignment")

    def _gather_project(self) -> dict:
        segments = self.segments
        return {
            "left_files": [str(p) for p in self.files_l],
            "right_files": [str(p) for p in self.files_r],
            "offset_sec": self.spin_offset.value(),
            "lens_profile": self.profile_combo.currentText(),
            "segments": segments,
            "user": {k: sp.value() for k, sp in self.spin_user.items()}
                    | {"feather_px": self.spin_feather.value(),
                       "el_top_deg": self.spin_el_top.value(),
                       "el_bottom_deg": self.spin_el_bottom.value()},
            "export": {
                "start": self.spin_start.value(), "end": self.spin_end.value(),
                "codec_index": self.combo_codec.currentIndex(),
                "crf": self.spin_crf.value(),
                "scale_index": self.combo_scale.currentIndex(),
                "perspective": {
                    "enabled": self.check_persp.isChecked(),
                    "k": self.spin_persp_k.value(),
                    "m": self.spin_persp_m.value(),
                },
            },
            # PTZ 분석/키프레임 자체는 파노라마 옆 사이드카에 저장되고,
            # 프로젝트는 어떤 파노라마를 열어놨는지만 기억한다
            # PitchStitch(ptz_tab 없음)에선 프로젝트에 있던 pano 참조를 보존
            "ptz": {"pano": (str(self.ptz_tab.pano_path)
                             if self.ptz_tab.pano_path else None)
                    if self.ptz_tab is not None else self._loaded_ptz_pano},
        }

    def _save_project(self, as_new=False):
        if self.project_path is None or as_new:
            path, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장",
                                                  "project.pystitch.json",
                                                  "PyStitch 프로젝트 (*.json)")
            if not path:
                return
            self.project_path = Path(path)
        try:
            save_project(self.project_path, self._gather_project())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "저장 실패", str(e))
            return
        self.setWindowTitle(f"{self._app_name} — {self.project_path.name}")
        self.log(f"[project] 저장: {self.project_path}")
        self._remember_recent(self.project_path)

    def _new_project(self):
        """새 프로젝트 — 초기 상태의 새 메인 윈도우로 교체.

        로드된 영상·세그먼트·PTZ 상태를 필드별로 되돌리는 대신 윈도우를
        새로 만든다 (초기화 누락 위험 없음). 진행 중 작업은 사라지므로
        확인을 받는다.
        """
        if QMessageBox.question(
                self, "새 프로젝트",
                "현재 상태를 닫고 새 프로젝트를 시작할까요?\n"
                "(저장하지 않은 변경·진행 중 작업은 사라집니다)") \
                != QMessageBox.StandardButton.Yes:
            return
        app = QApplication.instance()
        if self.ptz_tab is not None:
            app.removeEventFilter(self.ptz_tab)   # 구 윈도우 Space 필터 해제
        win = MainWindow(with_ptz=self._with_ptz)
        win._app_name = self._app_name
        win.setWindowTitle(self._app_name)
        win.setGeometry(self.geometry())
        win.show()
        app._pystitch_main = win              # 참조 유지 (GC 방지)
        self.close()

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "",
                                              "PyStitch 프로젝트 (*.json)")
        if path:
            self._open_recent(path)

    def _apply_project(self, d: dict):
        if d.get("lens_profile") in self.profiles:
            self.profile_combo.setCurrentText(d["lens_profile"])
        if d.get("left_files"):
            self._load_side("L", d["left_files"])
        if d.get("right_files"):
            self._load_side("R", d["right_files"])
        self.spin_offset.setValue(float(d.get("offset_sec", 0.0)))
        user = d.get("user", {})
        for k, sp in self.spin_user.items():
            sp.setValue(float(user.get(k, 0.0)))
        self.spin_feather.setValue(int(user.get("feather_px", 40)))
        self.spin_el_top.setValue(float(user.get("el_top_deg", 10.0)))
        self.spin_el_bottom.setValue(float(user.get("el_bottom_deg", -38.0)))
        exp = d.get("export", {})
        self.spin_start.setValue(float(exp.get("start", 0.0)))
        self.spin_end.setValue(float(exp.get("end", 60.0)))
        self.combo_codec.setCurrentIndex(int(exp.get("codec_index", 0)))
        self.spin_crf.setValue(int(exp.get("crf", 19)))
        self.combo_scale.setCurrentIndex(int(exp.get("scale_index", 0)))
        persp = exp.get("perspective", {})
        self.check_persp.setChecked(bool(persp.get("enabled", False)))
        self.spin_persp_k.setValue(float(persp.get("k", 0.3)))
        self.spin_persp_m.setValue(float(persp.get("m", 1.3)))
        pano = d.get("ptz", {}).get("pano")
        self._loaded_ptz_pano = pano
        if pano and self.ptz_tab is not None:
            self.ptz_tab.open_path(pano, quiet=True)
        self.segments = d.get("segments", [])
        self._refresh_segment_list()
        self._update_auto_labels()
        if self.segments:
            self.lbl_align.setText(f"(프로젝트에서 복원) 세그먼트 {len(self.segments)}개")
            if self.cur_imgs[0] is not None:
                self._render_preview()

    # ------------------------------------------------------------ 공통

    def closeEvent(self, ev):
        if self.ptz_tab is not None and self.ptz_tab.mc is not None:
            self.ptz_tab.mc.close()       # 멀티캠 alt 디코드 스레드 정리
        super().closeEvent(ev)

    def log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def _load_profiles(self):
        self.profiles = builtin_profiles()
        self.profile_combo.clear()
        self.profile_combo.addItems(list(self.profiles))
        if self.profiles:
            self.lens = LensProfile.load(next(iter(self.profiles.values())))

    # ------------------------------------------------------------ 탭 1: 동기화

    def _build_sync_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # 상단: 파일 열기 + 렌즈 프로파일
        top = QHBoxLayout()
        self.btn_open_l = QPushButton("좌측 영상 열기...")
        self.btn_open_r = QPushButton("우측 영상 열기...")
        self.btn_open_l.clicked.connect(lambda: self._open_video("L"))
        self.btn_open_r.clicked.connect(lambda: self._open_video("R"))
        self.lbl_files = QLabel("영상을 열어주세요 (첫 챕터 GOPR*.MP4 선택 시 챕터 자동 연결)")
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        top.addWidget(self.btn_open_l)
        top.addWidget(self.btn_open_r)
        top.addWidget(self.lbl_files, 1)
        top.addWidget(QLabel("렌즈:"))
        top.addWidget(self.profile_combo)
        v.addLayout(top)

        # 중앙: 듀얼 뷰어
        panes = QHBoxLayout()
        self.pane_l = FramePane("좌측 영상")
        self.pane_r = FramePane("우측 영상")
        panes.addWidget(self.pane_l)
        panes.addWidget(self.pane_r)
        v.addLayout(panes, 1)

        # 타임라인
        tl = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_time = QLabel("--:--.--")
        self.lbl_time.setMinimumWidth(110)
        tl.addWidget(self.slider, 1)
        tl.addWidget(self.lbl_time)
        v.addLayout(tl)
        self._slider_timer = QTimer(singleShot=True, interval=120)
        self._slider_timer.timeout.connect(self._show_current_frames)

        # 프레임 이동 + 오프셋
        ctl = QHBoxLayout()
        for text, d in [("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)]:
            b = QPushButton(text)
            b.setMaximumWidth(48)
            b.clicked.connect(lambda _, dd=d: self._step(dd))
            ctl.addWidget(b)
        ctl.addSpacing(24)
        ctl.addWidget(QLabel("동기화 오프셋 (초, R−L):"))
        self.spin_offset = QDoubleSpinBox(decimals=3, minimum=-30.0, maximum=30.0,
                                          singleStep=0.033)
        self.spin_offset.valueChanged.connect(lambda _: self._show_current_frames())
        ctl.addWidget(self.spin_offset)
        self.btn_autosync = QPushButton("오디오 자동 동기화")
        self.btn_autosync.clicked.connect(self._auto_sync)
        ctl.addWidget(self.btn_autosync)
        ctl.addStretch(1)
        v.addLayout(ctl)
        return w

    def _last_video_dir(self) -> str:
        """영상 열기 대화상자 시작 폴더: 현재 열린 영상 → 최근 사용 폴더."""
        if self.files_l:
            return str(Path(self.files_l[0]).parent)
        return str(QSettings("PyStitch360", "PyStitch360").value("last_video_dir", ""))

    def _remember_video_dir(self, d: str):
        QSettings("PyStitch360", "PyStitch360").setValue("last_video_dir", d)

    def _open_video(self, side: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"{'좌측' if side == 'L' else '우측'} 영상 (첫 챕터)",
            self._last_video_dir(), "GoPro 영상 (*.MP4 *.mp4)")
        if not path:
            return
        self._remember_video_dir(str(Path(path).parent))
        self._load_side(side, find_chapters(path))

    def _load_side(self, side: str, files):
        self._busy(True, flush=True)
        try:
            chapters = [Path(f) for f in files]
            vid = ChapteredVideo(chapters)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "열기 실패", str(e))
            return
        finally:
            self._busy(False)
        if side == "L":
            self.video_l, self.files_l = vid, chapters
        else:
            self.video_r, self.files_r = vid, chapters
        names = " + ".join(p.name for p in chapters)
        self.log(f"[open] {side}: {names} ({vid.total_frames}프레임, {vid.duration/60:.1f}분)")
        self._update_file_label()
        if self.video_l and self.video_r:
            self.slider.setEnabled(True)
            self.slider.setRange(0, self.video_l.total_frames - 1)
            self.slider2.setEnabled(True)
            self.slider2.setRange(0, self.video_l.total_frames - 1)
            self._show_current_frames()

    def _update_file_label(self):
        l = self.files_l[0].name if self.files_l else "—"
        r = self.files_r[0].name if self.files_r else "—"
        nl = f" (+{len(self.files_l)-1}챕터)" if len(self.files_l) > 1 else ""
        nr = f" (+{len(self.files_r)-1}챕터)" if len(self.files_r) > 1 else ""
        self.lbl_files.setText(f"L: {l}{nl}   R: {r}{nr}")

    def _on_profile_changed(self, name):
        if name in getattr(self, "profiles", {}):
            self.lens = LensProfile.load(self.profiles[name])
            self.log(f"[lens] {name}")

    def _on_slider(self, _):
        self._slider_timer.start()   # 디바운스: 드래그 중 과도한 4K 디코딩 방지
        if self.video_l:
            t = self.slider.value() / self.video_l.fps
            cs = round(t * 100)                      # 표시 단위(1/100초)로 먼저 반올림
            m, cs = divmod(cs, 6000)                 # "04:60.00" 방지
            self.lbl_time.setText(f"{m:02d}:{cs/100:05.2f}")
            if hasattr(self, "lbl_time2"):
                self.lbl_time2.setText(self.lbl_time.text())

    def _step(self, d: int):
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    def _show_current_frames(self):
        if not (self.video_l and self.video_r) or self._playing:
            return
        fl = self.slider.value()
        fr = int(round(fl + self.spin_offset.value() * self.video_l.fps))
        self._busy(True)
        try:
            ok_l, img_l = self.video_l.read_at(fl)
            ok_r, img_r = self.video_r.read_at(fr)
        finally:
            self._busy(False)
        self.cur_imgs = (img_l if ok_l else None, img_r if ok_r else None)
        self.pane_l.set_frame(self.cur_imgs[0])
        self.pane_r.set_frame(self.cur_imgs[1])
        # 정합 탭에서 프레임을 옮기면 파노라마 미리보기도 따라간다
        self._update_auto_labels()   # 시각에 따라 담당 세그먼트가 바뀔 수 있음
        if self.tabs.currentIndex() == 1 and self.segments:
            self._preview_debounced()

    def _auto_sync(self):
        if not (self.files_l and self.files_r):
            return
        start = self.slider.value() / self.video_l.fps if self.video_l else 0.0
        self.btn_autosync.setEnabled(False)
        self.log(f"[sync] 오디오 상관 분석 중 (t={start:.0f}s 부터 90초)...")
        self._sync_worker = SyncWorker(self.files_l[0], self.files_r[0], start)
        self._sync_worker.done.connect(self._sync_done)
        self._sync_worker.failed.connect(self._worker_failed)
        self._busy(True)
        self._sync_worker.start()

    def _sync_done(self, offset, conf):
        self._busy(False)
        self.btn_autosync.setEnabled(True)
        self.spin_offset.setValue(offset)
        fps = self.video_l.fps if self.video_l else 29.97
        self.log(f"[sync] 오프셋 {offset:+.3f}s ({offset*fps:+.1f}프레임), 신뢰도 {conf:.1f}"
                 + (" — 낮음, 수동 확인 필요" if conf < 4 else ""))

    def _worker_failed(self, msg):
        self._busy(False)
        self.btn_autosync.setEnabled(True)
        self.btn_align.setEnabled(True)
        self.log(f"[오류] {msg}")

    # ------------------------------------------------------------ 탭 2: 정합

    def _build_align_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_align = QPushButton("현재 프레임으로 자동 정합 (세그먼트 추가)")
        self.btn_align.clicked.connect(self._run_align)
        self.lbl_align = QLabel("정합 없음")
        top.addWidget(self.btn_align)
        top.addWidget(self.lbl_align, 1)
        v.addLayout(top)

        mid = QHBoxLayout()
        self.pano_pane = FramePane("파노라마 미리보기 (드래그: yaw/pitch, Shift+드래그: roll)",
                                   interactive=True)
        self.pano_pane.dragged.connect(self._pano_dragged)
        self.pano_pane.set_grid(True)
        mid.addWidget(self.pano_pane, 1)

        seg_box = QGroupBox("세그먼트 (방향 변동 구간)")
        seg_v = QVBoxLayout(seg_box)
        self.segment_list = QListWidget()
        self.segment_list.setMaximumWidth(280)
        seg_v.addWidget(self.segment_list, 1)
        seg_btns = QHBoxLayout()
        btn_goto = QPushButton("이동")
        btn_goto.clicked.connect(self._goto_segment)
        btn_del = QPushButton("삭제")
        btn_del.clicked.connect(self._delete_segment)
        seg_btns.addWidget(btn_goto)
        seg_btns.addWidget(btn_del)
        seg_v.addLayout(seg_btns)

        seg_v.addWidget(QLabel("자이로 이벤트 후보 (더블클릭=이동)"))
        self.event_list = QListWidget()
        self.event_list.setMaximumWidth(280)
        self.event_list.itemDoubleClicked.connect(self._goto_event)
        seg_v.addWidget(self.event_list, 1)
        self.btn_gpmf = QPushButton("자이로에서 충격 이벤트 탐지")
        self.btn_gpmf.clicked.connect(self._run_gpmf)
        seg_v.addWidget(self.btn_gpmf)
        mid.addWidget(seg_box)
        v.addLayout(mid, 1)

        # 타임라인 (1번 탭 슬라이더와 양방향 동기) + 재생
        tl = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_play.setMaximumWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)
        tl.addWidget(self.btn_play)
        self.check_grid = QCheckBox("그리드")
        self.check_grid.setChecked(True)
        self.check_grid.toggled.connect(self.pano_pane.set_grid)
        tl.addWidget(self.check_grid)
        for text, d in [("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)]:
            b = QPushButton(text)
            b.setMaximumWidth(48)
            b.clicked.connect(lambda _, dd=d: (self._stop_playback(), self._step(dd)))
            tl.addWidget(b)
        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setEnabled(False)
        self.slider2.sliderPressed.connect(self._stop_playback)
        self.slider2.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.slider2.setValue)
        self.slider.sliderPressed.connect(self._stop_playback)
        tl.addWidget(self.slider2, 1)
        self.lbl_time2 = QLabel("--:--.--")
        self.lbl_time2.setMinimumWidth(110)
        tl.addWidget(self.lbl_time2)
        v.addLayout(tl)

        adj = QGroupBox("미세조정 (자동 추정값에 더해짐, 도 단위)")
        g = QGridLayout(adj)
        self.spin_user = {}
        self.lbl_auto = {}
        def step_btn(sp, text, delta):
            b = QPushButton(text)
            b.setMaximumWidth(36)
            b.setAutoRepeat(True)          # 꾹 누르면 연속 조절
            b.clicked.connect(lambda _, s=sp, d=delta: s.setValue(s.value() + d))
            return b

        for col, (key, label) in enumerate([("pitch", "수평(pitch)"),
                                            ("roll", "기울기(roll)"),
                                            ("yaw", "센터링(yaw)")]):
            g.addWidget(QLabel(label), 0, col * 2)
            sp = QDoubleSpinBox(decimals=1, minimum=-45.0, maximum=45.0, singleStep=0.1)
            sp.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            sp.valueChanged.connect(lambda _: self._preview_debounced())
            self.spin_user[key] = sp
            row = QHBoxLayout()
            row.setSpacing(2)
            row.addWidget(step_btn(sp, "−1", -1.0))
            row.addWidget(step_btn(sp, "−.1", -0.1))
            row.addWidget(sp, 1)
            row.addWidget(step_btn(sp, "+.1", 0.1))
            row.addWidget(step_btn(sp, "+1", 1.0))
            g.addLayout(row, 0, col * 2 + 1)
            la = QLabel("자동 —")            # 현재 세그먼트의 자동 추정값 표시
            la.setStyleSheet("color: gray")
            self.lbl_auto[key] = la
            g.addWidget(la, 1, col * 2 + 1)
        g.addWidget(QLabel("심 페더(px)"), 0, 6)
        self.spin_feather = QSpinBox(minimum=2, maximum=400, value=40)
        self.spin_feather.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_feather, 0, 7)
        # 출력 세로 범위 (elevation, 도) — 아래를 내리면 발밑이 더 담긴다
        g.addWidget(QLabel("세로 위(도)"), 0, 8)
        self.spin_el_top = QDoubleSpinBox(decimals=0, minimum=0.0, maximum=45.0,
                                          singleStep=1.0, value=10.0)
        self.spin_el_top.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_el_top, 0, 9)
        g.addWidget(QLabel("세로 아래(도)"), 1, 8)
        self.spin_el_bottom = QDoubleSpinBox(decimals=0, minimum=-60.0, maximum=-5.0,
                                             singleStep=1.0, value=-38.0)
        self.spin_el_bottom.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_el_bottom, 1, 9)
        v.addWidget(adj)

        self._preview_timer = QTimer(singleShot=True, interval=400)
        self._preview_timer.timeout.connect(self._render_preview)
        return w

    def _run_align(self):
        self._stop_playback()
        img_l, img_r = self.cur_imgs
        if img_l is None or img_r is None:
            QMessageBox.information(self, "정합", "먼저 1번 탭에서 영상을 열고 프레임을 선택하세요.")
            return
        if self.lens is None:
            QMessageBox.warning(self, "정합", "렌즈 프로파일이 없습니다.")
            return
        self.btn_align.setEnabled(False)
        self.lbl_align.setText("정합 중...")
        # 재정합(세그먼트 추가) 시 수평/센터링은 기존 값 재사용 — 한 경기에서
        # 수평이 바뀌는 일은 거의 없고 재추정 노이즈만 끼어든다
        self._align_worker = AlignWorker(img_l.copy(), img_r.copy(), self.lens,
                                         reuse_level=self.current_alignment())
        self._align_worker.log.connect(self.log)
        self._align_worker.done.connect(self._align_done)
        self._align_worker.failed.connect(self._align_failed)
        self._busy(True)
        self._align_worker.start()

    def _align_done(self, alignment):
        self._busy(False)
        self.btn_align.setEnabled(True)
        a = alignment
        # 첫 정합은 0초부터 커버, 이후는 현재 프레임부터 적용 (충격 이벤트 재정합).
        # 정합을 추정한 프레임 시각은 align_sec 으로 따로 기억한다.
        start = 0.0 if not self.segments else self.current_time()
        align_sec = self.current_time()
        replaced = False
        for s in self.segments:
            if abs(s["start_sec"] - start) < 0.5:
                s["alignment"] = alignment   # 같은 지점 재정합이면 교체
                s["align_sec"] = align_sec
                replaced = True
                break
        if not replaced:
            self.segments.append({"start_sec": start, "alignment": alignment,
                                  "align_sec": align_sec})
            self.segments.sort(key=lambda s: s["start_sec"])
        self._refresh_segment_list()
        self.lbl_align.setText(
            f"인라이어 {a.n_inliers}/{a.n_matches}, 잔차 {a.residual_deg:.2f}°, "
            f"상대회전 {a.yaw_split_deg:.1f}°, 자동 pitch {a.pitch_auto*57.3:+.1f}° "
            f"roll {a.roll_auto*57.3:+.1f}°")
        self._update_auto_labels()
        self._render_preview()   # 정지 미리보기 갱신 (재생은 ▶ 버튼으로)

    def _align_failed(self, msg):
        self._busy(False)
        self.btn_align.setEnabled(True)
        self.lbl_align.setText("정합 실패")
        self.log(f"[오류] {msg}")

    def _refresh_segment_list(self):
        self.segment_list.clear()
        for s in self.segments:
            t = s["start_sec"]
            at = s.get("align_sec", t)
            item = f"{int(t//60):02d}:{t%60:04.1f}~"
            if abs(at - t) >= 0.5:
                item += f"  (정합점 {int(at//60):02d}:{at%60:04.1f})"
            a = s["alignment"]
            item += f"  인라이어 {a.n_inliers}"
            self.segment_list.addItem(item)

    def _goto_segment(self):
        row = self.segment_list.currentRow()
        if 0 <= row < len(self.segments) and self.video_l:
            s = self.segments[row]
            t = s.get("align_sec", s["start_sec"])   # 정합 추정 프레임으로 이동
            self.slider.setValue(int(t * self.video_l.fps))

    def _delete_segment(self):
        row = self.segment_list.currentRow()
        if 0 <= row < len(self.segments):
            del self.segments[row]
            self._refresh_segment_list()

    def _run_gpmf(self):
        if not (self.video_l and self.files_l):
            QMessageBox.information(self, "이벤트 탐지", "먼저 좌측 영상을 열어주세요.")
            return
        durations = [n / self.video_l.fps for n in self.video_l.chapter_frames]
        self.btn_gpmf.setEnabled(False)
        self.log("[gpmf] 자이로 데이터 분석 중...")
        self._gpmf_worker = GpmfWorker(self.files_l, durations)
        self._gpmf_worker.log.connect(self.log)
        self._gpmf_worker.done.connect(self._gpmf_done)
        self._gpmf_worker.failed.connect(self._gpmf_failed)
        self._busy(True)
        self._gpmf_worker.start()

    def _gpmf_done(self, events):
        self._busy(False)
        self.btn_gpmf.setEnabled(True)
        self._gyro_events = events
        self.event_list.clear()
        for e in events:
            kind = "방향변경" if e.persistent else "일시흔들림"
            self.event_list.addItem(
                f"{int(e.time_sec//60):02d}:{e.time_sec%60:04.1f}  "
                f"{e.net_angle_deg:.1f}° [{kind}]")
        self.log(f"[gpmf] 이벤트 {len(events)}개 — 방향변경 후보로 이동해서 "
                 f"몇 초 뒤 프레임에서 재정합하면 새 세그먼트가 됩니다")

    def _gpmf_failed(self, msg):
        self._busy(False)
        self.btn_gpmf.setEnabled(True)
        self.log(f"[gpmf] 실패: {msg}")

    def _goto_event(self):
        row = self.event_list.currentRow()
        if 0 <= row < len(getattr(self, "_gyro_events", [])) and self.video_l:
            # 이벤트 몇 초 뒤(안정화된 시점)로 이동
            t = self._gyro_events[row].time_sec + 3.0
            self.slider.setValue(int(t * self.video_l.fps))

    def _preview_debounced(self):
        if self.segments:
            self._preview_timer.start()

    # ------------------------------------------------------------ 재생
    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
            return
        if self.current_alignment() is None:
            QMessageBox.information(self, "재생", "먼저 정합을 실행하세요.")
            return
        self._start_playback()

    def _start_playback(self):
        alignment = self.current_alignment()
        if (self._playing or alignment is None
                or not (self.files_l and self.files_r)):
            return
        if self._playback_worker is not None and self._playback_worker.isRunning():
            return
        k, m = self._persp_params()
        w = PlaybackWorker(
            self.lens, alignment, self.files_l, self.files_r,
            self.spin_offset.value(), self.slider.value(),
            self.spin_user["pitch"].value(), self.spin_user["roll"].value(),
            self.spin_user["yaw"].value(), self.spin_feather.value(),
            persp_k=k, persp_m=m, el0=self._view_el()[0], el1=self._view_el()[1])
        w.frame_ready.connect(self._playback_frame)
        w.log.connect(self.log)
        w.failed.connect(lambda msg: self.log(f"[오류] 재생: {msg}"))
        w.finished.connect(self._playback_finished)
        self._playback_worker = w
        self._playing = True
        self._play_busy = True
        self._busy(True)
        self.btn_play.setText("⏸ 정지")
        w.start()

    def _stop_playback(self):
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._playback_worker.stop()   # _playing 해제는 finished 에서

    def _playback_frame(self, pano, f):
        if getattr(self, '_play_busy', False):
            self._play_busy = False
            self._busy(False)
        self.pano_pane.set_frame(pano)
        self.slider.setValue(f)            # _show_current_frames 는 재생 중 무시

    def _playback_finished(self):
        if getattr(self, '_play_busy', False):
            self._play_busy = False
            self._busy(False)
        self._playing = False
        self.btn_play.setText("▶ 재생")
        self._show_current_frames()

    def _pano_dragged(self, dx: float, dy: float, shift: bool):
        """미리보기 드래그로 yaw/pitch(Shift: roll) 조절 — 내용이 커서를 따라온다."""
        a = self.current_alignment()
        w_px = self.pano_pane.displayed_width()
        h_px = self.pano_pane.displayed_height()
        if a is None or w_px == 0 or h_px == 0:
            return
        yaw0, yaw1 = a.window(self.spin_user["yaw"].value())
        deg_x = (yaw1 - yaw0) * 57.29578 / w_px
        el0, el1 = self._view_el()
        deg_y = (el1 - el0) * 57.29578 / h_px
        if shift:
            sp = self.spin_user["roll"]
            sp.setValue(sp.value() + dx * deg_x * 0.3)   # roll 은 완만하게
        else:
            sy = self.spin_user["yaw"]
            sy.setValue(sy.value() - dx * deg_x)
            sp = self.spin_user["pitch"]
            sp.setValue(sp.value() + dy * deg_y)

    def _update_auto_labels(self):
        """미세조정 그룹에 현재 세그먼트의 자동 pitch/roll/yaw 표시."""
        a = self.current_alignment()
        for key in self.lbl_auto:
            if a is None:
                self.lbl_auto[key].setText("자동 —")
            else:
                deg = {"pitch": a.pitch_auto, "roll": a.roll_auto,
                       "yaw": a.yaw_auto}[key] * 57.29578
                self.lbl_auto[key].setText(f"자동 {deg:+.2f}°")

    def _view_el(self) -> tuple[float, float]:
        """출력 세로 범위 (el0=아래, el1=위, 라디안)."""
        return (self.spin_el_bottom.value() / 57.29578,
                self.spin_el_top.value() / 57.29578)

    def _persp_params(self) -> tuple[float, float]:
        """내보내기 탭의 원근비 조절 설정 (미체크 시 항등)."""
        if not self.check_persp.isChecked():
            return 0.0, 1.0
        return self.spin_persp_k.value(), self.spin_persp_m.value()

    def _render_preview(self):
        img_l, img_r = self.cur_imgs
        alignment = self.current_alignment()
        if alignment is None or img_l is None or img_r is None:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_timer.start()   # 이전 렌더 끝난 뒤 재시도
            return
        self._preview_worker = PreviewWorker(
            self.lens, alignment, img_l.copy(), img_r.copy(),
            self.spin_user["pitch"].value(), self.spin_user["roll"].value(),
            self.spin_user["yaw"].value(), self.spin_feather.value(),
            persp_k=self._persp_params()[0], persp_m=self._persp_params()[1],
            el0=self._view_el()[0], el1=self._view_el()[1])
        self._preview_worker.log.connect(self.log)
        self._preview_worker.done.connect(self.pano_pane.set_frame)
        self._preview_worker.failed.connect(lambda m: self.log(f"[오류] {m}"))
        self._preview_worker.start()

    # ------------------------------------------------------------ 탭 3: 내보내기

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGridLayout()
        g.addWidget(QLabel("시작 (초)"), 0, 0)
        self.spin_start = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1e6)
        g.addWidget(self.spin_start, 0, 1)
        g.addWidget(QLabel("끝 (초)"), 0, 2)
        self.spin_end = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1e6, value=60.0)
        g.addWidget(self.spin_end, 0, 3)
        g.addWidget(QLabel("코덱"), 1, 0)
        self.combo_codec = QComboBox()
        self.encoders = available_encoders()   # NVENC 등 실제 가용 인코더만
        self.combo_codec.addItems(list(self.encoders))
        g.addWidget(self.combo_codec, 1, 1)
        g.addWidget(QLabel("CRF"), 1, 2)
        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=19)
        g.addWidget(self.spin_crf, 1, 3)
        g.addWidget(QLabel("해상도"), 2, 0)
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100% (~5900px)", "50% (~2950px)"])
        g.addWidget(self.combo_scale, 2, 1)
        g.addWidget(QLabel("출력 형식"), 2, 2)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["파노라마 전체", "가상 PTZ (1080p, 공 추적)"])
        if not ptz_available():
            self.combo_mode.setItemData(1, "ultralytics 미설치 (pip install ultralytics)",
                                        Qt.ItemDataRole.ToolTipRole)
        g.addWidget(self.combo_mode, 2, 3)

        self.check_persp = QCheckBox("원근비 조절 (근경 축소 / 원경 확대)")
        self.check_persp.stateChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.check_persp, 3, 0, 1, 2)
        g.addWidget(QLabel("수직 k"), 3, 2)
        self.spin_persp_k = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=0.9,
                                           singleStep=0.05, value=0.3)
        self.spin_persp_k.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_persp_k, 3, 3)
        g.addWidget(QLabel("키스톤 m"), 3, 4)
        self.spin_persp_m = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=2.5,
                                           singleStep=0.05, value=1.3)
        self.spin_persp_m.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_persp_m, 3, 5)
        v.addLayout(g)

        h = QHBoxLayout()
        self.btn_export = QPushButton("내보내기 시작...")
        self.btn_export.clicked.connect(self._start_export)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_export)
        h.addWidget(self.btn_export)
        h.addWidget(self.btn_cancel)
        h.addStretch(1)
        v.addLayout(h)

        self.progress = QProgressBar()
        v.addWidget(self.progress)
        v.addStretch(1)
        return w

    def _start_export(self):
        self._stop_playback()
        if not self.segments:
            QMessageBox.information(self, "내보내기", "먼저 2번 탭에서 정합을 실행하세요.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "출력 파일", "panorama.mp4",
                                             "MP4 (*.mp4)")
        if not out:
            return
        ptz = self.combo_mode.currentIndex() == 1
        if ptz and not ptz_available():
            QMessageBox.warning(self, "내보내기",
                                "가상 PTZ 에는 ultralytics 가 필요합니다:\n"
                                "pip install ultralytics")
            return
        codec = self.encoders[self.combo_codec.currentText()]
        scale = 1.0 if self.combo_scale.currentIndex() == 0 else 0.5
        self._export_worker = ExportWorker(
            self.lens, self.segments,
            [str(p) for p in self.files_l], [str(p) for p in self.files_r],
            self.spin_offset.value(), self.spin_start.value(), self.spin_end.value(),
            out,
            self.spin_user["pitch"].value(), self.spin_user["roll"].value(),
            self.spin_user["yaw"].value(),
            codec=codec, crf=self.spin_crf.value(), scale=scale,
            feather_px=self.spin_feather.value(), ptz=ptz,
            persp_k=self._persp_params()[0], persp_m=self._persp_params()[1],
            el0=self._view_el()[0], el1=self._view_el()[1])
        self._export_worker.log.connect(self.log)
        self._export_worker.progress.connect(self._export_progress)
        self._export_worker.finished_ok.connect(self._export_done)
        self._export_worker.failed.connect(self._export_failed)
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)   # 준비 단계(렌더러 구성) 동안 불확정 표시
        self.progress.setFormat("준비 중...")
        self._export_worker.start()

    def _export_progress(self, done, total, fps):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        remain = (total - done) / fps if fps > 0 else 0
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain/60:.0f}분)")

    def _export_done(self, path):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("완료")
        self.log(f"[export] 저장: {path}")
        # 핸드오프 (P05): 통합 실행이면 PTZ 탭으로, PitchStitch 단독
        # 실행(탭 제거됨)이면 PitchWatch 프로세스로 잇는다.
        box = QMessageBox(self)
        box.setWindowTitle("내보내기")
        box.setText(f"완료: {path}")
        in_app = (self.ptz_tab is not None
                  and self.tabs.indexOf(self.ptz_tab) >= 0)
        b_open = box.addButton(
            "PTZ 탭에서 열기" if in_app else "PitchWatch 에서 열기",
            QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is b_open:
            if in_app:
                self.ptz_tab.open_path(str(path))
                self.tabs.setCurrentWidget(self.ptz_tab)
            else:
                import subprocess
                launcher = (Path(__file__).resolve().parents[2]
                            / "pitchwatch.py")
                subprocess.Popen([sys.executable, str(launcher), str(path)])
                self.log(f"[export] PitchWatch 실행: {Path(path).name}")

    def _export_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.log(f"[export] 실패: {msg}")

    def _cancel_export(self):
        if self._export_worker:
            self._export_worker.cancel()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
