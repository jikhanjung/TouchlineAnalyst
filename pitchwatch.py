"""PitchWatch — 경기 분석 앱 (P05 분리 런처).

PtzTab 을 자체 메인 윈도우로 승격: 파노라마 mp4 를 직접 연다 (상태는
전부 영상 사이드카 — 프로젝트 파일 불필요). 스티칭은 PitchStitch
(pitchstitch.py) 에서.

사용법: python pitchwatch.py [pano.mp4]
"""
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow

from pystitch.gui.ptz_tab import PtzTab

_MAX_RECENT = 10


class PitchWatchWindow(QMainWindow):
    """PtzTab + 파일(최근 파노라마)/분석 메뉴만 있는 얇은 껍데기."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PitchWatch")
        self.resize(1600, 980)
        self.ptz = PtzTab(self._log, self._last_dir, self._remember_dir)
        self.setCentralWidget(self.ptz)
        self._build_menu()

    # ------------------------------------------------------------ 로그/폴더
    def _log(self, msg):                  # PtzTab 자체 로그 탭이 주 화면
        print(msg, flush=True)

    def _last_dir(self) -> str:
        return str(QSettings("PyStitch360", "PyStitch360")
                   .value("last_video_dir", ""))

    def _remember_dir(self, d: str):
        QSettings("PyStitch360", "PyStitch360").setValue("last_video_dir", d)

    # ------------------------------------------------------------ 메뉴
    def _build_menu(self):
        m = self.menuBar().addMenu("파일(&F)")
        m.addAction("파노라마 열기...", self._open_pano)
        self._recent_menu = m.addMenu("최근 파노라마")
        m.addSeparator()
        m.addAction("경기 열기 (멀티캠)...", self._open_match)
        self._recent_match_menu = m.addMenu("최근 경기")
        m.addAction("멀티캠 경기 만들기...", self._build_match)
        self._half_menu = m.addMenu("하프")
        self._half_menu.setEnabled(False)
        m.addSeparator()
        m.addAction("종료", self.close)
        a = self.menuBar().addMenu("분석(&A)")
        a.addAction("갭필 2차 패스 (트랙 갭 재검출)...",
                    lambda: self.ptz.start_gapfill())
        a.addAction("킥오프 검출 (호각 × 대형)",
                    lambda: self.ptz.detect_events())
        a.addAction("트랙릿 병합 제안 (ReID 라이트)",
                    lambda: self.ptz.suggest_tracklet_merges())
        a.addAction("등번호 OCR (파노라마는 근측만)...",
                    lambda: self.ptz.run_jersey_ocr())
        a.addSeparator()
        a.addAction("하이라이트 후보 생성 (이벤트 융합)",
                    lambda: self.ptz.detect_highlights())
        a.addAction("하이라이트 일괄 내보내기...",
                    lambda: self.ptz.export_highlights())
        a.addAction("득점 역추론 (경기 중 킥오프 → 골 제안)",
                    lambda: self.ptz.suggest_goals())
        a.addAction("경기 정보 (시계 앵커·중단 구간)...",
                    lambda: self.ptz.edit_match_info())
        a.addAction("선수 히트맵/활동량 리포트",
                    lambda: self.ptz.generate_report())
        a.addAction("경기 지표 (점유율·패스)...",
                    lambda: self.ptz.show_match_stats())
        a.addSeparator()
        a.addAction("공/키프레임 편집 초기화",
                    lambda: self.ptz.reset_edits("ball"))
        a.addAction("선수 역할 지정 초기화",
                    lambda: self.ptz.reset_edits("roles"))
        a.addAction("경기장 캘리브레이션 초기화",
                    lambda: self.ptz.reset_edits("field"))
        a.addSeparator()
        a.addAction("모든 사용자 편집 초기화 (분석 원본으로)",
                    lambda: self.ptz.reset_edits("all"))
        self._rebuild_recent()
        self._rebuild_recent_matches()

    # ------------------------------------------------------------ 최근 파일
    def _recent(self) -> list[str]:
        v = QSettings("PyStitch360", "PyStitch360").value(
            "pitchwatch_recent", [])
        if isinstance(v, str):            # 원소 1개 리스트가 str 로 올 수 있음
            v = [v]
        return [p for p in (v or []) if Path(p).exists()]

    def _remember_recent(self, path):
        lst = [str(path)] + [p for p in self._recent() if p != str(path)]
        QSettings("PyStitch360", "PyStitch360").setValue(
            "pitchwatch_recent", lst[:_MAX_RECENT])
        self._rebuild_recent()

    def _rebuild_recent(self):
        self._recent_menu.clear()
        for p in self._recent():
            self._recent_menu.addAction(
                Path(p).name, lambda _=False, q=p: self.open_pano(q))
        self._recent_menu.setEnabled(bool(self._recent()))

    # ------------------------------------------------------------ 열기
    def _open_pano(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "완성 파노라마 영상", self._last_dir(),
            "영상 (*.mp4 *.MP4 *.mkv)")
        if path:
            self.open_pano(path)

    def open_pano(self, path):
        self.ptz.open_path(str(path))
        if self.ptz.pano_path is not None:
            self.setWindowTitle(f"PitchWatch — {Path(path).name}")
            self._remember_recent(path)
            self._half_menu.setEnabled(False)

    # ------------------------------------------------------ 멀티캠 (P07)
    def _open_match(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "멀티캠 경기", self._last_dir(), "경기 (*.match.json)")
        if path:
            self.open_match(path)

    def open_match(self, path, half=0):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from pystitch.core.match import load_match
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            doc = load_match(path)
        except (OSError, ValueError) as e:
            QApplication.restoreOverrideCursor()
            self.ptz.log(f"[match] 열기 실패: {e}")
            return
        finally:
            # open_match(ptz) 가 자체 커서를 관리 — 여기선 로드 구간만
            QApplication.restoreOverrideCursor()
        self._match_path = str(path)
        self.ptz.open_match(doc, half=half, path=str(path))
        if self.ptz.match is None:
            return
        title = doc.get("title") or Path(path).stem
        label = doc["halves"][self.ptz.match_half]["label"]
        self.setWindowTitle(f"PitchWatch — {title} [{label}]")
        self._remember_recent_match(path)
        self._half_menu.clear()
        self._half_menu.setEnabled(len(doc["halves"]) > 1)
        for i, h in enumerate(doc["halves"]):
            act = self._half_menu.addAction(
                h["label"], lambda _=False, j=i: self.open_match(path, j))
            act.setCheckable(True)
            act.setChecked(i == self.ptz.match_half)

    def _build_match(self):
        from PyQt6.QtWidgets import QFileDialog as FD

        from pystitch.core.match import MATCH_SUFFIX, save_match
        from pystitch.gui.multicam import MatchBuildDialog
        dlg = MatchBuildDialog(self, self.ptz.log, self._last_dir())
        if dlg.exec() != MatchBuildDialog.DialogCode.Accepted:
            return
        doc = dlg.doc()
        if doc is None:
            return
        first = Path(doc["halves"][0]["primary"])
        default = str(first.parent / (first.parent.name + MATCH_SUFFIX))
        path, _ = FD.getSaveFileName(self, "경기 저장", default,
                                     "경기 (*.match.json)")
        if not path:
            return
        if not path.endswith(MATCH_SUFFIX):
            path += MATCH_SUFFIX
        save_match(path, doc)
        self.ptz.log(f"[match] 저장: {path}")
        self.open_match(path)

    def _recent_matches(self) -> list[str]:
        v = QSettings("PyStitch360", "PyStitch360").value(
            "pitchwatch_recent_matches", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if Path(p).exists()]

    def _remember_recent_match(self, path):
        lst = [str(path)] + [p for p in self._recent_matches()
                             if p != str(path)]
        QSettings("PyStitch360", "PyStitch360").setValue(
            "pitchwatch_recent_matches", lst[:_MAX_RECENT])
        self._rebuild_recent_matches()

    def _rebuild_recent_matches(self):
        self._recent_match_menu.clear()
        for p in self._recent_matches():
            self._recent_match_menu.addAction(
                Path(p).name, lambda _=False, q=p: self.open_match(q))
        self._recent_match_menu.setEnabled(bool(self._recent_matches()))

    def closeEvent(self, ev):
        if self.ptz.mc is not None:
            self.ptz.mc.close()           # alt 디코드 스레드 정리
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    win = PitchWatchWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        arg = sys.argv[1]
        if arg.endswith(".match.json"):
            win.open_match(arg)           # 멀티캠 경기 직접 열기 (P07)
        else:
            win.open_pano(arg)            # PitchStitch 핸드오프 (P05-2)
    app._pitchwatch_main = win            # 참조 유지
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
