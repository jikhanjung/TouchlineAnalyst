"""공용 위젯: 비디오/파노라마 프레임 표시."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class FramePane(QLabel):
    """BGR ndarray 를 종횡비 유지로 표시하는 라벨.

    interactive=True 면 드래그를 dragged(dx, dy, shift) 로 방출한다
    (dx/dy 는 표시된 픽스맵 좌표계 픽셀, shift 는 Shift 키 여부).
    좌표 비율(0~1)로 clicked / context_requested(우클릭) / hover 도 방출한다.

    줌/팬: 휠 = 커서 기준 줌 (1~12×), 가운데 버튼 드래그 = 팬
    (비인터랙티브 페인은 좌드래그로도 팬), 가운데 더블클릭 = 리셋.
    방출되는 비율 좌표는 항상 **전체 프레임 기준** — 호출부(편집 로직)는
    줌을 모른 채 그대로 동작한다.
    """

    dragged = pyqtSignal(float, float, bool)
    clicked = pyqtSignal(float, float)   # 표시 픽스맵 기준 좌표 비율 (0~1)
    context_requested = pyqtSignal(float, float, QPoint)   # fx, fy, 전역좌표
    hover = pyqtSignal(float, float)     # 버튼 안 눌린 이동 시 커서 위치 비율
    pressed = pyqtSignal(float, float)   # 좌버튼 프레스 위치 비율
    drag_moved = pyqtSignal(float, float)   # 드래그 중 절대 위치 비율(클램프)
    released = pyqtSignal(float, float)  # 좌버튼 릴리스 위치 비율(클램프)

    def __init__(self, placeholder="영상 없음", interactive=False):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 120)
        self.setStyleSheet("background-color: #202020; color: #808080;")
        self._pixmap: QPixmap | None = None
        self._interactive = interactive
        self._drag_pos = None
        self._pan_pos = None              # 가운데 버튼(또는 비인터랙티브 좌) 팬
        self._zoom = 1.0                  # 1 = 전체 맞춤
        self._vc = [0.5, 0.5]             # 보이는 영역 중심 (전체 프레임 비율)
        self._grid = False
        if interactive:
            # 평상시엔 일반 화살표 (사용자 방향) — 드래그/팬 중에만 손
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setMouseTracking(True)   # 버튼 없이도 hover 위치 추적

    # ------------------------------------------------------------ 줌/팬
    def _vis_rect(self):
        """보이는 영역 (x0, y0, w, h) — 전체 프레임 비율. 줌 1 = 전체."""
        w = 1.0 / self._zoom
        x0 = min(max(self._vc[0] - w / 2, 0.0), 1.0 - w)
        y0 = min(max(self._vc[1] - w / 2, 0.0), 1.0 - w)
        return x0, y0, w, w

    def reset_view(self):
        self._zoom = 1.0
        self._vc = [0.5, 0.5]
        self._update_scaled()

    def _disp_frac(self, pos):
        """위젯 좌표 → 표시 픽스맵 내 비율 (범위 검사 없음)."""
        p = self.pixmap()
        if p is None or p.isNull():
            return None
        x0 = (self.width() - p.width()) / 2
        y0 = (self.height() - p.height()) / 2
        return ((pos.x() - x0) / p.width(), (pos.y() - y0) / p.height())

    def _to_full(self, fx, fy):
        """표시 비율 → 전체 프레임 비율 (줌 반영)."""
        vx, vy, vw, vh = self._vis_rect()
        return vx + fx * vw, vy + fy * vh

    def _frac_at(self, pos):
        """위젯 좌표 → 전체 프레임 비율 (0~1). 표시 영역 밖이면 None."""
        d = self._disp_frac(pos)
        if d is None or not (0.0 <= d[0] <= 1.0 and 0.0 <= d[1] <= 1.0):
            return None
        return self._to_full(*d)

    def _frac_clamped(self, pos):
        """위젯 좌표 → 전체 프레임 비율, 표시 가장자리로 클램프 (드래그용)."""
        d = self._disp_frac(pos)
        if d is None:
            return None
        return self._to_full(min(max(d[0], 0.0), 1.0),
                             min(max(d[1], 0.0), 1.0))

    def wheelEvent(self, ev):
        if self._pixmap is None:
            return super().wheelEvent(ev)
        d = self._disp_frac(ev.position())
        step = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
        new_zoom = min(max(self._zoom * step, 1.0), 12.0)
        if new_zoom == self._zoom:
            return
        if d is not None:                 # 커서 아래 지점 고정 줌
            fx = min(max(d[0], 0.0), 1.0)
            fy = min(max(d[1], 0.0), 1.0)
            full = self._to_full(fx, fy)
            w2 = 1.0 / new_zoom
            self._vc = [full[0] - (fx - 0.5) * w2,
                        full[1] - (fy - 0.5) * w2]
        self._zoom = new_zoom
        self._update_scaled()

    def _pan_by(self, dx_px, dy_px):
        p = self.pixmap()
        if p is None or p.isNull() or self._zoom <= 1.0:
            return
        _vx, _vy, vw, vh = self._vis_rect()
        self._vc[0] -= dx_px / max(p.width(), 1) * vw
        self._vc[1] -= dy_px / max(p.height(), 1) * vh
        self._update_scaled()

    def set_grid(self, on: bool):
        """정렬 가이드라인 그리드 표시 (중앙 세로선 강조 — 하프라인 맞춤용)."""
        self._grid = bool(on)
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = self.pixmap()
        if not self._grid or p is None or p.isNull():
            return
        pw, ph = p.width(), p.height()
        x0 = (self.width() - pw) // 2
        y0 = (self.height() - ph) // 2
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        for i in range(1, 8):                       # 세로 8등분
            x = x0 + pw * i // 8
            if i != 4:
                painter.drawLine(x, y0, x, y0 + ph)
        for j in range(1, 4):                       # 가로 4등분
            y = y0 + ph * j // 4
            painter.drawLine(x0, y, x0 + pw, y)
        painter.setPen(QPen(QColor(255, 210, 0, 210), 3))   # 중앙 세로선 강조
        painter.drawLine(x0 + pw // 2, y0, x0 + pw // 2, y0 + ph)
        painter.end()

    def displayed_width(self) -> int:
        p = self.pixmap()
        return p.width() if p is not None and not p.isNull() else 0

    def displayed_height(self) -> int:
        p = self.pixmap()
        return p.height() if p is not None and not p.isNull() else 0

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.reset_view()
            return
        super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton \
                or (not self._interactive
                    and ev.button() == Qt.MouseButton.LeftButton
                    and self._zoom > 1.0):
            self._pan_pos = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._interactive and ev.button() == Qt.MouseButton.RightButton:
            fr = self._frac_at(ev.position())
            if fr is not None:
                self.context_requested.emit(
                    fr[0], fr[1], ev.globalPosition().toPoint())
            return
        if self._interactive and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.position()
            self._press_pos = ev.position()
            self._moved = 0.0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            fr = self._frac_at(ev.position())
            if fr is not None:
                self.pressed.emit(fr[0], fr[1])
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_pos is not None:
            d = ev.position() - self._pan_pos
            self._pan_pos = ev.position()
            self._pan_by(d.x(), d.y())
            return
        if self._drag_pos is not None:
            d = ev.position() - self._drag_pos
            self._drag_pos = ev.position()
            self._moved += abs(d.x()) + abs(d.y())
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.dragged.emit(d.x(), d.y(), shift)
            fr = self._frac_clamped(ev.position())
            if fr is not None:
                self.drag_moved.emit(fr[0], fr[1])
        elif self._interactive:              # 버튼 없이 이동 = hover
            fr = self._frac_at(ev.position())
            if fr is not None:
                self.hover.emit(fr[0], fr[1])
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan_pos is not None:
            self._pan_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._drag_pos is not None:
            fr = self._frac_clamped(ev.position())
            if fr is not None:
                self.released.emit(fr[0], fr[1])   # clicked 보다 먼저
            if self._moved < 4.0:        # 이동 없는 프레스+릴리스 = 클릭
                fr = self._frac_at(self._press_pos)
                if fr is not None:
                    self.clicked.emit(fr[0], fr[1])
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(ev)

    def set_frame(self, bgr: np.ndarray | None):
        if bgr is None:
            self._pixmap = None
            self.setText("영상 없음")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img.copy())
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap is None:
            return
        src = self._pixmap
        if self._zoom > 1.0:
            vx, vy, vw, vh = self._vis_rect()
            W, H = src.width(), src.height()
            src = src.copy(int(vx * W), int(vy * H),
                           max(1, int(vw * W)), max(1, int(vh * H)))
        self.setPixmap(src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_scaled()
