"""GoPro 챕터 분할 파일을 하나의 연속 타임라인으로 다루기.

GoPro 명명 규칙 (HERO5 기준):
  첫 챕터   GOPRxxxx.MP4
  이후 챕터 GP01xxxx.MP4, GP02xxxx.MP4, ...   (xxxx = 동일한 영상 번호)
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2

_FIRST = re.compile(r"^GOPR(\d{4})\.MP4$", re.IGNORECASE)
_CHAPTER = re.compile(r"^GP(\d{2})(\d{4})\.MP4$", re.IGNORECASE)


def find_chapters(first_file: str | Path) -> list[Path]:
    """첫 챕터 파일(GOPRxxxx.MP4)로부터 전체 챕터 체인을 찾는다."""
    first = Path(first_file)
    m = _FIRST.match(first.name)
    if not m:
        return [first]  # GoPro 명명 규칙이 아니면 단일 파일로 취급
    vid_no = m.group(1)
    chapters = [first]
    for p in sorted(first.parent.iterdir()):
        cm = _CHAPTER.match(p.name)
        if cm and cm.group(2) == vid_no:
            chapters.append(p)
    chapters[1:] = sorted(chapters[1:], key=lambda p: int(_CHAPTER.match(p.name).group(1)))
    return chapters


def group_directory(directory: str | Path) -> list[list[Path]]:
    """디렉터리의 GoPro 파일들을 영상 단위(챕터 체인)로 그룹핑."""
    directory = Path(directory)
    # Windows(대소문자 무시 FS)에선 *.MP4 와 *.mp4 가 같은 파일을 둘 다
    # 매칭해 그룹이 중복된다 (CI 실측) — 소문자 이름으로 중복 제거.
    seen: set[str] = set()
    files = []
    for p in sorted(directory.glob("*.MP4")) + sorted(directory.glob("*.mp4")):
        if p.name.lower() in seen:
            continue
        seen.add(p.name.lower())
        files.append(p)
    groups = []
    for p in files:
        if _FIRST.match(p.name):
            groups.append(find_chapters(p))
    return groups


class ChapteredVideo:
    """챕터 파일 목록을 하나의 연속 영상처럼 읽는 래퍼 (cv2.VideoCapture 기반)."""

    def __init__(self, files: list[str | Path]):
        self.files = [str(f) for f in files]
        if not self.files:
            raise ValueError("빈 파일 목록")
        self.fps = 0.0
        self.chapter_frames: list[int] = []
        for f in self.files:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                raise IOError(f"열 수 없음: {f}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if self.fps and abs(fps - self.fps) > 0.01:
                raise ValueError(f"챕터 간 fps 불일치: {fps} vs {self.fps}")
            self.fps = fps
            self.chapter_frames.append(n)
        self.cum_frames = [0]
        for n in self.chapter_frames:
            self.cum_frames.append(self.cum_frames[-1] + n)
        self.total_frames = self.cum_frames[-1]

        self._cap: cv2.VideoCapture | None = None
        self._chapter = -1
        self._pos = 0  # 다음에 read() 될 전역 프레임 번호

    @classmethod
    def from_first_file(cls, first_file: str | Path) -> "ChapteredVideo":
        return cls(find_chapters(first_file))

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0

    def _open_chapter(self, idx: int):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.files[idx])
        self._chapter = idx

    def _chapter_of(self, frame: int) -> int:
        for i in range(len(self.files)):
            if frame < self.cum_frames[i + 1]:
                return i
        return len(self.files) - 1

    def seek_frame(self, frame: int):
        frame = max(0, min(frame, self.total_frames - 1))
        idx = self._chapter_of(frame)
        if idx != self._chapter or self._cap is None:
            self._open_chapter(idx)
        local = frame - self.cum_frames[idx]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        self._pos = frame

    def seek_time(self, sec: float):
        self.seek_frame(int(round(sec * self.fps)))

    def read(self):
        """(ok, frame). 챕터 경계를 자동으로 넘어간다."""
        if self._cap is None:
            self.seek_frame(self._pos)
        ok, frame = self._cap.read()
        if not ok and self._chapter + 1 < len(self.files):
            self._open_chapter(self._chapter + 1)
            ok, frame = self._cap.read()
        if ok:
            self._pos += 1
        return ok, frame

    def grab(self) -> bool:
        """다음 프레임을 디코딩만 하고 반환하지 않음 (건너뛰기용, read()보다 쌈)."""
        if self._cap is None:
            self.seek_frame(self._pos)
        ok = self._cap.grab()
        if not ok and self._chapter + 1 < len(self.files):
            self._open_chapter(self._chapter + 1)
            ok = self._cap.grab()
        if ok:
            self._pos += 1
        return ok

    def read_at(self, frame: int):
        self.seek_frame(frame)
        return self.read()

    @property
    def position(self) -> int:
        return self._pos

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._chapter = -1
