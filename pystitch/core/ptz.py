"""가상 PTZ: 파노라마에서 공/선수를 추적해 16:9 크롭 창을 부드럽게 이동.

PitchStitch/PitchAnalysis.py 의 가중평균 + 슬라이딩 윈도우 스무딩 아이디어를
재구현. YOLO(ultralytics) 는 선택 의존성 — 없으면 PTZ 기능만 비활성화된다.

  pip install ultralytics   (CPU torch 로 충분)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# COCO 클래스
_CLS_PERSON = 0
_CLS_BALL = 32  # sports ball

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "presets" / "yolov8n.pt"


def ptz_available() -> bool:
    # 설치 여부만 본다 — import ultralytics 는 torch 전체(수 GB)를 끌고
    # 들어와 GUI 생성 시점 호출(툴팁/버튼 활성)이 기동을 수 초 지연시킴
    from importlib.util import find_spec
    return find_spec("ultralytics") is not None


class Detector:
    """YOLO 래퍼. detect() → (N,4) [cx, cy, weight, is_ball]."""

    def __init__(self, weights: str | Path | None = None, imgsz: int = 960):
        from ultralytics import YOLO
        w = str(weights or _DEFAULT_WEIGHTS)
        self.model = YOLO(w)
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf,
                                 classes=[_CLS_PERSON, _CLS_BALL],
                                 verbose=False)[0]
        out = []
        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            is_ball = cls == _CLS_BALL
            # 공은 압도적 가중치 — 공이 보이면 공을 따라간다
            weight = c * (30.0 if is_ball else 1.0)
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, weight, float(is_ball)))
        return np.array(out) if out else np.zeros((0, 4))


class PTZSmoother:
    """감지 결과의 가중평균 목표점을 EMA + 속도 제한으로 부드럽게 추종."""

    def __init__(self, alpha: float = 0.08, max_speed_px: float = 25.0):
        self.alpha = alpha            # EMA 계수 (프레임당)
        self.max_speed = max_speed_px  # 프레임당 최대 이동 픽셀
        self.pos: np.ndarray | None = None

    def update(self, detections: np.ndarray, fallback: tuple[float, float]):
        """detections: (N,4) [cx, cy, weight, is_ball]. 반환: 부드러운 (x, y)."""
        if len(detections):
            w = detections[:, 2]
            target = (detections[:, :2] * w[:, None]).sum(axis=0) / w.sum()
        elif self.pos is not None:
            target = self.pos          # 감지 없음 → 제자리 유지
        else:
            target = np.array(fallback, dtype=np.float64)
        if self.pos is None:
            self.pos = np.array(target, dtype=np.float64)
        else:
            step = self.alpha * (np.asarray(target) - self.pos)
            n = np.linalg.norm(step)
            if n > self.max_speed:
                step *= self.max_speed / n
            self.pos = self.pos + step
        return float(self.pos[0]), float(self.pos[1])


class VirtualPTZ:
    """파노라마 프레임 → 공 추적 16:9 크롭 (기본 1920x1080).

    감지는 detect_every 프레임마다 축소본에서 수행하고,
    크롭 중심은 매 프레임 스무더로 갱신한다.
    """

    def __init__(self, pano_w: int, pano_h: int, out_w: int = 1920, out_h: int = 1080,
                 detect_every: int = 3, detect_width: int = 2944,
                 weights: str | Path | None = None):
        if pano_w < out_w or pano_h < out_h:
            raise ValueError(
                f"파노라마({pano_w}x{pano_h})가 크롭({out_w}x{out_h})보다 작음 — "
                "내보내기 해상도를 100% 로 하세요")
        self.pano_w, self.pano_h = pano_w, pano_h
        self.out_w, self.out_h = out_w, out_h
        self.detect_every = detect_every
        self.det_scale = detect_width / pano_w
        # imgsz 를 축소본 크기와 일치시켜 YOLO 내부 재축소로 인한
        # 실효 해상도 손실(공 소실의 주범)을 막는다
        self.detector = Detector(weights, imgsz=detect_width)
        self.smoother = PTZSmoother()
        self._i = 0
        self._last_det = np.zeros((0, 4))

    def process(self, pano: np.ndarray) -> np.ndarray:
        if self._i % self.detect_every == 0:
            small = cv2.resize(pano, (int(self.pano_w * self.det_scale),
                                      int(self.pano_h * self.det_scale)))
            det = self.detector.detect(small)
            if len(det):
                det[:, :2] /= self.det_scale
            self._last_det = det
        self._i += 1

        cx, cy = self.smoother.update(self._last_det,
                                      fallback=(self.pano_w / 2, self.pano_h * 0.45))
        x0 = int(round(cx - self.out_w / 2))
        y0 = int(round(cy - self.out_h / 2))
        x0 = max(0, min(x0, self.pano_w - self.out_w))
        y0 = max(0, min(y0, self.pano_h - self.out_h))
        return np.ascontiguousarray(
            pano[y0 : y0 + self.out_h, x0 : x0 + self.out_w])


# ================================================================ 2패스 PTZ
# 오프라인 내보내기용: 1패스에서 전체 검출 궤적을 모으고(전역 스무딩이
# 가능해짐), 2패스에서 크롭·인코딩한다. 실측(devlog 007): 검출 해상도를
# 2944px 로 올리면 경기장 안 공 검출률 0% → 52% (conf>=0.25).

def detect_raw(model, frame, det_w=2944, conf=0.2):
    """(N,6) [cx, cy, conf, is_ball, w, h] — 파노라마 원본 좌표계."""
    scale = det_w / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    r = model.predict(small, imgsz=det_w, conf=conf,
                      classes=[_CLS_PERSON, _CLS_BALL], verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(((x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale,
                    float(b.conf[0]), float(int(b.cls[0]) == _CLS_BALL),
                    (x2 - x1) / scale, (y2 - y1) / scale))
    return np.array(out) if out else np.zeros((0, 6))


def analyze_video(path, detect_every=3, det_w=None, field_top_frac=0.26,
                  weights=None, far_boost=True, far_band_frac=0.58,
                  cancel=None, progress=None, checkpoint_path=None,
                  checkpoint_every=1500, crop_hook=None, log=print):
    """1패스: 프레임 샘플마다 공/선수 검출. 반환 dict 는 JSON 직렬화 가능.

    field_top_frac 위(원경 트랙·관중석)의 공/선수는 장외로 버린다.
    far_boost: 원경 밴드(field_top~far_band_frac)를 원본 해상도 정사각 타일로
    잘라 공 전용 추가 검출 — 전체 프레임은 ~50% 축소라 원경 공(~10px)이
    뭉개지는 문제를 보완한다. 트래커 상태 보호를 위해 별도 모델 인스턴스 사용.

    checkpoint_path 지정 시 checkpoint_every 샘플마다 부분 결과를 저장하고,
    같은 조건(영상·파라미터·모델)의 체크포인트가 있으면 그 프레임부터 재개.
    취소 시에도 그 지점까지 저장한다. (재개 시 ByteTrack 선수 ID 는 경계에서
    새로 시작 — 팀 분류는 트랙릿 단위라 영향 미미.)

    det_w 기본값(None)은 파노라마 폭의 1/2 (32의 배수, 4416 상한) — 8K 급
    소스에서도 검출 해상도가 소스에 비례해 따라간다 (5906px → 2944 동일).
    """
    import json as _json
    from ultralytics import YOLO
    model = YOLO(str(weights or _DEFAULT_WEIGHTS))
    model_far = YOLO(str(weights or _DEFAULT_WEIGHTS)) if far_boost else None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"열 수 없음: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if det_w is None:
        det_w = min(round(pano_w / 2 / 32) * 32, 4416)
    field_top = field_top_frac * pano_h
    frames_idx, balls, players = [], [], []
    ball_cands = []
    import time
    t0 = time.perf_counter()
    i = 0
    ckpt_key = {"video": str(path), "detect_every": detect_every, "det_w": det_w,
                "weights": str(weights or _DEFAULT_WEIGHTS),
                "far_boost": bool(far_boost)}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ck = _json.loads(Path(checkpoint_path).read_text())
            if all(ck.get(k) == v for k, v in ckpt_key.items()):
                frames_idx = ck["frames"]
                balls = ck["balls"]
                ball_cands = ck["ball_cands"]
                players = ck["players"]
                i = int(ck["next_frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                log(f"[analyze] 체크포인트 재개: {i}/{total} 프레임부터")
        except Exception as e:  # noqa: BLE001
            log(f"[analyze] 체크포인트 무시: {e}")
    resume_base = i

    def _save_ckpt():
        if checkpoint_path is None:
            return
        tmp = str(checkpoint_path) + ".tmp"
        Path(tmp).write_text(_json.dumps(
            ckpt_key | {"next_frame": i, "frames": frames_idx, "balls": balls,
                        "ball_cands": ball_cands, "players": players}))
        Path(tmp).replace(checkpoint_path)
    while True:
        if cancel is not None and cancel():
            cap.release()
            _save_ckpt()
            log(f"[analyze] 사용자 취소 — {i}프레임까지 체크포인트 저장"
                if checkpoint_path else "[analyze] 사용자 취소")
            return None
        ok = cap.grab()
        if not ok:
            break
        if i % detect_every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            scale = det_w / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            r = model.track(small, imgsz=det_w, conf=0.2,
                            classes=[_CLS_PERSON, _CLS_BALL],
                            persist=True, verbose=False)[0]
            far_balls = []
            if model_far is not None:
                # 정사각형 타일: 학습 크기(~640) 분포에 맞고 네이티브 해상도 유지
                y0b, y1b = int(field_top), int(pano_h * far_band_frac)
                strip = frame[y0b:y1b]
                th = strip.shape[0]
                step = max(1, int(th * 0.85))          # 15% 겹침
                offs = list(range(0, max(strip.shape[1] - th, 1), step))
                offs.append(strip.shape[1] - th)       # 우측 끝 보장
                tiles = [strip[:, x:x + th] for x in offs]
                imgsz_t = (th + 31) // 32 * 32
                results = model_far.predict(tiles, imgsz=imgsz_t, conf=0.15,
                                            classes=[_CLS_BALL], verbose=False)
                for x_off, r2 in zip(offs, results):
                    for b2 in r2.boxes:
                        bx1, by1, bx2, by2 = b2.xyxy[0].tolist()
                        far_balls.append([
                            round((bx1 + bx2) / 2 + x_off, 1),
                            round((by1 + by2) / 2 + y0b, 1),
                            round(float(b2.conf[0]), 3),
                            round(bx2 - bx1, 1), round(by2 - by1, 1)])
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            bcands = []                            # 이 샘플의 공 후보들
            prow = []
            for b_ in r.boxes:
                x1, y1, x2, y2 = b_.xyxy[0].tolist()
                cxs, cys = (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale
                conf = float(b_.conf[0])
                if cys < field_top:
                    continue                       # 장외 (원경 트랙·관중석)
                if int(b_.cls[0]) == _CLS_BALL:
                    bcands.append([round(cxs, 1), round(cys, 1), round(conf, 3),
                                   round((x2 - x1) / scale, 1),
                                   round((y2 - y1) / scale, 1)])
                    continue
                tid = int(b_.id[0]) if b_.id is not None else -1
                # 유니폼 색: 박스 상반신(위 절반, 가로 중앙 60%) HSV 평균
                tx1 = int(x1 + (x2 - x1) * 0.2)
                tx2 = int(x2 - (x2 - x1) * 0.2)
                torso = hsv[int(y1):int((y1 + y2) / 2), max(tx1, 0):max(tx2, 1)]
                hm, sm, vm = (torso.reshape(-1, 3).mean(axis=0).tolist()
                              if torso.size else (0.0, 0.0, 0.0))
                prow.append([round(cxs, 1), round(cys, 1),
                             round((x2 - x1) / scale, 1), round((y2 - y1) / scale, 1),
                             tid, round(hm, 1), round(sm, 1), round(vm, 1)])
            # 원경 네이티브 검출 병합 → 근접 중복 제거 후 상위 3개 저장.
            # 후보 다수 보존: 미끼가 conf 를 이겨도 진짜 공이 살아남아
            # 트랙 연결 단계에서 별도 트랙으로 경쟁할 수 있다.
            bcands += far_balls
            bcands.sort(key=lambda b: -b[2])
            kept = []
            for b in bcands:
                if all(np.hypot(b[0] - k[0], b[1] - k[1]) > 30 for k in kept):
                    kept.append(b)
                if len(kept) == 3:
                    break
            frames_idx.append(i)
            balls.append(kept[0] if kept else None)
            ball_cands.append(kept)
            players.append(prow)
            if crop_hook is not None:     # P09: 디코드 1회 다중 소비 —
                crop_hook(frame, len(frames_idx) - 1, i, prow)   # OCR/포즈 크롭
            done_new = i - resume_base
            if len(frames_idx) % 30 == 0 and progress is not None:
                el = time.perf_counter() - t0
                progress(i, total, done_new / max(el, 1e-9))
            if len(frames_idx) % 300 == 0:
                el = time.perf_counter() - t0
                log(f"[analyze] {i}/{total} ({done_new/max(el,1e-9):.1f}fps)")
            if checkpoint_path is not None and len(frames_idx) % checkpoint_every == 0:
                _save_ckpt()
        i += 1
    cap.release()
    if checkpoint_path is not None:
        Path(checkpoint_path).unlink(missing_ok=True)   # 완료 — 체크포인트 정리
        Path(str(checkpoint_path) + ".tmp").unlink(missing_ok=True)
    return {"video": str(path), "total_frames": i, "fps": fps,
            "players_fmt": "cxcywh_id_hsv",   # 구캐시(2/4열)와 소비부 호환
            "far_boost": bool(far_boost),
            "pano_w": pano_w, "pano_h": pano_h,
            "detect_every": detect_every, "det_w": det_w,
            "field_top_frac": field_top_frac,
            "frames": frames_idx, "balls": balls, "ball_cands": ball_cands,
            "players": players}


# 역할 번호: classify_teams 반환값과 GUI 색상 테이블이 공유하는 규약.
ROLE_TEAM0, ROLE_TEAM1, ROLE_OTHER = 0, 1, 2
ROLE_GK0, ROLE_GK1, ROLE_REF = 3, 4, 5


def gapfill_targets(analysis, ignore_ranges=None, force_ranges=None,
                    linked=None, max_gap_s=4.0):
    """수락 트랙 사이 갭의 보간 목표 [(si, x, y), ...] (갭필 2차 패스용).

    수락 커버리지(사용자 편집 반영)에서 이웃한 두 수락 샘플 사이가
    max_gap_s 이하인 갭만 — 그보다 길면 공이 화면 밖/플레이 중단일
    가능성이 높아 보간 위치를 신뢰할 수 없다.
    """
    if linked is None:
        linked = link_ball_tracks(analysis)
    _, acc, _ = accept_ball_tracks(
        analysis, ignore_ranges=ignore_ranges or [],
        force_ranges=force_ranges or [], linked=linked, log=lambda s: None)
    fps = analysis["fps"]
    de = analysis["detect_every"]
    fin = np.where(np.isfinite(acc[:, 0]))[0]
    max_gap = max(2, int(round(max_gap_s * fps / de)))
    out = []
    for a, b in zip(fin[:-1], fin[1:]):
        if b - a <= 1 or b - a > max_gap:
            continue
        for si in range(int(a) + 1, int(b)):
            w = (si - a) / (b - a)
            out.append((int(si),
                        float(acc[a, 0] * (1 - w) + acc[b, 0] * w),
                        float(acc[a, 1] * (1 - w) + acc[b, 1] * w)))
    return out


def gapfill_analysis(pano_path, analysis, targets, weights=None,
                     conf=0.06, ball_radius=250.0, tile=640,
                     progress=None, cancel=None, log=print):
    """갭 보간 위치의 저문턱 640 타일 검출 → 분석에 공/선수 주입.

    A/B 실험(devlog 020) 근거: 운영 패스가 놓친 지점의 ~49% 가 같은
    파노라마에서 저문턱 중심 타일로 잡힌다. 같은 추론으로 사람도 함께
    검출해 원경 선수 커버리지를 보강한다 (id -1, 상반신 HSV 포함).

    공 주입 행은 저장 conf 를 링크 문턱(0.25)을 넘는 0.26 으로 바닥
    처리하고 원래 conf 를 6번째 원소로 보존 — 길이 6 = 갭필 행.
    반환: (주입 공 수, 주입 선수 수).
    """
    import time

    from ultralytics import YOLO
    model = YOLO(str(weights or _DEFAULT_WEIGHTS))
    cap = cv2.VideoCapture(str(pano_path))
    W, H = analysis["pano_w"], analysis["pano_h"]
    frames = analysis["frames"]
    field_top = analysis.get("field_top_frac", 0.26) * H
    if not analysis.get("ball_cands"):
        cap.release()
        raise ValueError("구형 분석(ball_cands 없음) — 재분석 필요")
    n_ball = n_person = 0
    t0 = time.perf_counter()
    pos = -10 ** 9                       # 마지막으로 읽은 프레임 위치
    for k, (si, x, y) in enumerate(sorted(targets)):
        if cancel is not None and cancel():
            break
        F = int(frames[si])
        # 갭 목표는 연속 샘플이 대부분 — 가까우면 시크 대신 순차 grab
        # (7.7GB 파일 랜덤 시크는 ~1.4s, grab 은 프레임당 ~10ms)
        if 0 <= F - pos <= 90:
            for _ in range(F - pos - 1):
                cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, F)
        ok, frame = cap.read()
        pos = F
        if not ok:
            continue
        x0 = int(np.clip(x - tile / 2, 0, max(W - tile, 0)))
        y0 = int(np.clip(y - tile / 2, 0, max(H - tile, 0)))
        crop = frame[y0:y0 + tile, x0:x0 + tile]
        r = model.predict(crop, imgsz=tile, conf=conf,
                          classes=[_CLS_PERSON, _CLS_BALL], verbose=False)[0]
        hsv = None
        best = None
        prow = analysis["players"][si]
        cands = analysis["ball_cands"][si]
        for b in r.boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            cx, cy = x0 + (bx1 + bx2) / 2, y0 + (by1 + by2) / 2
            c = float(b.conf[0])
            if cy < field_top:
                continue                     # 장외 (원경 트랙·관중석)
            if int(b.cls[0]) == _CLS_BALL:
                if (cx - x) ** 2 + (cy - y) ** 2 <= ball_radius ** 2 \
                        and (best is None or c > best[2]):
                    best = [cx, cy, c, bx2 - bx1, by2 - by1]
            else:
                if any((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                       < max(p[3] / 2, 20.0) ** 2
                       for p in prow if len(p) >= 4):
                    continue                 # 기존 선수와 중복
                if hsv is None:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                tx1 = int(bx1 + (bx2 - bx1) * 0.2)
                tx2 = int(bx2 - (bx2 - bx1) * 0.2)
                torso = hsv[int(by1):int((by1 + by2) / 2),
                            max(tx1, 0):max(tx2, 1)]
                hm, sm, vm = (torso.reshape(-1, 3).mean(axis=0).tolist()
                              if torso.size else (0.0, 0.0, 0.0))
                prow.append([round(cx, 1), round(cy, 1),
                             round(bx2 - bx1, 1), round(by2 - by1, 1), -1,
                             round(hm, 1), round(sm, 1), round(vm, 1)])
                n_person += 1
        if best is not None and all(
                np.hypot(best[0] - p[0], best[1] - p[1]) > 30
                for p in cands):
            cands.append([round(best[0], 1), round(best[1], 1),
                          max(0.26, round(best[2], 3)),
                          round(best[3], 1), round(best[4], 1),
                          round(best[2], 3)])
            if analysis["balls"][si] is None:
                analysis["balls"][si] = list(cands[-1][:5])
            n_ball += 1
        if progress is not None and (k + 1) % 30 == 0:
            progress(k + 1, len(targets),
                     (k + 1) / max(time.perf_counter() - t0, 1e-9))
    cap.release()
    analysis["gapfill"] = {"targets": len(targets), "ball": n_ball,
                           "person": n_person, "conf": conf,
                           "radius": ball_radius}
    log(f"[gapfill] 목표 {len(targets)}개 → 공 {n_ball}개, "
        f"선수 {n_person}명 주입")
    return n_ball, n_person


def propagate_seed(pano_path, analysis, f0, x0, y0, kind="ball",
                   weights=None, max_extend_s=4.0, conf=0.06,
                   radius=150.0, tile=640, miss_limit=3,
                   cancel=None, log=print):
    """수동 인식(공/선수) 시드를 앞뒤 샘플로 따라가며 확장.

    시드 (f0, x0, y0) 의 최근접 샘플에서 출발해 양방향으로 저문턱 타일
    검출을 체인으로 잇는다 — 크롭 중심은 직전 매칭 위치, 매칭은 반경
    radius 내 최고 conf. 연속 miss_limit 회 놓치면 그 방향 중단.
    순방향은 순차 grab(빠름), 역방향은 프레임 시크(느림 — 조기 중단이
    비용을 줄임). 반환: [(si, x, y, w, h, conf), ...] (시드 샘플 포함).
    """
    from ultralytics import YOLO
    model = YOLO(str(weights or _DEFAULT_WEIGHTS))
    cls = _CLS_BALL if kind == "ball" else _CLS_PERSON
    cap = cv2.VideoCapture(str(pano_path))
    W = analysis["pano_w"]
    H = analysis["pano_h"]
    frames = np.asarray(analysis["frames"])
    fps = analysis["fps"]
    de = analysis["detect_every"]
    si0 = int(np.argmin(np.abs(frames - f0)))
    n_ext = max(1, int(round(max_extend_s * fps / de)))

    def detect_at(frame, cx, cy):
        x0_ = int(np.clip(cx - tile / 2, 0, max(W - tile, 0)))
        y0_ = int(np.clip(cy - tile / 2, 0, max(H - tile, 0)))
        crop = frame[y0_:y0_ + tile, x0_:x0_ + tile]
        r = model.predict(crop, imgsz=tile, conf=conf, classes=[cls],
                          verbose=False)[0]
        best = None
        for b in r.boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            bx, by = x0_ + (bx1 + bx2) / 2, y0_ + (by1 + by2) / 2
            c = float(b.conf[0])
            if (bx - cx) ** 2 + (by - cy) ** 2 <= radius ** 2 \
                    and (best is None or c > best[4]):
                best = (bx, by, bx2 - bx1, by2 - by1, c)
        return best

    out = []
    pos_f = -10 ** 9
    for direction in (1, -1):
        cx, cy = float(x0), float(y0)
        misses = 0
        rng = range(si0, min(si0 + n_ext + 1, len(frames))) \
            if direction == 1 else range(si0 - 1, max(si0 - n_ext - 1, -1), -1)
        for si in rng:
            if cancel is not None and cancel():
                break
            F = int(frames[si])
            if direction == 1 and 0 <= F - pos_f <= 90:
                for _ in range(F - pos_f - 1):
                    cap.grab()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, F)
            ok, frame = cap.read()
            pos_f = F
            if not ok:
                break
            hit = detect_at(frame, cx, cy)
            if hit is None:
                misses += 1
                if misses >= miss_limit:
                    break
                continue
            misses = 0
            cx, cy = hit[0], hit[1]
            if direction == -1 or si != si0 or not out:
                out.append((int(si), round(hit[0], 1), round(hit[1], 1),
                            round(hit[2], 1), round(hit[3], 1),
                            round(hit[4], 3)))
    cap.release()
    out.sort()
    log(f"[seed] {kind} 시드 {f0/fps:.1f}s → {len(out)}샘플 연결 "
        f"(±{max_extend_s:.0f}s 창)")
    return out


def team_features(analysis):
    """트랙릿별 색 특징·스팬 추출 — classify_teams 의 비싼 전처리.

    전체 검출을 한 번 훑는 유일한 단계라 호출부(GUI)가 분석당 1회
    캐시해 재사용한다 (역할 지정마다 재추출하면 수 초 프리즈).
    반환: {ids, X(중앙값 특징), n_det, spans}.
    """
    feats: dict[int, list] = {}
    spans: dict[int, list] = {}          # {tid: [첫, 끝 프레임]} — GK 단일성
    frames = np.asarray(analysis["frames"])
    for si, prow in enumerate(analysis["players"]):
        for p in prow:
            if len(p) >= 8 and p[4] >= 0:
                h, s_, v = p[5], p[6], p[7]
                a = h / 90.0 * np.pi          # OpenCV H 는 0~180
                feats.setdefault(int(p[4]), []).append(
                    (s_ * np.cos(a), s_ * np.sin(a), v))
                f = int(frames[si]) if si < len(frames) else si
                e = spans.setdefault(int(p[4]), [f, f])
                e[1] = f
    ids = sorted(feats)
    return {"ids": ids,
            "X": (np.array([np.median(feats[t], axis=0) for t in ids])
                  if ids else np.zeros((0, 3))),
            "n_det": np.array([len(feats[t]) for t in ids], float),
            "spans": spans}


def classify_teams(analysis, k=3, roles=None, feats=None):
    """트랙릿(선수 ID)별 유니폼 색으로 팀/역할 분류.

    분석의 선수 행이 [cx,cy,w,h,id,h,s,v] 형식일 때만 동작. 특징은
    채도로 가중한 색상 벡터 (s·cos h, s·sin h, v) 의 트랙릿 중앙값.
    farthest-point 초기화 k-means (k=3) → 검출 수 기준 상위 2개 군집이
    팀 0/1, 나머지는 2(심판·GK 등).

    roles({track_id: 역할번호}, ROLE_*)가 주어지면 사용자 시드로 역할을
    전파한다: 시드 트랙릿들의 색 중앙값을 역할 센터로 삼고, 자기 기본
    군집 중심보다 그 센터가 더 가까운 트랙릿에 역할을 부여한다 (GK·심판은
    필드 플레이어와 다른 색 옷이므로 시드 한 명이면 ID가 갈라져도 같은
    사람/역할의 다른 트랙릿까지 잡힌다). 시드 자신은 무조건 그 역할.

    색 전파는 GK·심판(역할 ≥3)만 — 팀 역할(0/1) 시드는 자신에게만
    적용된다. 팀 수정 시드는 대개 오분류된 애매한 색(그늘 등)이라 그
    색을 팀 센터로 삼으면 멀쩡한 다른 트랙릿(어두운 옷 심판 포함)까지
    끌려오는 오동작이 있었다.
    반환: {track_id: 역할번호}.
    """
    tf = feats if feats is not None else team_features(analysis)
    ids, X, n_det, spans = tf["ids"], tf["X"], tf["n_det"], tf["spans"]
    if not ids:
        return {}
    k = min(k, len(ids))
    # farthest-point 초기화 (결정적)
    centers = [X[int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]]
    while len(centers) < k:
        d = np.min([np.linalg.norm(X - c, axis=1) for c in centers], axis=0)
        centers.append(X[int(np.argmax(d))])
    C = np.array(centers)
    for _ in range(20):
        lab = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    # 군집 크기(검출 수 합) 순으로 팀 번호 재배열: 0,1 = 양 팀, 2 = 기타
    sizes = [n_det[lab == j].sum() for j in range(k)]
    order = np.argsort(sizes)[::-1]
    remap = {int(order[r]): min(r, 2) for r in range(k)}
    base = {t: remap[int(l)] for t, l in zip(ids, lab)}
    if not roles:
        return base
    pos = {t: i for i, t in enumerate(ids)}
    seed_feats: dict[int, list] = {}
    for t, r in roles.items():
        if int(t) in pos and int(r) >= 3:     # GK·심판만 색 전파
            seed_feats.setdefault(int(r), []).append(X[pos[int(t)]])
    role_C = {r: np.median(np.array(v), axis=0)
              for r, v in seed_feats.items()}
    base_C = {remap[j]: X[lab == j].mean(0)
              for j in range(k) if np.any(lab == j)}
    team_C = [base_C[j] for j in (0, 1) if j in base_C]
    out = {}
    for t in ids:
        x = X[pos[t]]
        d_team = (min(np.linalg.norm(x - c) for c in team_C)
                  if team_C else np.inf)
        best_r, best_d = None, np.inf
        for r, c in role_C.items():
            d = np.linalg.norm(x - c)
            if d < best_d:
                best_r, best_d = r, d
        # 시드 역할 색이 어느 팀 색보다 가까우면 그 역할 — GK/심판은 팀과
        # 다른 색 옷이라는 전제. 아니면 기본 군집 결과 유지.
        out[t] = int(best_r) if best_d < d_team else base[t]
    for t, r in roles.items():              # 시드는 색과 무관하게 확정
        if int(t) in pos:
            out[int(t)] = int(r)
    # GK 단일성 (휴리스틱): 팀당 GK 는 한 명 — 같은 GK 역할(3/4)이
    # 시간상 겹치는 여러 트랙릿에 붙을 수 없다. 시드 우선, 다음 역할
    # 색 근접 순으로 남기고 겹치는 전파분은 기본 군집으로 되돌린다.
    # (시간 비겹침 조각 전파는 유지 — ID 갈라짐 커버가 전파 목적.)
    # 심판(5)은 주심+선심 3명이라 제외. docs/heuristics.md 참고.
    tol = 0.5 * float(analysis.get("fps", 30.0))
    for gk in (3, 4):
        if gk not in role_C:
            continue
        seeds_r = {int(t) for t, r in roles.items() if int(r) == gk}
        cands = sorted(
            (t for t in ids if out.get(t) == gk),
            key=lambda t: (t not in seeds_r,
                           float(np.linalg.norm(X[pos[t]] - role_C[gk]))))
        kept = []
        for t in cands:
            s0, s1 = spans.get(t, (0, 0))
            clash = any(min(s1, spans[k][1]) - max(s0, spans[k][0]) > tol
                        for k in kept)
            if clash and t not in seeds_r:
                out[t] = base[t]
            else:
                kept.append(t)
    return out


def tracklet_colors(analysis):
    """트랙릿별 유니폼 대표색 {track_id: (h, s, v)} (OpenCV 스케일).

    H 는 원형(빨강이 0/180 경계에 걸침)이라 채널별 중앙값 대신
    classify_teams 와 같은 s-가중 벡터의 중앙값을 취해 되돌린다.
    """
    feats: dict[int, list] = {}
    for prow in analysis["players"]:
        for p in prow:
            if len(p) >= 8 and p[4] >= 0:
                a = p[5] / 90.0 * np.pi
                feats.setdefault(int(p[4]), []).append(
                    (p[6] * np.cos(a), p[6] * np.sin(a), p[7]))
    out = {}
    for t, v in feats.items():
        cx, cy, vv = np.median(np.array(v), axis=0)
        h = (np.arctan2(cy, cx) % (2 * np.pi)) * 90.0 / np.pi
        out[t] = (float(h), float(min(np.hypot(cx, cy), 255.0)), float(vv))
    return out


def ground_positions(players_row, pano_w, pano_h, cam_height=4.0,
                     el_top_deg=10.0, el_bottom_deg=-38.0):
    """선수 발 위치(박스 하단 중앙)를 카메라 기준 지면 좌표(m)로 투영.

    원통 파노라마: 열 ↔ yaw 선형, 행 ↔ tan(elevation) 선형. 초점거리는
    f = H / (tan el_top − tan el_bottom) 으로 역산되므로 파라미터는 카메라
    높이뿐. 반환 X = 터치라인 방향(우+), Y = 경기장 안쪽 깊이(m).
    수평선 근처(el > −0.5°)는 기하적으로 불안정하므로 제외.

    입력 행 형식: [cx, cy, w, h, (id), (h,s,v)] — 4열 이상이면 동작.
    반환: [(X, Y, track_id, 원소 인덱스), ...]
    """
    t_top, t_bot = np.tan(np.deg2rad(el_top_deg)), np.tan(np.deg2rad(el_bottom_deg))
    f = pano_h / (t_top - t_bot)
    yaw_span = pano_w / f
    out = []
    for j, p in enumerate(players_row):
        if len(p) < 4:
            continue
        foot_x, foot_y = p[0], p[1] + p[3] / 2.0
        phi = (foot_x / max(pano_w - 1, 1) - 0.5) * yaw_span
        t = t_top + (foot_y / max(pano_h - 1, 1)) * (t_bot - t_top)
        if t > -0.0087:                  # el > -0.5도 — 수평선 근처 제외
            continue
        d = cam_height / (-t)
        tid = int(p[4]) if len(p) >= 5 else -1
        out.append((d * np.sin(phi), d * np.cos(phi), tid, j))
    return out


def same_spot_spans(linked, f0, f1, radius=60.0, static_r80=40.0):
    """기준 트랙(f0~f1 구간)과 '같은 자리'의 정적 트랙 구간 전부 반환.

    낙엽·마킹 같은 오브젝트는 한 위치에서 시간대만 다른 여러 트랙으로
    쪼개져 반복 검출된다. 기준 트랙의 중앙 위치에서 radius 이내이고
    자체 요동(r80)이 static_r80 이하인 트랙을 모두 모아, 한 번의 무시로
    일괄 처리할 수 있게 한다. 반환: [(시작, 끝) 프레임, ...] (기준 포함).

    기준 트랙 선정: (f0, f1) 과 **정확히 일치**하는 트랙 우선, 없으면
    겹치는 것 중 가장 짧은(특정적인) 것 하나 — 겹치는 트랙 전부를
    합쳐 중앙을 내면 동시 트랙(공 2개) 상황에서 큰 트랙의 자리가
    이겨서 엉뚱한 트랙까지 같이 무시된다 (pano_5316 40:26 실사례).
    """
    idx, tracks = linked["idx"], linked["tracks"]
    cover = [t for t in tracks
             if idx[t["i"][0]] <= f1 and f0 <= idx[t["i"][-1]]]
    if not cover:
        return []
    exact = [t for t in cover
             if int(idx[t["i"][0]]) == int(f0)
             and int(idx[t["i"][-1]]) == int(f1)]
    pick = exact[0] if exact else min(
        cover, key=lambda t: idx[t["i"][-1]] - idx[t["i"][0]])
    ref_med = np.median(pick["pts"], axis=0)
    # 기준 자신은 항상 위치와 함께 포함 — "이 트랙은 공이 아님" 이지
    # "이 시간대엔 공이 없음" 이 아니다: 동시간의 다른 트랙(진짜
    # 매치볼)은 살아남아야 하므로 시간만의 범위는 절대 만들지 않는다.
    out = [(int(idx[pick["i"][0]]), int(idx[pick["i"][-1]]),
            round(float(ref_med[0]), 1), round(float(ref_med[1]), 1))]
    for t in tracks:
        if t is pick:
            continue
        med = np.median(t["pts"], axis=0)
        if np.hypot(*(med - ref_med)) > radius:
            continue
        r80 = (float(np.percentile(np.hypot(*(t["pts"] - med).T), 80))
               if len(t["i"]) > 1 else 0.0)
        if r80 <= static_r80:
            out.append((int(idx[t["i"][0]]), int(idx[t["i"][-1]]),
                        round(float(med[0]), 1), round(float(med[1]), 1)))
    return sorted(out)


def export_training_labels(analysis, keyframes=None, ignore_ranges=None,
                           force_ranges=None, linked=None):
    """사용자 마킹을 커스텀 공 검출 모델 학습 라벨로 변환 (P03-5).

    - 수락 트랙 위 검출     → "ball" (자동 양성)
    - 갭필/시드 주입 검출   → "ball_lowconf" (양성, 저신뢰 태그 —
      후보 행 길이 6 이 주입 마커, 6번째 원소가 원래 conf)
    - 무시 구간 안의 후보   → "not_ball" (하드 네거티브 — 낙엽 등,
      수락 판정보다 후순위라 승격 트랙은 양성 유지)
    - 사용자 키프레임       → "ball_manual" (사람이 확인한 양성, 박스 없음)
    승격(force_ranges) 트랙은 수락 경로를 거쳐 양성에 포함된다.

    반환: [{"frame", "x", "y", "w", "h", "conf", "label"}, ...]
    원본 분석은 수정하지 않는다 (마킹은 비파괴).
    """
    idx, acc, _spans = accept_ball_tracks(
        analysis, ignore_ranges=ignore_ranges, force_ranges=force_ranges,
        linked=linked, log=None)
    ig = [(r[0], r[1]) for r in (ignore_ranges or [])]
    cands = analysis.get("ball_cands")
    out = []
    for si in range(len(idx)):
        f = int(idx[si])
        row = (cands[si] if cands is not None else
               ([list(analysis["balls"][si])]
                if analysis["balls"][si] is not None else []))
        in_ig = any(lo <= f <= hi for lo, hi in ig)
        ax = acc[si] if acc is not None and si < len(acc) else (np.nan, np.nan)
        for p in row:
            rec = {"frame": f, "x": float(p[0]), "y": float(p[1]),
                   "w": float(p[3]) if len(p) >= 5 else 0.0,
                   "h": float(p[4]) if len(p) >= 5 else 0.0,
                   "conf": float(p[5] if len(p) >= 6 else p[2])}
            if np.isfinite(ax[0]) \
                    and np.hypot(ax[0] - p[0], ax[1] - p[1]) <= 25.0:
                rec["label"] = ("ball_lowconf" if len(p) >= 6 else "ball")
            elif in_ig:
                rec["label"] = "not_ball"
            else:
                continue                  # 자동 기각(불확실) — 라벨로 안 씀
            out.append(rec)
    for kf, kx, ky in (keyframes or []):
        out.append({"frame": int(kf), "x": float(kx), "y": float(ky),
                    "w": 0.0, "h": 0.0, "conf": 1.0, "label": "ball_manual"})
    out.sort(key=lambda r: r["frame"])
    return out


def _gaussian1d(x, sigma):
    """NaN 없는 1D 배열의 가우시안 스무딩 (경계는 edge 패딩, zero-phase)."""
    if sigma <= 0:
        return x.copy()
    n = int(4 * sigma) | 1
    t = np.arange(n) - n // 2
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(n // 2, x[0]), x, np.full(n // 2, x[-1])])
    return np.convolve(pad, k, "valid")


def _ball_candidates(analysis, ball_conf):
    """샘플별 공 후보 [(x, y, conf), ...] — 신형 ball_cands 우선, 구형은 balls."""
    bc = analysis.get("ball_cands")
    if bc is not None:
        return [[(p[0], p[1], p[2]) for p in row if p[2] >= ball_conf]
                for row in bc]
    return [[(b[0], b[1], b[2])] if b is not None and b[2] >= ball_conf else []
            for b in analysis["balls"]]


def link_ball_tracks(analysis, ball_conf=0.25, max_jump_per_frame=120.0):
    """공 후보 트랙 연결 (느린 단계 — 분석에만 의존하므로 캐시 가능).

    샘플당 후보가 여러 개면 conf 순으로 각자 가장 가까운 트랙에 배정
    (트랙당 샘플당 1개). 남는 후보는 새 트랙 — 미끼와 진짜 공이 같은
    시간대에 별도 트랙으로 공존하고, 수락/무시가 트랙 단위로 고른다.
    반환 dict 를 accept_ball_tracks(linked=...) 에 넘기면 수락 단계만
    다시 돌면 된다 (즉각 반응).
    """
    fps = analysis["fps"]
    idx = np.array(analysis["frames"])
    n = len(idx)
    cands = _ball_candidates(analysis, ball_conf)

    assoc_gap = 2.5 * fps               # 이 이상 끊기면 새 트랙
    tracks: list[dict] = []             # {"i": [샘플], "x": [], "y": []}
    for i in range(n):
        used = set()
        for x, y, c in sorted(cands[i], key=lambda p: -p[2]):
            best, best_d = None, None
            for ti, t in enumerate(tracks):
                if ti in used:
                    continue
                gap = idx[i] - idx[t["i"][-1]]
                if not 0 < gap <= assoc_gap:
                    continue
                # 최근 5점 평균 기준 — 검출 노이즈로 트랙이 갈라지지 않게.
                # +150px 은 빠른 이동 시 평균 지연 보상. 거리 상한 700px:
                # 긴 공백 뒤 원거리 오브젝트가 흡수되는 것을 차단.
                rx = sum(t["x"][-5:]) / len(t["x"][-5:])
                ry = sum(t["y"][-5:]) / len(t["y"][-5:])
                d = float(np.hypot(x - rx, y - ry))
                if (d <= min(max_jump_per_frame * gap, 700) + 150
                        and (best_d is None or d < best_d)):
                    best, best_d = ti, d
            if best is None:
                tracks.append({"i": [i], "x": [x], "y": [y]})
                used.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["i"].append(i)
                t["x"].append(x)
                t["y"].append(y)
                used.add(best)
    for t in tracks:
        t["i"] = np.array(t["i"])
        t["pts"] = np.column_stack([t["x"], t["y"]]).astype(float)
        del t["x"], t["y"]

    # 샘플별 선수 통계 프리컴퓨트 (공 부재 시 목표/줌 계산용 — 캐시 대상)
    p_cnt, p_tx, p_ty, p_span = player_aggregates(analysis)
    return {"idx": idx, "tracks": tracks, "fps": fps,
            "p_cnt": p_cnt, "p_tx": p_tx, "p_ty": p_ty, "p_span": p_span}


def player_aggregates(analysis, exclude_tids=None, base=None):
    """샘플별 선수 통계 (공 부재 시 크롭 목표/줌) — 숨김 제외 지원.

    exclude_tids (숨긴 관중·오인식 대표+멤버) 를 빼고 집계 — 숨겨도
    크롭이 그쪽을 보던 문제의 수정 지점. base=(p_cnt,p_tx,p_ty,p_span)
    (linked 캐시)가 있으면 **제외 대상이 등장하는 샘플만** 재계산 —
    전체 재집계 6.8s → 수백 ms (숨김 때마다 계획 재계산이 느리다는
    사용자 보고의 수정).
    """
    n = len(analysis["frames"])
    ex = set(exclude_tids or ())

    def _one(rows):
        if ex:
            rows = [r for r in rows
                    if len(r) < 5 or int(r[4]) not in ex]
        pl = np.asarray(rows, dtype=float)
        pl = pl[:, :2] if pl.ndim == 2 else pl.reshape(-1, 2)
        cnt = len(pl)
        if cnt >= 3:
            med = np.median(pl[:, 0])
            keep = pl[np.abs(pl[:, 0] - med)
                      < analysis.get("pano_w", 5906) * 0.25]
            tx = keep[:, 0].mean() if len(keep) else np.nan
            ty = keep[:, 1].mean() if len(keep) else np.nan
            span = (np.percentile(pl[:, 0], 90)
                    - np.percentile(pl[:, 0], 10))
            return cnt, tx, ty, span
        return cnt, np.nan, np.nan, 0.0

    if ex and base is not None:
        p_cnt = np.array(base[0], int)
        p_tx = np.array(base[1], float)
        p_ty = np.array(base[2], float)
        p_span = np.array(base[3], float)
        for i, rows in enumerate(analysis["players"]):
            if any(len(r) >= 5 and int(r[4]) in ex for r in rows):
                p_cnt[i], p_tx[i], p_ty[i], p_span[i] = _one(rows)
        return p_cnt, p_tx, p_ty, p_span
    p_cnt = np.zeros(n, int)
    p_tx = np.full(n, np.nan)
    p_ty = np.full(n, np.nan)
    p_span = np.zeros(n)
    for i, rows in enumerate(analysis["players"]):
        p_cnt[i], p_tx[i], p_ty[i], p_span[i] = _one(rows)
    return p_cnt, p_tx, p_ty, p_span


def _track_iso_frac(analysis, t, iso_px):
    """트랙 점별 최근접 선수 거리 > iso_px 비율 (선수 없으면 0=근접 취급)."""
    far = 0
    n_players = len(analysis["players"])
    for k, i in enumerate(t["i"]):
        if int(i) >= n_players:               # 다른 analysis 의 낡은 인덱스
            continue
        prow = analysis["players"][int(i)]
        if not prow:
            continue
        pl = np.asarray(prow, dtype=float)[:, :2]
        if np.hypot(*(pl - t["pts"][k]).T).min() > iso_px:
            far += 1
    return far / max(len(t["i"]), 1)


def accept_ball_tracks(analysis, ball_conf=0.25, max_jump_per_frame=120.0,
                       decoy_static_px=30.0, decoy_iso_px=700.0,
                       decoy_win_sec=3.0, ignore_ranges=None, force_ranges=None,
                       linked=None, spot_radius=60.0, log=None):
    """공 트랙 수락/기각 (트랙 단위 오검출 처리).

    트랙 통계(정지/고립/중복)로 통째로 기각 — 정지 낙엽, 장외에서 움직이는
    다른 공, 순간 오검출. ignore_ranges 와 겹치는 트랙은 사용자 지정
    오인식 — 무조건 기각. force_ranges (승격) 는 그 반대 — 자동 필터가
    기각했더라도 그 (시각,자리) 를 지나는 트랙을 무조건 수락한다 (사용자가
    회색 공을 클릭해 트랙으로 올린 경우). 무시가 승격보다 우선한다.
    linked 에 link_ball_tracks 결과를 주면 연결 단계를 건너뛴다
    (GUI 즉각 반응용).

    반환: (idx, ball(n,2), spans) — spans 는 수락 트랙의 (시작,끝) 프레임.
    """
    if linked is None:
        linked = link_ball_tracks(analysis, ball_conf, max_jump_per_frame)
    idx, tracks, fps = linked["idx"], linked["tracks"], linked["fps"]
    n = len(idx)

    def _track_stats(t):
        pts = t["pts"]
        # 정지 판정은 중앙값 기준 80% 반경 — 흡수된 이탈 샘플에 강건
        med = np.median(pts, axis=0)
        r80 = float(np.percentile(np.hypot(*(pts - med).T), 80))
        dur = (idx[t["i"][-1]] - idx[t["i"][0]]) / fps
        if "iso_frac" not in t:
            t["iso_frac"] = _track_iso_frac(analysis, t, decoy_iso_px)
        return r80, dur, t["iso_frac"]

    def _ignored(t):
        # 항목: (f0, f1) = 시간만 (구형), (f0, f1, x, y) = 시간+자리.
        # 다중 후보에서는 같은 시간대에 미끼/진짜 공 트랙이 공존하므로
        # 위치가 있으면 그 자리(150px)의 트랙만 기각한다.
        if not ignore_ranges:
            return False
        f0, f1 = idx[t["i"][0]], idx[t["i"][-1]]
        med = None
        for rng in ignore_ranges:
            lo, hi = rng[0], rng[1]
            if not (f0 <= hi and lo <= f1):
                continue
            if len(rng) >= 4:
                if med is None:
                    med = np.median(t["pts"], axis=0)
                if np.hypot(med[0] - rng[2], med[1] - rng[3]) > 150:
                    continue
            return True
        return False

    def _forced(t):
        # 승격 항목 (f, x, y): 그 시각(±0.5s) 부근에서 그 자리(spot_radius)
        # 를 지나는 트랙이면 자동 필터를 우회해 수락.
        if not force_ranges:
            return False
        for rng in force_ranges:
            ff, fx, fy = rng[0], rng[1], rng[2]
            for k, i in enumerate(t["i"]):
                if abs(idx[i] - ff) <= 0.5 * fps and \
                        np.hypot(t["pts"][k][0] - fx,
                                 t["pts"][k][1] - fy) <= spot_radius:
                    return True
        return False

    accepted: list[dict] = []
    covered = np.zeros(n, bool)
    rej = {"static": 0, "isolated": 0, "overlap": 0, "user": 0}
    # 점수: 길이 + 선수 근접 가점 (경기 공은 선수 곁에 오래 머문다)
    order = sorted(tracks, key=lambda t: -(len(t["i"]) * (2.0 - _track_stats(t)[2])))
    # 승격 트랙을 먼저 처리해 커버리지를 선점 (사용자 의도 우선). stable sort.
    order = sorted(order, key=lambda t: 0 if _forced(t) else 1)
    for t in order:
        if _ignored(t):
            rej["user"] += 1            # 사용자 지정 오인식 구간
            continue
        if _forced(t):                  # 사용자 승격 — 자동 필터 우회
            accepted.append(t)
            covered[t["i"][0]:t["i"][-1] + 1] = True
            continue
        r80, dur, iso_frac = _track_stats(t)
        if r80 <= decoy_static_px and iso_frac > 0.5 and dur >= decoy_win_sec:
            rej["static"] += 1          # 정지 + 고립 (낙엽·마킹)
            continue
        # 실측: 진짜 공 트랙의 고립 비율은 0.00~0.04, 미끼는 0.8 안팎 —
        # 5초 이상 지속 트랙이 절반 넘게 고립돼 있으면 경기 공이 아니다
        if dur >= 5.0 and iso_frac > 0.5:
            rej["isolated"] += 1        # 선수와 무관한 물체 (낙엽·장외 공)
            continue
        lo, hi = t["i"][0], t["i"][-1]
        if covered[lo:hi + 1].mean() > 0.5:
            rej["overlap"] += 1         # 이미 수락된 트랙과 시간 중복 (경합 오검출)
            continue
        accepted.append(t)
        covered[lo:hi + 1] = True

    # 무시된 트랙들의 위치(스팟): 수락 트랙에 흡수된 같은 자리 샘플도
    # 제거해, 크롭이 오인식 지점으로 끌려가는 것을 막는다 (시간+공간 무시)
    spots = [np.median(t["pts"], axis=0) for t in tracks
             if _ignored(t) and len(t["i"]) >= 2]
    ball = np.full((n, 2), np.nan)
    dropped_spot = 0
    for t in accepted:                  # 점수 순 — 먼저 수락된 트랙이 우선
        for k, i in enumerate(t["i"]):
            if not np.isnan(ball[i, 0]):
                continue
            if spots and min(float(np.hypot(*(t["pts"][k] - sp)))
                             for sp in spots) <= spot_radius:
                dropped_spot += 1
                continue
            ball[i] = t["pts"][k]
    if log and dropped_spot:
        log(f"[plan] 무시 지점 근처 샘플 {dropped_spot}개 추가 제거")
    if log and (tracks or accepted):
        log(f"[plan] 공 트랙 {len(tracks)}개 → 수락 {len(accepted)}개 "
            f"(기각: 정지미끼 {rej['static']}, 장외고립 {rej['isolated']}, "
            f"중복 {rej['overlap']}, 사용자무시 {rej['user']})")
    spans = sorted((int(idx[t["i"][0]]), int(idx[t["i"][-1]])) for t in accepted)
    return idx, ball, spans


def build_plan(analysis, pano_w, pano_h, out_w=1920, out_h=1080,
               ball_conf=0.25, max_jump_per_frame=120.0, gap_fill_sec=2.0,
               sigma_slow=1.2, sigma_fast=0.35, fast_err_px=400.0,
               zoom_margin=260.0, top_margin=160, near_widen=1.6,
               far_zoom=1.0,
               decoy_static_px=30.0, decoy_iso_px=700.0, decoy_win_sec=3.0,
               keyframes=None, kf_suppress_sec=1.5, kf_bridge_sec=8.0,
               wide=False, ignore_ranges=None, force_ranges=None,
               linked=None, exclude_tids=None, log=print):
    """검출 궤적 → 프레임별 (cx, cy, crop_w) 계획.

    - 공: conf 게이팅 + 점프 게이팅, gap_fill_sec 까지 선형 보간.
    - 정적 미끼 필터: decoy_win_sec 창 내내 decoy_static_px 안에 정지해 있고
      모든 선수로부터 decoy_iso_px 이상 고립된 '공'은 오검출(낙엽·마킹 등)로
      기각. 실제 공은 경기 중 장시간 정지+고립 상태가 없다 (코너킥 준비 등
      짧은 예외는 줌아웃으로 처리돼 해가 없음).
    - keyframes: [(frame_idx, x, y), ...] 사용자 지정 공 위치. 자동 검출보다
      우선한다 — 키프레임 ±kf_suppress_sec 의 자동 샘플은 버리고, 인접
      키프레임끼리는 kf_bridge_sec 까지 직접 보간으로 잇는다.
    - 공 없는 구간: 선수 중앙값 주변(트림 평균) 목표 + 줌아웃(선수 분포 커버).
    - 스무딩: 기본 sigma_slow(초). 잔차가 fast_err_px 를 넘는 구간(공이 빠르게
      이동)만 sigma_fast 추종으로 크로스페이드 — "평상시 최대한 부드럽게".
    - top_margin: 파노라마 상단 검은 스티칭 경계를 크롭에 넣지 않기 위한
      상단 여백(px). 최대 줌아웃 높이도 이만큼 줄어든다.
    - near_widen: 공이 화면 아래(근경)일수록 크롭을 넓힘 — 가까운 선수는
      크게 보이므로 타이트한 줌인이 불필요. 원경 1.0배 → 최하단 near_widen배.
    - far_zoom: 원경 공에 대한 추가 줌인 배율 (1.0=없음). 크롭 폭이
      out_w/far_zoom 까지 줄어들며 출력 시 업스케일된다 (원경은 원본
      디테일이 작아 체감 손실 미미).
    - wide=True: 감상용 와이드 모드 — 크롭 폭을 항상 최대(세로 꽉 채움)로
      고정하고 가로만 완만하게 팬. out_w/out_h 를 21:9 등으로 주고
      sigma_slow 를 크게(권장 3.0) 주면 방송 와이드샷처럼 움직인다.
    """
    fps = analysis["fps"]
    total = analysis["total_frames"]
    idx = np.array(analysis["frames"])
    n = len(idx)

    # --- 0~1. 공 트랙 수락 (accept_ball_tracks 참조) --------------------
    if linked is None:
        linked = link_ball_tracks(analysis, ball_conf, max_jump_per_frame)
    _, ball, _ = accept_ball_tracks(
        analysis, ball_conf=ball_conf, max_jump_per_frame=max_jump_per_frame,
        decoy_static_px=decoy_static_px, decoy_iso_px=decoy_iso_px,
        decoy_win_sec=decoy_win_sec, ignore_ranges=ignore_ranges,
        force_ranges=force_ranges, linked=linked, log=log)

    # --- 1.5 사용자 키프레임 병합 (자동보다 우선) -----------------------
    # 키프레임: [frame, x, y] 또는 [frame, x, y, crop_w] (4번째=줌 오버라이드,
    # 크롭 폭 px; None 이면 자동 깊이 기반 줌 유지).
    kf_idx = []
    kf_zoom = {}                          # 샘플 i -> 사용자 지정 크롭 폭(px)
    kfs_norm = [(k[0], k[1], k[2], (k[3] if len(k) > 3 else None))
                for k in sorted(keyframes)] if keyframes else []
    if keyframes:
        sup = kf_suppress_sec * fps
        for kf_f, kx, ky, kw in kfs_norm:
            near = np.abs(idx - kf_f) <= sup
            ball[near] = np.nan          # 키프레임 주변 자동 샘플 무효화
        for kf_f, kx, ky, kw in kfs_norm:
            i = int(np.argmin(np.abs(idx - kf_f)))
            ball[i] = (kx, ky)
            kf_idx.append(i)
            if kw is not None:
                kf_zoom[i] = float(kw)

    # --- 2. 갭 보간 (짧은 가림/미검출) ----------------------------------
    known = ~np.isnan(ball[:, 0])
    gap_max = gap_fill_sec * fps
    filled = ball.copy()
    ki = np.where(known)[0]
    kf_set = set(kf_idx)
    for a, b_ in zip(ki[:-1], ki[1:]):
        gap = idx[b_] - idx[a]
        # 인접 키프레임 쌍은 더 긴 간격도 직접 보간으로 잇는다
        limit = kf_bridge_sec * fps if (a in kf_set and b_ in kf_set) else gap_max
        if 0 < gap <= limit:
            for col in (0, 1):
                filled[a:b_ + 1, col] = np.interp(idx[a:b_ + 1], [idx[a], idx[b_]],
                                                  [ball[a, col], ball[b_, col]])
    known = ~np.isnan(filled[:, 0])

    # --- 3. 목표점 + 줌 ------------------------------------------------
    max_crop_w = int(min(pano_w, (pano_h - top_margin) * out_w / out_h))
    min_crop_w = out_w / 6.0            # 사용자 줌인 하한 (업스케일 한도)
    tx = np.empty(n)
    ty = np.empty(n)
    zw = np.full(n, float(out_w))
    prev = (pano_w / 2, pano_h * 0.55)
    field_top = analysis.get("field_top_frac", 0.26) * pano_h
    if exclude_tids:
        # 숨긴 선수(관중·오인식) 제외 — 등장 샘플만 부분 재집계
        p_cnt, p_tx, p_ty, p_span = player_aggregates(
            analysis, exclude_tids=set(exclude_tids),
            base=(linked["p_cnt"], linked["p_tx"],
                  linked["p_ty"], linked["p_span"]))
    else:
        p_cnt, p_tx = linked["p_cnt"], linked["p_tx"]
        p_ty, p_span = linked["p_ty"], linked["p_span"]
    for i in range(n):                      # 프리컴퓨트 덕에 경량 루프
        if known[i]:
            tx[i], ty[i] = filled[i]
            # 원경 out_w/far_zoom(추가 줌인) → 최하단 out_w*near_widen 선형 보간
            depth_t = min(max((ty[i] - field_top) / max(pano_h - field_top, 1), 0.0), 1.0)
            zw_far = out_w / max(far_zoom, 1.0)
            zw[i] = min(zw_far + (out_w * near_widen - zw_far) * depth_t, max_crop_w)
        elif p_cnt[i] >= 3:
            tx[i], ty[i] = p_tx[i], p_ty[i]
            # 공 없이 선수만 있을 때: 무리가 원경(위쪽)에 몰려 있으면
            # 화면 하한(out_w)을 고집하지 말고 적극 줌인 — 원경 무리는
            # 픽셀 스팬이 작아 1:1 크롭에선 콩알이 된다. 깊이에 따라
            # 하한을 out_w/줌 ↔ out_w 로 보간 (far_zoom 과 별개 최소
            # 1.35 배 보장, 사용자 설정이 더 크면 그쪽).
            depth_t = min(max((ty[i] - field_top)
                              / max(pano_h - field_top, 1), 0.0), 1.0)
            # 원경 판정 램프: 깊이 0.35 까지는 최대 줌인, 0.75 부터는
            # 기존 하한(out_w) — 근경 무리의 동작은 바뀌지 않는다
            t2 = min(max((depth_t - 0.35) / 0.40, 0.0), 1.0)
            fb_zoom = max(far_zoom, 1.35)
            floor_w = out_w / fb_zoom + (out_w - out_w / fb_zoom) * t2
            zw[i] = min(max(p_span[i] + 2 * zoom_margin, floor_w),
                        max_crop_w)
        else:
            tx[i], ty[i] = prev
            zw[i] = max_crop_w   # 정보 부족 — 최대한 넓게
        prev = (tx[i], ty[i])

    # 사용자 키프레임 줌 오버라이드 (자동 깊이 줌보다 우선)
    for i, kw in kf_zoom.items():
        zw[i] = min(max(kw, min_crop_w), max_crop_w)

    # --- 4. 프레임별 업샘플 + 적응 스무딩 -------------------------------
    if wide:
        zw[:] = max_crop_w              # 항상 최대 폭 (줌 변동 없음)
    fr = np.arange(total)
    fx = np.interp(fr, idx, tx)
    fy = np.interp(fr, idx, ty)
    fz = np.interp(fr, idx, zw)
    slow = _gaussian1d(fx, sigma_slow * fps)
    fast = _gaussian1d(fx, sigma_fast * fps)
    err = np.abs(fx - slow)
    w = 1.0 / (1.0 + np.exp(-(err - fast_err_px) / (fast_err_px * 0.25)))
    w = _gaussian1d(w, 0.5 * fps)
    cx = (1 - w) * slow + w * fast
    cy = _gaussian1d(fy, 2.0 * fps)
    cw = _gaussian1d(fz, 2.0 * fps)
    # 키프레임 앵커: 스무딩이 뭉갠 위치를 국소 보정해 클릭 시점에는
    # 카메라가 정확히 그 지점을 보도록 보장 (가우시안 창으로 부드럽게)
    if keyframes:
        anchor_sigma = 0.8 * fps
        for kf_f, kx, ky, kw in kfs_norm:
            j = int(np.clip(kf_f, 0, total - 1))
            g = np.exp(-0.5 * ((fr - j) / anchor_sigma) ** 2)
            cx = cx + (kx - cx[j]) * g
            cy = cy + (ky - cy[j]) * g
            if kw is not None and not wide:
                kwc = min(max(float(kw), min_crop_w), max_crop_w)
                cw = cw + (kwc - cw[j]) * g
    # 상단 클램프 완화 (높이 뜬 공): top_margin 은 상단 검은 경계 보호가
    # 목적인데, 공이 실측으로 그 위에 있으면 공을 놓치는 클램프가 된다 —
    # 공을 아는 구간에서 계획이 원하는 만큼(0 까지) 상한을 따라 올린다.
    h_arr = np.minimum(np.minimum(cw, pano_w),
                       pano_h * out_w / out_h) * out_h / out_w
    known_f = _gaussian1d(
        np.interp(fr, idx, known.astype(float)), 0.5 * fps) > 0.5
    top_lim = np.full(total, float(top_margin))
    want = cy - h_arr / 2
    m = known_f & (want < top_margin)
    top_lim[m] = np.clip(want[m], 0.0, top_margin)
    if log:
        log(f"[plan] 공 궤적 {known.mean():.0%} (보간 포함), "
            f"빠른 추종 구간 {(w > 0.5).mean():.0%}, "
            f"줌아웃(>1.05x) {(cw > out_w * 1.05).mean():.0%}"
            + (f", 상단 완화 {m.mean():.0%}" if m.any() else ""))
    return {"cx": cx, "cy": cy, "crop_w": cw, "top_margin": top_margin,
            "top_lim": top_lim}


def build_radar_data(analysis, teams, calib=None, field_size=(105.0, 68.0),
                     extra_players=None, palette=None):
    """내보내기 레이더용 샘플별 필드 좌표 데이터.

    teams = {track_id: 역할} (사용자 역할 오버라이드 포함), calib 가 있으면
    경기장 절대 좌표(pano_to_field), 없으면 카메라 기준 근사(가정 위치).
    반환: {frames, points, balls, length, width, palette}.
    """
    from .field import pano_to_field
    pano_w, pano_h = analysis["pano_w"], analysis["pano_h"]
    frames = np.asarray(analysis["frames"], dtype=np.int64)
    cam = (0.0, -(field_size[1] / 2.0 + 5.0))
    pts, balls = [], []
    for si, prow in enumerate(analysis["players"]):
        rows = list(prow) + ((extra_players or {}).get(si, []))
        out = []
        if calib is not None:
            feet = [(p[0], p[1] + p[3] / 2.0,
                     int(p[4]) if len(p) >= 5 else -1)
                    for p in rows if len(p) >= 4]
            if feet:
                fc = pano_to_field(calib, [(a, b) for a, b, _ in feet])
                for (gx, gy), (_, _, tid) in zip(fc, feet):
                    if np.isfinite(gx):
                        out.append((float(gx), float(gy),
                                    int(teams.get(tid, 2))))
        else:
            for X, Y, tid, j in ground_positions(rows, pano_w, pano_h):
                out.append((cam[0] + X, cam[1] + Y, int(teams.get(tid, 2))))
        pts.append(out)
        bb = analysis["balls"][si] if si < len(analysis["balls"]) else None
        bg = None
        if bb is not None:
            if calib is not None:
                g = pano_to_field(calib, [[bb[0], bb[1]]])[0]
                if np.isfinite(g[0]):
                    bg = (float(g[0]), float(g[1]))
            else:
                g = ground_positions([[bb[0], bb[1], 0.0, 0.0]],
                                     pano_w, pano_h)
                if g:
                    bg = (cam[0] + g[0][0], cam[1] + g[0][1])
        balls.append(bg)
    # 공중볼 보정: 지면 투영 왜곡(높이 → 카메라 반대쪽 밀림)을 탄도
    # 피팅으로 되돌린다 — 레이더에서 공중볼 XY 가 직선이 된다.
    if calib is not None:
        from .airborne import correct_ball_track
        t = frames / analysis["fps"]
        g = np.array([[b[0], b[1]] if b is not None else [np.nan, np.nan]
                      for b in balls])
        corr, z, segs = correct_ball_track(
            t, g, (calib["ex"], calib["ey"]), calib["h"])
        for i0, i1, _fit in segs:
            for si in range(i0, i1 + 1):
                if balls[si] is not None:
                    balls[si] = (float(corr[si, 0]), float(corr[si, 1]))
    return {"frames": frames, "points": pts, "balls": balls,
            "length": float(field_size[0]), "width": float(field_size[1]),
            "palette": palette or {}}


def draw_radar_panel(radar, si, panel_w):
    """샘플 si 의 탑다운 레이더 패널 (BGR). 등방 축척 — 경기장 사각형이
    입력한 실측 크기(예: 100×62m) 비율 그대로 그려진다."""
    L, Wd = radar["length"], radar["width"]
    mx = 4.0                                  # 바깥 여백 (m)
    s = panel_w / (L + 2 * mx)                # px/m (가로=세로 동일)
    ph = int(round((Wd + 2 * mx) * s)) & ~1
    img = np.full((max(ph, 2), panel_w, 3), (26, 52, 30), np.uint8)

    def px(X, Y):
        return (int(round((X + L / 2 + mx) * s)),
                int(round((Wd / 2 + mx - Y) * s)))

    wcol = (235, 235, 235)
    lw = max(1, panel_w // 300)
    cv2.rectangle(img, px(-L / 2, Wd / 2), px(L / 2, -Wd / 2), wcol, lw)
    cv2.line(img, px(0, -Wd / 2), px(0, Wd / 2), wcol, lw)
    cv2.circle(img, px(0, 0), int(round(9.15 * s)), wcol, lw)
    pal = radar.get("palette", {})
    r_dot = max(2, panel_w // 110)
    for X, Y, role in radar["points"][si]:
        cv2.circle(img, px(X, Y), r_dot,
                   tuple(int(v) for v in pal.get(role, (160, 160, 160))), -1)
    if radar["balls"][si] is not None:
        q = px(*radar["balls"][si])
        cv2.circle(img, q, r_dot + 2, (30, 30, 30), 2)   # 외곽선 — 대비
        cv2.circle(img, q, r_dot + 1, (0, 140, 255), -1)   # 공 = 주황 (BGR)
    return img


def clock_string(clock, frame):
    """파노라마 프레임 → "1H 12:34" (+스코어) 경기 시계 문자열.

    clock = {anchor_f, fps, base_s, tag, pauses[[f0,f1]], score}.
    앵커(킥오프) 이전은 base_s 에 고정, 중단 구간(pauses)에서는 시계가
    멈춘다 (hydration break 등). score = (팀A, 팀B, [[f, 1|2]]) 이면
    frame 까지의 골 수를 세어 "A 1-0 B" 를 덧붙인다 — 득점 순간부터
    스코어가 바뀐다.
    """
    fps = clock["fps"]
    anchor = clock["anchor_f"]
    el = max(0.0, (frame - anchor) / fps)
    for p0, p1 in clock.get("pauses") or []:
        el -= max(0.0, (min(frame, p1) - max(anchor, p0)) / fps)
    t = clock.get("base_s", 0.0) + max(0.0, el)
    s = f"{clock.get('tag', '')} {int(t // 60):02d}:{int(t % 60):02d}".strip()
    sc = clock.get("score")
    if sc:
        a, b, goals = sc
        ga = sum(1 for f_, tm in goals if f_ <= frame and tm == 1)
        gb = sum(1 for f_, tm in goals if f_ <= frame and tm == 2)
        s += f"  {a} {ga}-{gb} {b}"
    return s


def _draw_clock(img, text, out_w, out_h, alpha=0.55):
    """좌상단 반투명 박스 + 흰 글자 시계 (미니맵과 동일한 스케일 규칙)."""
    scale = out_h / 1080.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.9 * scale, max(2, int(round(2 * scale)))
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    mgn = out_w // 96
    pad = int(round(8 * scale))
    roi = img[mgn:mgn + tht + base + 2 * pad, mgn:mgn + tw + 2 * pad]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    cv2.putText(img, text, (mgn + pad, mgn + pad + tht), font, fs,
                (255, 255, 255), th, cv2.LINE_AA)


def render_plan(pano_path, out_path, plan, out_w=1920, out_h=1080,
                codec="libx264", crf=20, log=print, progress=None,
                cancel=None, radar=None, radar_alpha=0.55,
                start=0, end=None, clock=None):
    """2패스: 계획대로 크롭(필요 시 줌아웃 다운스케일)해 인코딩. fps 반환.

    radar(build_radar_data 결과)가 있으면 우하단에 반투명 탑다운
    레이더를 합성한다. start/end(프레임)로 구간 내보내기 — 오디오도
    같은 오프셋으로 잘라 맞춘다. clock(clock_string 입력)이 있으면
    좌상단에 경기 시계(+스코어)를 얹는다 — cv2 폰트 제약으로 ASCII
    (1H/2H, 팀 이름은 호출부에서 ASCII 보장)."""
    import subprocess
    import time

    from .encoders import encoder_args, ffmpeg_bin
    cap = cv2.VideoCapture(str(pano_path))
    pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = len(plan["cx"])
    end = n if end is None else max(0, min(int(end), n))
    start = max(0, min(int(start), end))
    audio_in = (["-ss", f"{start / fps:.3f}"] if start else []) \
        + ["-i", str(pano_path)]
    cmd = ([ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-"]
           + audio_in + ["-map", "0:v", "-map", "1:a?"]
           + encoder_args(codec, crf)
           + ["-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out_path)])
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cx, cy, cw = plan["cx"], plan["cy"], plan["crop_w"]
    top_margin = int(plan.get("top_margin", 0))
    top_arr = plan.get("top_lim")
    r_frames = radar["frames"] if radar is not None else None
    r_cache = {"si": -1, "img": None}
    panel_w = (out_w // 5) & ~1
    t0 = time.perf_counter()
    done = 0
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end):
            if cancel is not None and cancel():
                log("[render] 사용자 취소")
                break
            ok, frame = cap.read()
            if not ok:
                break
            w = int(round(min(cw[i], pano_w, pano_h * out_w / out_h))) & ~1
            h = int(round(w * out_h / out_w)) & ~1
            x0 = int(round(cx[i] - w / 2))
            y0 = int(round(cy[i] - h / 2))
            x0 = max(0, min(x0, pano_w - w))
            tm = int(top_arr[i]) if top_arr is not None else top_margin
            y0 = max(min(tm, pano_h - h), min(y0, pano_h - h))
            crop = frame[y0:y0 + h, x0:x0 + w]
            if w != out_w:
                interp = cv2.INTER_AREA if w > out_w else cv2.INTER_CUBIC
                crop = cv2.resize(crop, (out_w, out_h), interpolation=interp)
            if radar is not None and len(r_frames):
                si = int(np.searchsorted(r_frames, i))
                si = min(si, len(r_frames) - 1)
                if si > 0 and abs(int(r_frames[si - 1]) - i) \
                        <= abs(int(r_frames[si]) - i):
                    si -= 1
                if si != r_cache["si"]:
                    r_cache = {"si": si,
                               "img": draw_radar_panel(radar, si, panel_w)}
                pimg = r_cache["img"]
                ph_, pw_ = pimg.shape[:2]
                mgn = out_w // 96
                crop = np.ascontiguousarray(crop)
                roi = crop[out_h - mgn - ph_:out_h - mgn,
                           out_w - mgn - pw_:out_w - mgn]
                cv2.addWeighted(pimg, radar_alpha, roi, 1.0 - radar_alpha,
                                0.0, dst=roi)
            if clock is not None:
                crop = np.ascontiguousarray(crop)
                _draw_clock(crop, clock_string(clock, i), out_w, out_h)
            enc.stdin.write(np.ascontiguousarray(crop).tobytes())
            done += 1
            if done % 90 == 0 and progress is not None:
                progress(done, end - start, done / (time.perf_counter() - t0))
            if done % 900 == 0:
                el = time.perf_counter() - t0
                log(f"[render] {done}/{len(cx)} @ {done/el:.2f}fps")
    finally:
        enc.stdin.close()
        enc.wait()
        cap.release()
    return done / max(time.perf_counter() - t0, 1e-9)


def analysis_summary(analysis_path, analysis, log=print):
    """분석 파생 요약 캐시 (.analysis.cache.json) — 로드 시간 절감.

    스팬({tid: [f0, f1, n]})·대표색·발 위치 중앙값은 분석에만 의존하는
    순수 파생물인데 GUI 가 열 때마다 검출 수십만 행을 재스캔했다 (46GB
    경기 실측 ~5s). 분석 파일 (size, mtime) 키로 캐시 — 갭필 등으로
    분석이 다시 써지면 자동 무효화. 실패는 조용히 재계산 (캐시는 항상
    버려도 되는 파일).
    """
    import json as _json
    from pathlib import Path as _P
    p = _P(str(analysis_path))
    cp = p.with_suffix(".cache.json")     # <pano>.analysis.cache.json
    try:
        st = p.stat()
        key = {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        key = None
    if key is not None and cp.exists():
        try:
            d = _json.loads(cp.read_text())
            if d.get("key") == key:
                return {"spans": {int(t): v
                                  for t, v in d["spans"].items()},
                        "colors": {int(t): tuple(v)
                                   for t, v in d["colors"].items()},
                        "foot_med": {int(t): tuple(v)
                                     for t, v in d["foot_med"].items()}}
        except (OSError, ValueError, KeyError) as e:
            log(f"[cache] 요약 캐시 무시: {e}")
    frames = analysis["frames"]
    spans: dict[int, list] = {}
    feet: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        f = int(frames[si])
        for pl in prow:
            if len(pl) >= 5 and pl[4] >= 0:
                t = int(pl[4])
                e = spans.get(t)
                if e is None:
                    spans[t] = [f, f, 1]
                else:
                    e[1] = f
                    e[2] += 1
                feet.setdefault(t, []).append((pl[0], pl[1] + pl[3] / 2.0))
    colors = tracklet_colors(analysis)
    foot_med = {t: (float(np.median([q[0] for q in v])),
                    float(np.median([q[1] for q in v])))
                for t, v in feet.items()}
    if key is not None:
        try:
            cp.write_text(_json.dumps(
                {"key": key,
                 "spans": {str(t): v for t, v in spans.items()},
                 "colors": {str(t): list(v) for t, v in colors.items()},
                 "foot_med": {str(t): list(v)
                              for t, v in foot_med.items()}}))
        except OSError as e:
            log(f"[cache] 요약 캐시 저장 실패: {e}")
    return {"spans": spans, "colors": colors, "foot_med": foot_med}


def link_ball_tracks_cached(analysis_path, analysis, ball_conf=0.25,
                            max_jump_per_frame=120.0, log=print):
    """link_ball_tracks 디스크 캐시 (.link.cache.npz) — GUI 열기 가속.

    트랙 연결은 분석에만 의존하는 순수 파생물인데 열 때마다 수십 초를
    재계산했다 ("트랙 로드가 오래 걸림"). 분석 파일 (size, mtime) +
    파라미터 키. numpy 구조라 npz 가 자연스럽고 빠르다.
    """
    from pathlib import Path as _P
    p = _P(str(analysis_path))
    cp = p.with_suffix(".link.cache.npz")   # <pano>.analysis.link.cache.npz
    try:
        st = p.stat()
        key = np.array([st.st_size, int(st.st_mtime),
                        int(ball_conf * 1000), int(max_jump_per_frame)])
    except OSError:
        key = None
    if key is not None and cp.exists():
        try:
            z = np.load(cp, allow_pickle=False)
            if np.array_equal(z["key"], key):
                nt = int(z["n_tracks"])
                tracks = [{"i": z[f"t{i}_i"],
                           "pts": z[f"t{i}_pts"],
                           "x": list(z[f"t{i}_pts"][:, 0]),
                           "y": list(z[f"t{i}_pts"][:, 1])}
                          for i in range(nt)]
                return {"idx": z["idx"], "tracks": tracks,
                        "fps": float(z["fps"]),
                        "p_cnt": z["p_cnt"], "p_tx": z["p_tx"],
                        "p_ty": z["p_ty"], "p_span": z["p_span"]}
        except Exception as e:  # noqa: BLE001 — 캐시는 버려도 되는 파일
            log(f"[cache] 링크 캐시 무시: {e}")
    linked = link_ball_tracks(analysis, ball_conf, max_jump_per_frame)
    if key is not None:
        try:
            arrs = {"key": key, "idx": np.asarray(linked["idx"]),
                    "fps": np.float64(linked["fps"]),
                    "n_tracks": np.int64(len(linked["tracks"])),
                    "p_cnt": np.asarray(linked["p_cnt"]),
                    "p_tx": np.asarray(linked["p_tx"]),
                    "p_ty": np.asarray(linked["p_ty"]),
                    "p_span": np.asarray(linked["p_span"])}
            for i, t in enumerate(linked["tracks"]):
                arrs[f"t{i}_i"] = np.asarray(t["i"])
                arrs[f"t{i}_pts"] = np.asarray(t["pts"])
            np.savez_compressed(cp, **arrs)
        except OSError as e:
            log(f"[cache] 링크 캐시 저장 실패: {e}")
    return linked


def _ffmpeg_gray_strip(video_path, t, w, rows, timeout=30.0):
    """빠른 시크(-ss)로 t초 프레임의 상단 rows행만 그레이스케일로 디코드.

    cap.set(POS_FRAMES) 는 성긴 GOP 원본(파노라마 ~8초 GOP)에서 키프레임
    부터 순차 디코드라 한 지점이 수 초 걸린다 — ffmpeg 는 -ss 를 입력
    앞에 둬 키프레임으로 바로 점프. 파노라마는 카메라가 거의 안 움직여
    상단 검은 경계가 프레임마다 같으니 -noaccurate_seek 로 정확한
    타임스탬프까지 앞으로 디코드하지 않고 **키프레임만** 뽑는다(GOP만큼
    앞 디코드 생략). 상단 스트립만 크롭(crop=iw:rows)해 전송·디코드량도
    줄인다. 실패(시간초과·바이트 부족)면 None."""
    import subprocess

    from .encoders import ffmpeg_bin
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t):.6f}",
           "-noaccurate_seek", "-i", str(video_path), "-frames:v", "1",
           "-vf", f"crop=iw:{rows}:0:0", "-f", "rawvideo",
           "-pix_fmt", "gray", "-"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    need = w * rows
    if len(p.stdout) < need:
        return None
    return np.frombuffer(p.stdout[:need], np.uint8).reshape(rows, w)


def measure_top_black(video_path, samples=5, dark=18, cover=0.90):
    """파노라마 상단 검은 스티칭 경계 높이 실측 (px).

    top_margin 의 근거는 이 경계뿐 — 고정 160px 대신 실측한다
    (--auto-el 산출물은 대개 0). 몇 프레임을 샘플해 행의 어두운 픽셀
    비율이 cover 이상인 최상단 연속 구간의 최대 높이 + 8px 여유.

    시크는 ffmpeg 빠른 시크(-ss) — 성긴 GOP 원본에서 cap.set 순차
    디코드가 지점당 수 초 걸리던 걸 회피 (실패 시 cv2 폴백).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = min(240, h) or 240
    worst = 0
    for k in range(samples):
        fno = int(total * (k + 0.5) / samples)
        strip = (_ffmpeg_gray_strip(video_path, fno / max(fps, 1e-9), w, rows)
                 if w > 0 else None)
        if strip is None:                     # 빠른 시크 실패 → cv2 순차 폴백
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frame = cap.read()
            if not ok:
                continue
            strip = cv2.cvtColor(frame[:rows], cv2.COLOR_BGR2GRAY)
        frac = (strip < dark).mean(axis=1)
        hh = 0
        while hh < len(frac) and frac[hh] >= cover:
            hh += 1
        worst = max(worst, hh)
    cap.release()
    return worst + (8 if worst else 0)
