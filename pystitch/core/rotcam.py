"""회전 카메라 (AX700) 필드 정합 수학 (P06-3).

고정 위치 삼각대 + 팬/틸트(+드문 줌) 카메라의 표준 원근 모델:
  x_cam = R·X_world + t,  K = [[f,0,cx],[0,f,cy],[0,0,1]]
world = 필드 좌표 (X=길이 방향, Y=폭 방향, Z=위, 원점 센터마크).

- 기준 프레임: 랜드마크로 f·자세·설치 위치 추정 (f 는 1D 탐색 + PnP).
- 프레임 간: H = K′·R_rel·K⁻¹ 분해로 회전과 줌 비율 동시 추정
  (삼각대 = 병진 0 → 순수 회전 호모그래피가 정확한 모델).
- 지면 투영: 광선-평면(z=0) 교차.

체인 추적의 누적 드리프트는 랜드마크/흰 라인 재정렬로 주기 보정
(Windows 단계) — 여기는 수학 코어와 합성 검증까지.
"""
from __future__ import annotations

import cv2
import numpy as np

from .field import CENTER_CIRCLE_R, PEN_DEPTH, PEN_HALF_W


def make_K(f, w, h):
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]])


def calibrate_reference(px_pts, field_pts, img_size, f_frac=(0.4, 3.0),
                        steps=40):
    """기준 프레임 랜드마크 → {f, K, R, t, cam_pos, rms_px}.

    px_pts: 이미지 픽셀 [(u, v)], field_pts: 필드 [(X, Y)] (z=0),
    img_size: (w, h). f 를 [f_frac]×폭 구간에서 탐색(거친 스캔 →
    황금분할)하며 평면 PnP 재투영 오차 최소화. 점 4개 이상.
    """
    w, h = img_size
    obj = np.array([[x, y, 0.0] for x, y in field_pts], np.float64)
    img = np.array(px_pts, np.float64)
    if len(obj) < 4:
        return None

    def solve(f):
        K = make_K(f, w, h)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.inf, None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return rms, (rvec, tvec)

    fs = np.linspace(f_frac[0] * w, f_frac[1] * w, steps)
    errs = [solve(f)[0] for f in fs]
    i = int(np.argmin(errs))
    lo = fs[max(i - 1, 0)]
    hi = fs[min(i + 1, len(fs) - 1)]
    g = (np.sqrt(5) - 1) / 2                     # 황금분할
    a, b = lo, hi
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if solve(c)[0] < solve(d)[0]:
            b = d
        else:
            a = c
    f = 0.5 * (a + b)
    rms, sol = solve(f)
    if sol is None or not np.isfinite(rms):
        return None
    rvec, tvec = sol
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    cam_pos = (-R.T @ t)
    return {"f": float(f), "K": make_K(f, w, h), "R": R, "t": t,
            "cam_pos": cam_pos, "rms_px": rms,
            "img_size": (int(w), int(h))}


def decompose_H(H, K, ratio_range=(0.7, 1.4), steps=36):
    """프레임 간 호모그래피 → (R_rel, f_ratio, 잔차).

    H ≈ K′·R·K⁻¹ (K′ = f×ratio). ratio 를 1D 탐색하며
    G = K′⁻¹HK 의 정규직교 잔차를 최소화, R 은 SVD 최근접 회전.
    """
    H = np.asarray(H, np.float64)
    Kinv_p = None

    def ortho_res(r):
        K2 = K.copy()
        K2[0, 0] *= r
        K2[1, 1] *= r
        G = np.linalg.inv(K2) @ H @ K
        s = np.linalg.svd(G, compute_uv=False)[1]   # 중간 특이값 = 스케일
        Gn = G / max(s, 1e-12)
        return float(np.linalg.norm(Gn @ Gn.T - np.eye(3))), Gn

    rs = np.linspace(ratio_range[0], ratio_range[1], steps)
    errs = [ortho_res(r)[0] for r in rs]
    i = int(np.argmin(errs))
    a, b = rs[max(i - 1, 0)], rs[min(i + 1, len(rs) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(35):
        c, d = b - g * (b - a), a + g * (b - a)
        if ortho_res(c)[0] < ortho_res(d)[0]:
            b = d
        else:
            a = c
    ratio = 0.5 * (a + b)
    res, Gn = ortho_res(ratio)
    U, _s, Vt = np.linalg.svd(Gn)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, float(ratio), res


def pixel_to_field(K, R, cam_pos, pts_px):
    """픽셀 → 지면(z=0) 필드 좌표 (N,2) — 수평선 위는 NaN."""
    p = np.asarray(pts_px, np.float64).reshape(-1, 2)
    ones = np.ones((len(p), 1))
    rays = (np.linalg.inv(K) @ np.hstack([p, ones]).T).T   # 카메라 좌표
    d = rays @ R                                            # = R.T @ ray
    out = np.full((len(p), 2), np.nan)
    m = d[:, 2] < -1e-9                                     # 지면 방향만
    lam = -cam_pos[2] / d[m, 2]
    out[m] = cam_pos[None, :2] + lam[:, None] * d[m, :2]
    return out


def field_to_pixel(K, R, t, pts_field):
    """필드 (X, Y) → 픽셀 (N,2) — 카메라 뒤는 NaN."""
    P = np.asarray(pts_field, np.float64).reshape(-1, 2)
    X = np.hstack([P, np.zeros((len(P), 1))])
    xc = (R @ X.T).T + t[None]
    out = np.full((len(P), 2), np.nan)
    m = xc[:, 2] > 1e-9
    uv = (K @ xc[m].T).T
    out[m] = uv[:, :2] / uv[:, 2:3]
    return out


def anchor_rotation(cam_pos, px_pts, field_pts, f_init, img_size,
                    f_span=0.15):
    """현재 프레임 랜드마크 → (R, f) 절대 재정렬 (설치 위치 고정).

    체인 추적(track_step)의 누적 드리프트 제거용 — world 광선
    (cam_pos→랜드마크)과 카메라 광선(K⁻¹p)을 Kabsch(직교 프로크루스테스)
    로 정렬, f 는 f_init±span 황금분할. 랜드마크 3개 이상.
    반환: {"R", "f", "K", "res_deg"} 또는 None.
    """
    w, h = img_size
    P = np.asarray(field_pts, np.float64).reshape(-1, 2)
    px = np.asarray(px_pts, np.float64).reshape(-1, 2)
    if len(P) < 3:
        return None
    Xw = np.hstack([P, np.zeros((len(P), 1))]) - np.asarray(cam_pos)[None]
    rw = Xw / np.linalg.norm(Xw, axis=1, keepdims=True)

    def solve(f):
        K = make_K(f, w, h)
        rc = (np.linalg.inv(K)
              @ np.hstack([px, np.ones((len(px), 1))]).T).T
        rc = rc / np.linalg.norm(rc, axis=1, keepdims=True)
        M = rc.T @ rw
        U, _s, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        cosang = np.clip(np.sum(rc * (rw @ R.T), axis=1), -1.0, 1.0)
        return float(np.rad2deg(np.mean(np.arccos(cosang)))), R

    a, b = f_init * (1 - f_span), f_init * (1 + f_span)
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if solve(c)[0] < solve(d)[0]:
            b = d
        else:
            a = c
    f = 0.5 * (a + b)
    res, R = solve(f)
    return {"R": R, "f": float(f), "K": make_K(f, w, h),
            "res_deg": res}


def match_frames(img_a, img_b, nfeatures=3000, ratio=0.75):
    """두 프레임 SIFT 매칭 → (N,2), (N,2) — 회전 추적 입력."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    ka, da = sift.detectAndCompute(img_a, None)
    kb, db = sift.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for m, n in bf.knnMatch(da, db, k=2):
        if m.distance < ratio * n.distance:
            pairs.append((ka[m.queryIdx].pt, kb[m.trainIdx].pt))
    if not pairs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return a, b


def track_step(prev_state, pts_prev, pts_cur, min_matches=25,
               max_ortho_res=0.08):
    """이전 상태 {R, f} + 매칭 → 새 상태 (실패 시 None — 품질 플래그).

    H 는 RANSAC 으로 추정, 분해 잔차가 크면(가림·블러·모델 위반) 기각.
    """
    if len(pts_prev) < min_matches:
        return None
    H, mask = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3.0)
    if H is None or mask is None or mask.sum() < min_matches:
        return None
    K = prev_state["K"]
    R_rel, ratio, res = decompose_H(H, K)
    if res > max_ortho_res:
        return None
    f2 = prev_state["f"] * ratio
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = f2
    return {"f": float(f2), "K": K2,
            "R": R_rel @ prev_state["R"],
            "ortho_res": res, "n_inliers": int(mask.sum())}


# ---------------------------------------------------------------- 자동 앵커
# 필드 지형지물(흰 라인·센터서클·박스)로 (R, f) 를 자동 재정렬한다.
# 체인 추적이 근사 자세를 항상 제공하므로 전역 인식이 아니라 국소 정렬:
# 근사 자세로 템플릿 라인을 투영 → 각 투영점의 법선 방향으로 흰 선
# 중심을 탐색(화이트니스 가중 중심) → (픽셀, 필드) 대응으로
# anchor_rotation 반복. 수동 랜드마크의 자동화 (P06-3).

def template_polylines(length=105.0, width=68.0, step=0.5):
    """필드 마킹 샘플 (M,2) + 단위 접선 (M,2) + 패밀리 (M,).

    패밀리 인덱스는 _marking_lines 의 직선 순서와 일치, 센터서클 = 9.
    라벨을 샘플에 붙여 두는 이유: 탐색으로 찾은 흰 선 중심의 소속
    직선은 "시드 샘플의 패밀리" 가 필드 재스냅보다 훨씬 강건하다
    (원측은 역투영 증폭으로 수 m 흘러 이웃 라인에 오스냅된다 — 실측
    23% 오라벨).
    """
    hl, hw = length / 2.0, width / 2.0
    polys = []

    def seg(a, b, fam):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step))
        t = np.linspace(0.0, 1.0, n)[:, None]
        polys.append((a[None] + t * (b - a)[None],
                      np.tile((b - a) / np.linalg.norm(b - a), (n, 1)),
                      np.full(n, fam, np.int64)))

    seg((-hl, -hw), (hl, -hw), 0)         # 근측 터치라인 (y=-hw)
    seg((-hl, hw), (hl, hw), 1)           # 원측 터치라인 (y=+hw)
    seg((-hl, -hw), (-hl, hw), 2)         # 골라인 L
    seg((hl, -hw), (hl, hw), 3)           # 골라인 R
    seg((0.0, -hw), (0.0, hw), 4)         # 하프라인
    for sx, vfam in ((-1.0, 5), (1.0, 6)):        # 페널티박스 3변
        seg((sx * (hl - PEN_DEPTH), -PEN_HALF_W),
            (sx * (hl - PEN_DEPTH), PEN_HALF_W), vfam)
        seg((sx * hl, -PEN_HALF_W), (sx * (hl - PEN_DEPTH), -PEN_HALF_W), 7)
        seg((sx * hl, PEN_HALF_W), (sx * (hl - PEN_DEPTH), PEN_HALF_W), 8)
    th = np.linspace(0.0, 2 * np.pi,
                     max(8, int(2 * np.pi * CENTER_CIRCLE_R / step)))
    circ = np.stack([CENTER_CIRCLE_R * np.cos(th),
                     CENTER_CIRCLE_R * np.sin(th)], axis=1)
    tan = np.stack([-np.sin(th), np.cos(th)], axis=1)
    polys.append((circ, tan, np.full(len(th), 9, np.int64)))
    pts = np.vstack([p for p, _t, _f in polys])
    tans = np.vstack([t for _p, t, _f in polys])
    fams = np.concatenate([f for _p, _t, f in polys])
    return pts, tans, fams


def whiteness(frame_bgr):
    """흰 선 응답 맵 [0,1] — 밝고 무채색 (V·(1-S), field 검출과 동일 발상)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    return v * (1.0 - s)


def _line_centers(wmap, proj, normals, rad=14, min_contrast=0.10):
    """투영점별 법선 1D 탐색 → 흰 선 중심 픽셀 (없으면 NaN)."""
    h, w = wmap.shape
    off = np.arange(-rad, rad + 1, dtype=np.float64)
    out = np.full_like(proj, np.nan)
    for i, ((u, v), nrm) in enumerate(zip(proj, normals)):
        if not np.isfinite(u):
            continue
        us = u + off * nrm[0]
        vs = v + off * nrm[1]
        m = (us >= 0) & (us < w - 1) & (vs >= 0) & (vs < h - 1)
        if m.sum() < len(off) * 0.7:
            continue
        prof = wmap[vs[m].astype(int), us[m].astype(int)]
        base = np.percentile(prof, 20)
        peak = prof.max()
        if peak - base < min_contrast:
            continue                       # 대비 없음 (잔디/가림)
        wgt = np.clip(prof - base, 0.0, None) ** 2
        j = np.sum(wgt * np.arange(len(prof))) / np.sum(wgt)
        out[i] = (us[m][0] + (us[m][-1] - us[m][0]) * j / (len(prof) - 1),
                  vs[m][0] + (vs[m][-1] - vs[m][0]) * j / (len(prof) - 1))
    return out


def _snap_to_marking(pts, length=105.0, width=68.0):
    """필드 점 (N,2) → (스냅 점 (N,2), 거리 (N,), 패밀리 인덱스 (N,)).

    직사각형 경계·하프라인·페널티박스 3변×2·센터서클. 패밀리는
    _marking_lines 의 직선 인덱스와 일치 (센터서클 = len(lines)).
    """
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    hl, hw = length / 2.0, width / 2.0
    cands = []

    def add(x, y):
        cands.append(np.stack([x, y], axis=1))

    X, Y = p[:, 0], p[:, 1]
    add(np.clip(X, -hl, hl), np.full_like(Y, -hw))       # 0 근측 터치라인
    add(np.clip(X, -hl, hl), np.full_like(Y, hw))        # 1 원측 터치라인
    add(np.full_like(X, -hl), np.clip(Y, -hw, hw))       # 2 골라인 L
    add(np.full_like(X, hl), np.clip(Y, -hw, hw))        # 3 골라인 R
    add(np.full_like(X, 0.0), np.clip(Y, -hw, hw))       # 4 하프라인
    bl, br = -(hl - PEN_DEPTH), hl - PEN_DEPTH
    add(np.full_like(X, bl), np.clip(Y, -PEN_HALF_W, PEN_HALF_W))  # 5 박스 L
    add(np.full_like(X, br), np.clip(Y, -PEN_HALF_W, PEN_HALF_W))  # 6 박스 R
    gx = np.where(X < 0, np.clip(X, -hl, bl), np.clip(X, br, hl))
    add(gx, np.full_like(Y, -PEN_HALF_W))                # 7 박스 근측 변
    add(gx, np.full_like(Y, PEN_HALF_W))                 # 8 박스 원측 변
    r = np.linalg.norm(p, axis=1)
    safe = np.where(r > 1e-9, r, 1.0)
    add(X / safe * CENTER_CIRCLE_R, Y / safe * CENTER_CIRCLE_R)  # 9 센터서클
    C = np.stack(cands, axis=0)                          # (패밀리, N, 2)
    d = np.linalg.norm(C - p[None], axis=2)
    j = np.argmin(d, axis=0)
    idx = np.arange(len(p))
    return C[j, idx], d[j, idx], j


def _marking_lines(length=105.0, width=68.0):
    """_snap_to_marking 패밀리 0..8 의 동차 직선 (a, b, c): ax+by+c=0."""
    hl, hw = length / 2.0, width / 2.0
    b = hl - PEN_DEPTH
    return np.array([
        [0.0, 1.0, hw],       # y = -hw
        [0.0, 1.0, -hw],      # y = +hw
        [1.0, 0.0, hl],       # x = -hl
        [1.0, 0.0, -hl],      # x = +hl
        [1.0, 0.0, 0.0],      # x = 0
        [1.0, 0.0, b],        # x = -(hl-깊이)
        [1.0, 0.0, -b],       # x = +(hl-깊이)
        [0.0, 1.0, PEN_HALF_W],
        [0.0, 1.0, -PEN_HALF_W],
    ])


def _refine_pose_p2l(px_pts, fam, state, cam, img_size, length=105.0,
                     width=68.0, iters=12):
    """point-to-line 가우스-뉴턴: (rvec, f) 4 파라미터 직접 정밀화.

    미지수는 R(3)+f(1)뿐 (cam_pos 고정) — 호모그래피 8-DOF DLT 는
    과잉이라 근측 라인 3~4 패밀리로도 미정이 된다. 잔차 = 픽셀을
    필드로 역투영한 점의 소속 라인까지 부호 거리(직선) / 반경
    편차(센터서클), 1/지면거리 가중(≈각도 오차) + Huber.
    """
    lines = _marking_lines(length, width)
    px = np.asarray(px_pts, np.float64)
    fam = np.asarray(fam)
    w, h = img_size
    cam = np.asarray(cam, np.float64)
    rvec0, _ = cv2.Rodrigues(np.asarray(state["R"], np.float64))
    th = np.concatenate([rvec0.ravel(), [state["f"]]])

    def residuals(t):
        K = make_K(t[3], w, h)
        R, _ = cv2.Rodrigues(t[:3])
        fp = pixel_to_field(K, R, cam, px)
        r = np.full(len(px), np.nan)
        straight = fam < len(lines)
        ls = lines[fam[straight]]
        n = np.linalg.norm(ls[:, :2], axis=1)
        r[straight] = (np.sum(ls[:, :2] * fp[straight], axis=1)
                       + ls[:, 2]) / n
        circ = fam == len(lines)
        r[circ] = np.linalg.norm(fp[circ], axis=1) - CENTER_CIRCLE_R
        gd = np.linalg.norm(fp - cam[None, :2], axis=1)
        wgt = 1.0 / np.clip(gd, 8.0, None)
        r = r * wgt
        r[~np.isfinite(r)] = 0.0
        # Huber (0.02 ≈ 20m 거리에서 0.4m)
        c = 0.02
        a = np.abs(r)
        with np.errstate(invalid="ignore"):
            r = np.where(a <= c, r, np.sign(r) * np.sqrt(c * (2 * a - c)))
        # f 사전 제약: 제한된 시야에선 (f, R) 준퇴화 — 검출 노이즈만으로
        # f 가 폭주한다 (실측 1400→1766). 체인의 줌비 추적은 신뢰 가능
        # (043: 80스텝 f<2%) → 초기 f 에 σ≈3% 앵커. 데이터가 관측
        # 가능한 방향은 그대로 데이터가 지배한다.
        return np.append(r, (t[3] / state["f"] - 1.0) * 0.2)

    eps = np.array([1e-4, 1e-4, 1e-4, 0.5])
    lam = 1e-3
    r0 = residuals(th)
    for _ in range(iters):
        J = np.stack([(residuals(th + e * np.eye(4)[i]) - r0) / e
                      for i, e in enumerate(eps)], axis=1)
        A = J.T @ J + lam * np.eye(4)
        g = J.T @ r0
        try:
            d = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            return None
        r1 = residuals(th + d)
        if np.sum(r1 ** 2) < np.sum(r0 ** 2):
            th, r0 = th + d, r1
            lam = max(lam * 0.5, 1e-6)
            if np.linalg.norm(d[:3]) < 1e-6:
                break
        else:
            lam *= 4.0
    R, _ = cv2.Rodrigues(th[:3])
    rms = float(np.sqrt(np.mean(r0 ** 2)))
    return {"R": R, "f": float(th[3]), "K": make_K(th[3], w, h),
            "res_w": rms}


def auto_anchor(frame_bgr, state, cam_pos, length=105.0, width=68.0,
                rads=(16, 10, 8), min_pts=40, f_span=0.06):
    """근사 자세 → 필드 마킹 자동 정렬로 (R, f) 재추정.

    state = {"R", "f", "K"}. 반환 {"R", "f", "K", "res_deg", "n_pts"}
    또는 None (대응 부족 — 급팬/가림 구간은 호출부가 다음 기회에).

    용도는 체인 추적의 **주기 드리프트 보정** (앵커 간 ≤0.5° 급) —
    큰 섭동의 전역 복원은 단일 뷰 라인 기하로는 불량조건이라 임무가
    아니다 (기준 캘리브레이션이 담당). 탐색 반경 rads 축소 스케줄,
    첫 반경 16px ≈ f1400 에서 ~0.65°. 반경이 작아야 원측의 라인
    밀집(터치라인↔박스 ~21px)에서 이웃 라인을 줍지 않는다.
    """
    wmap = whiteness(frame_bgr)
    h, w = wmap.shape
    fld, tans, fams = template_polylines(length, width)
    cam = np.asarray(cam_pos, np.float64)
    cur = {"R": np.asarray(state["R"], float), "f": float(state["f"])}
    res = None
    for rad in rads:
        K = make_K(cur["f"], w, h)
        t = -cur["R"] @ cam
        proj = field_to_pixel(K, cur["R"], t, fld)
        # 접선의 이미지 방향 → 법선 (탐색 방향)
        proj2 = field_to_pixel(K, cur["R"], t, fld + 0.5 * tans)
        d = proj2 - proj
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            nrm = np.where(ln > 1e-9, nrm / ln, np.nan)
        # 수평선 부근(80m+)만 제외 — 원측 터치라인(~79m)은 수평 정보의
        # 주공급원이라 유지 (반경이 작아 이웃 라인 오염 없음)
        gd = np.linalg.norm(fld - cam[None, :2], axis=1)
        inside = (np.isfinite(proj).all(1) & np.isfinite(nrm).all(1)
                  & (gd < 80.0)
                  & (proj[:, 0] >= rad) & (proj[:, 0] < w - rad)
                  & (proj[:, 1] >= rad) & (proj[:, 1] < h - rad))
        px = _line_centers(wmap, np.where(inside[:, None], proj, np.nan),
                           np.where(inside[:, None], nrm, np.nan), rad=rad)
        m = np.isfinite(px).all(1)
        if m.sum() < min_pts:
            return None
        # point-to-line 가우스-뉴턴 — 라벨은 탐색 시드 샘플의 패밀리
        # (필드 재스냅 라벨은 원측 역투영 증폭으로 오라벨 — 실측 23%).
        # 점 대 점 ICP 는 접선 성분이 보정을 되끌어 정체한다 (실측).
        got = _refine_pose_p2l(px[m], fams[m],
                               {"R": cur["R"], "f": cur["f"]}, cam,
                               (w, h), length, width)
        if got is None:
            return None
        cur = {"R": got["R"], "f": got["f"]}
        res = {"res_w": got["res_w"], "n_pts": int(m.sum())}
    return {"R": cur["R"], "f": cur["f"], "K": make_K(cur["f"], w, h),
            **res}


def chain_homography(video_path, f0, f1, det_w=1920, every_s=0.5):
    """프레임 f0 → f1 픽셀 이송 호모그래피 (원본 좌표계).

    순수 회전 카메라라 H 가 픽셀을 시차 없이 정확히 옮긴다 — 다른
    프레임에서 찍은 랜드마크를 기준 프레임으로 합치는 다중 프레임
    캘리브레이션의 기반 (f/K 불필요, H 합성만). every_s 간격 중간
    프레임을 거쳐 누적. 실패(매칭 부족) 시 None.
    """
    if f0 == f1:
        return np.eye(3)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(every_s * fps)) * (1 if f1 > f0 else -1)

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            return None
        sc = 1.0
        if det_w and fr.shape[1] > det_w:
            sc = det_w / fr.shape[1]
            fr = cv2.resize(fr, (det_w, int(fr.shape[0] * sc)),
                            interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), sc

    got = grab(f0)
    if got is None:
        cap.release()
        return None
    prev, scale = got
    H = np.eye(3)
    idx = f0
    while idx != f1:
        nxt = f1 if abs(f1 - idx) <= abs(step) else idx + step
        got = grab(nxt)
        if got is None:
            cap.release()
            return None
        cur, _ = got
        pa, pb = match_frames(prev, cur)
        if len(pa) < 25:
            cap.release()
            return None
        Hi, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
        if Hi is None or mask is None or mask.sum() < 25:
            cap.release()
            return None
        H = Hi @ H
        prev, idx = cur, nxt
    cap.release()
    # det_w 좌표계 H → 원본 좌표계: S⁻¹ H S
    S = np.diag([scale, scale, 1.0])
    return np.linalg.inv(S) @ H @ S


def transfer_points(H, pts):
    """H 로 픽셀 점 (N,2) 이송."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = (H @ np.hstack([p, np.ones((len(p), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


# ---------------------------------------------------------------- 궤적 번들
def bundle_calibrate(rel_rots, landmarks, img_size, cam_init,
                     f_init_fracs=(0.5, 0.7, 0.9, 1.1), iters=100):
    """궤적 상대회전 + 다프레임 랜드마크 → (cam_pos, f, R0) 번들 (P06-3).

    순수 팬은 f 자가캘리브가 퇴화(devlog 060)하므로 f 는 랜드마크로
    푼다 — 단, 여러 프레임 랜드마크를 궤적 회전(rel_rots)으로 한
    추정에 모아 깊이 다양성으로 f 를 구속한다. 상대회전은 고정,
    미지수는 cam_pos(3)+f(1)+절대회전 rvec(3)=7.

    rel_rots: {frame: R_rel(기준→그 프레임 world→cam 상대회전)}.
    landmarks: [(px[2], field[2], frame)]. 반환 {cam_pos,f,K,R0,rms_px,n}
    또는 None.
    """
    w, h = img_size
    pts = [(np.asarray(p, float), np.asarray(X, float), fr)
           for p, X, fr in landmarks if rel_rots.get(fr) is not None]
    if len(pts) < 4:
        return None
    cam0 = np.asarray(cam_init, float)
    # 초기 절대회전: 센터마크 응시
    fwd = -cam0 / (np.linalg.norm(cam0) + 1e-9)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right) + 1e-9
    down = np.cross(fwd, right); down /= np.linalg.norm(down) + 1e-9
    rv0, _ = cv2.Rodrigues(np.stack([right, down, fwd]))

    def resid(prm):
        cam, f = prm[:3], prm[3]
        R0, _ = cv2.Rodrigues(prm[4:7])
        K = make_K(f, w, h)
        out = []
        for p, X, fr in pts:
            Ri = rel_rots[fr] @ R0
            xc = Ri @ (np.array([X[0], X[1], 0.0]) - cam)
            if xc[2] <= 1e-6:
                out += [1e3, 1e3]; continue
            u = K @ xc
            out += [u[0] / u[2] - p[0], u[1] / u[2] - p[1]]
        return np.array(out)

    best = None
    eps = np.array([0.1, 0.1, 0.1, 10.0, 1e-3, 1e-3, 1e-3])
    for frac in f_init_fracs:
        prm = np.concatenate([cam0, [frac * w], rv0.ravel()])
        r = resid(prm); lam = 1e-3
        for _ in range(iters):
            J = np.stack([(resid(prm + eps[j] * np.eye(7)[j]) - r) / eps[j]
                          for j in range(7)], axis=1)
            try:
                dp = np.linalg.solve(J.T @ J + lam * np.eye(7), -(J.T @ r))
            except np.linalg.LinAlgError:
                break
            r2 = resid(prm + dp)
            if np.sum(r2 ** 2) < np.sum(r ** 2):
                prm, r, lam = prm + dp, r2, max(lam * 0.5, 1e-7)
                if np.linalg.norm(dp[4:7]) < 1e-7 and abs(dp[3]) < 0.1:
                    break
            else:
                lam *= 4
        rms = float(np.sqrt(np.mean(r ** 2)))
        if best is None or rms < best[1]:
            best = (prm, rms)
    prm, rms = best
    R0, _ = cv2.Rodrigues(prm[4:7])
    return {"cam_pos": prm[:3], "f": float(prm[3]),
            "K": make_K(prm[3], w, h), "R0": R0,
            "rms_px": rms, "n": len(pts)}


# ------------------------------------------------------------ 회전 평균
def rotation_average(n, edges, R_init, iters=100, tol=1e-7):
    """전역 회전 평균 (chordal L2, 반복) — 체인 드리프트를 전역 분산.

    edges: [(i, j, R_ij)] — 규약 R_i ≈ R_ij @ R_j (world→cam 상대회전).
    루프 클로저 엣지가 있어야 드리프트가 실제로 분산된다 (체인만이면
    변화 없음). R_init: 초기 절대회전 리스트(체인/스패닝트리). 반환:
    절대회전 리스트 (일부 노드는 엣지 없으면 R_init 유지).
    """
    R = [np.array(r, float) if r is not None else None for r in R_init]
    nbr = [[] for _ in range(n)]
    for i, j, Rij in edges:
        nbr[i].append((j, Rij, False))         # R_i 추정 = Rij @ R_j
        nbr[j].append((i, Rij, True))          # R_j 추정 = Rij^T @ R_i
    for _ in range(iters):
        maxupd = 0.0
        for i in range(n):
            if R[i] is None or not nbr[i]:
                continue
            ests = []
            for j, Rij, inv in nbr[i]:
                if R[j] is None:
                    continue
                ests.append((Rij.T @ R[j]) if inv else (Rij @ R[j]))
            if not ests:
                continue
            M = sum(ests) / len(ests)
            U, _s, Vt = np.linalg.svd(M)
            d = np.sign(np.linalg.det(U @ Vt))
            Rn = U @ np.diag([1.0, 1.0, d]) @ Vt
            ang = np.arccos(np.clip((np.trace(Rn @ R[i].T) - 1) / 2, -1, 1))
            maxupd = max(maxupd, ang)
            R[i] = Rn
        if maxupd < tol:
            break
    return R
