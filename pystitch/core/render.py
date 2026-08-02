"""파노라마 렌더링: remap 캐싱, 게인 보정, 심/페더 블렌딩."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# remap 맵을 CV_16SC2(고정소수점)로 둘지 float32 로 둘지.
#
# 통념은 고정소수점이 빠르다는 것이고 Linux 빌드에선 실제로 2% 빠르다.
# 그런데 **Windows 용 opencv-python 휠에서는 정반대로 1.54배 느리다**
# (5976×2306 실측: float32 15.1ms vs CV_16SC2 23.2ms, devlog 097).
# 같은 5.0.0 이지만 GCC/pthreads 빌드와 MSVC/ConcRT 빌드의 차이다.
# 실제 렌더로도 Windows 에서 +11% / CPU −20%.
#
# 정확도는 float32 쪽이 낫다 — 고정소수점은 좌표를 1/32 픽셀로 양자화한다
# (두 경로 출력 차이 PSNR 59.7dB, 최대차 4).
#
# 다른 OpenCV 빌드(4.13 PyPI, conda-forge)도 재봤지만 현행 5.0.0 +
# float32 조합을 못 이겼다. PYSTITCH_FIXED_MAPS=1/0 으로 강제 가능.
_env = os.environ.get("PYSTITCH_FIXED_MAPS")
USE_FIXED_MAPS = (_env == "1") if _env is not None else sys.platform != "win32"

from .geometry import build_cylindrical_maps
from .lens import LensProfile
from .perspective import build_perspective_maps


def gain_lut(gain) -> np.ndarray:
    """채널 게인 → 256엔트리 LUT (1,256,3).

    `cv2.multiply(img, gain)` 는 **단일 스레드**라 render 의 최대 항목이었다
    (5976×2306 실측: remap 두 개 합보다 2.4배 오래 걸림, devlog 098).
    같은 연산을 LUT 로 하면 병렬화돼 5~6배 빠르고, 결과는 **비트 동일**하다
    (게인 조합 1000건 검증 — 양쪽 다 round-half-even + 포화).
    """
    g = np.asarray(gain[:3], np.float64)
    tbl = np.clip(np.rint(np.arange(256, dtype=np.float64)[:, None] * g[None, :]),
                  0, 255)
    return tbl.astype(np.uint8).reshape(1, 256, 3)


def compute_gains(warped_l, warped_r, mask_l, mask_r):
    """겹침 영역 채널 평균을 기하평균으로 정렬하는 채널별 게인."""
    overlap = (mask_l > 0) & (mask_r > 0)
    if overlap.sum() < 1000:
        return np.ones(3), np.ones(3)
    mean_l = np.maximum(warped_l[overlap].reshape(-1, 3).mean(axis=0), 1e-3)
    mean_r = np.maximum(warped_r[overlap].reshape(-1, 3).mean(axis=0), 1e-3)
    target = np.sqrt(mean_l * mean_r)
    return target / mean_l, target / mean_r


def feather_weights(mask_l, mask_r, feather_px=120):
    """마스크 경계 거리 기반 페더 가중치 (자동 보정용 저해상도 렌더에 사용)."""
    dist_l = np.minimum(cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    dist_r = np.minimum(cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    wsum = dist_l + dist_r
    wsum[wsum == 0] = 1
    return (dist_l / wsum).astype(np.float32)


def seam_weights(mask_l, mask_r, yaw0, yaw1, seam_yaw, feather_px=40):
    """하프라인 수직 심 가중치: 심 좌측은 L, 우측은 R 카메라만 사용.

    심 주변 feather_px 폭만 선형 블렌딩. 한쪽 마스크가 없는 곳은 다른 쪽이 채움.
    """
    h, w = mask_l.shape
    yaws = np.linspace(yaw0, yaw1, w)
    half = feather_px / 2 * (yaw1 - yaw0) / w  # 픽셀 → 라디안
    ramp = np.clip((seam_yaw + half - yaws) / (2 * half), 0.0, 1.0).astype(np.float32)
    w_l = np.tile(ramp, (h, 1)) * (mask_l > 0)
    w_r = np.tile(1.0 - ramp, (h, 1)) * (mask_r > 0)
    wsum = w_l + w_r
    wsum[wsum == 0] = 1
    return (w_l / wsum).astype(np.float32)


def render_pano(imgs, Rs, lens: LensProfile, out_w, out_h, yaw0, yaw1, el0, el1,
                feather_px=120, seam_yaw=None):
    """1회용 파노라마 렌더 (자동 보정 저해상도 패스, 정지 프레임용)."""
    warped, masks = [], []
    for img, R_cam in zip(imgs, Rs):
        mx, my = build_cylindrical_maps(lens, R_cam, out_w, out_h, yaw0, yaw1, el0, el1)
        warped.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0))
        masks.append(cv2.remap(np.ones(img.shape[:2], np.uint8) * 255, mx, my,
                               cv2.INTER_NEAREST, borderValue=0))
    g_l, g_r = compute_gains(warped[0], warped[1], masks[0], masks[1])
    if seam_yaw is None:
        w_l = feather_weights(masks[0], masks[1], feather_px)[..., None]
    else:
        w_l = seam_weights(masks[0], masks[1], yaw0, yaw1, seam_yaw)[..., None]
    pano = (warped[0].astype(np.float32) * (w_l * g_l)
            + warped[1].astype(np.float32) * ((1 - w_l) * g_r))
    return np.clip(pano, 0, 255).astype(np.uint8)


class Renderer:
    """remap 테이블·가중치를 캐싱한 프레임 렌더러 (미리보기·내보내기 공용).

    분할 렌더링: 심 좌측은 L, 우측은 R 만 사용하므로 각 카메라를
    자기 절반 + 심 밴드까지만 워핑한다 (remap/게인/블렌딩 비용 ≈ 절반).
    scale 로 출력 해상도를 줄일 수 있다 (미리보기용).

    persp_k/persp_m 지정 시 원근비 조절 맵을 원통 맵에 합성해
    소스에서 한 번의 보간으로 보정된 파노라마를 렌더한다. 심은 중앙
    열이라 키스톤 후에도 수직으로 유지되고, 마스크·심 가중치·게인·
    refine_seam 은 모두 합성 이후의 출력 공간에서 동작한다.
    수평선은 elevation=0 행을 정확히 계산해 사용한다.
    """

    def __init__(self, lens: LensProfile, R_wl, R_wr, yaw0, yaw1, el0, el1,
                 scale=1.0, feather_px=40, persp_k=0.0, persp_m=1.0,
                 persp_widen=1.0):
        self.lens = lens
        f = lens.focal * scale
        self.scale = scale
        cyl_w = int((yaw1 - yaw0) * f) & ~1
        self.out_h = int((np.tan(el1) - np.tan(el0)) * f) & ~1
        self.el0, self.el1 = el0, el1
        self.persp_k, self.persp_m = persp_k, persp_m
        if persp_widen < 1.0:
            raise ValueError(f"persp_widen 은 1 이상이어야 함: {persp_widen}")
        if persp_widen > 1.0 and persp_m <= 1.0:
            # 키스톤 없이 넓히면 좌우에 검은 띠만 붙는다 — 설정 실수를 즉시 알린다
            raise ValueError("persp_widen 은 persp_m > 1 일 때만 의미가 있음 "
                             f"(m={persp_m}, widen={persp_widen})")
        self.persp_widen = persp_widen
        persp = None
        if persp_k > 0.0 or persp_m > 1.0:
            # elevation=0 행: t = linspace(tan(el1), tan(el0)) 이 0이 되는 위치
            t1, t0 = np.tan(el1), np.tan(el0)
            horizon = (self.out_h - 1) * t1 / (t1 - t0)
            self.out_w = int(cyl_w * persp_widen) & ~1
            persp = build_perspective_maps(self.out_w, self.out_h, horizon,
                                           persp_k, persp_m, src_w=cyl_w)
        else:
            self.out_w = cyl_w
        self._float_maps, masks = [], []
        src_ones = np.ones((lens.height, lens.width), np.uint8) * 255
        for R_cam in (R_wl, R_wr):
            mx, my = build_cylindrical_maps(lens, R_cam, cyl_w, self.out_h,
                                            yaw0, yaw1, el0, el1)
            if persp is not None:
                # 넓힌 캔버스의 빈 쐐기는 소스 밖을 가리켜야 한다 — REPLICATE
                # 로 두면 가장자리 화소가 쐐기로 번진다 (widen>1 에서만 발생).
                mx = cv2.remap(mx, persp[0], persp[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
                my = cv2.remap(my, persp[0], persp[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            self._float_maps.append((mx, my))
            masks.append(cv2.remap(src_ones, mx, my, cv2.INTER_NEAREST, borderValue=0))
        self._masks = masks

        # 심 밴드 경계: [x0, x1) 만 블렌딩, 바깥은 단일 카메라 복사
        # (키스톤 시 상단에서 심 주변 소스가 가로로 늘어나므로 밴드도 m배 확장)
        seam_feather = max(2, int(feather_px * scale))
        band_half = int(seam_feather * max(1.0, persp_m)) // 2 + 2
        xc = self.out_w // 2
        self._x0 = max(0, xc - band_half)
        self._x1 = min(self.out_w, xc + band_half)
        w_l_full = seam_weights(masks[0], masks[1], yaw0, yaw1,
                                (yaw0 + yaw1) / 2, seam_feather)
        self.w_l_band = np.ascontiguousarray(w_l_full[:, self._x0 : self._x1])
        self.w_r_band = (1.0 - self.w_l_band).astype(np.float32)

        self._rebuild_cropped_maps()
        self.gain_l = (1.0, 1.0, 1.0, 1.0)
        self.gain_r = (1.0, 1.0, 1.0, 1.0)

    def _rebuild_cropped_maps(self):
        """분할 렌더용 크롭 맵: L 은 [0, x1), R 은 [x0, out_w)."""
        (mxl, myl), (mxr, myr) = self._float_maps
        crops = [
            (np.ascontiguousarray(mxl[:, : self._x1]),
             np.ascontiguousarray(myl[:, : self._x1])),
            (np.ascontiguousarray(mxr[:, self._x0 :]),
             np.ascontiguousarray(myr[:, self._x0 :])),
        ]
        self.maps = [cv2.convertMaps(mx, my, cv2.CV_16SC2) if USE_FIXED_MAPS
                     else (mx, my) for mx, my in crops]
        self._crop_off = (0, self._x0)

    # 게인은 대입될 때마다 LUT 를 함께 만든다 (render 는 LUT 만 쓴다).
    @property
    def gain_l(self):
        return self._gain_l

    @gain_l.setter
    def gain_l(self, value):
        self._gain_l = value
        self._lut_l = gain_lut(value)

    @property
    def gain_r(self):
        return self._gain_r

    @gain_r.setter
    def gain_r(self, value):
        self._gain_r = value
        self._lut_r = gain_lut(value)

    def warp(self, img, side: int):
        """크롭 영역 워핑 (L: 폭 x1, R: 폭 out_w-x0)."""
        return cv2.remap(img, *self.maps[side], interpolation=cv2.INTER_LINEAR)

    def set_gains_from(self, img_l, img_r):
        """심 밴드 영역에서 채널 게인 추정."""
        x0, x1 = self._x0, self._x1
        band_l = self.warp(img_l, 0)[:, x0:x1]
        band_r = self.warp(img_r, 1)[:, : x1 - x0]
        g_l, g_r = compute_gains(band_l, band_r,
                                 self._masks[0][:, x0:x1], self._masks[1][:, x0:x1])
        self.gain_l = tuple(g_l) + (1.0,)
        self.gain_r = tuple(g_r) + (1.0,)

    def render(self, img_l, img_r):
        x0, x1 = self._x0, self._x1
        warp_l = cv2.LUT(self.warp(img_l, 0), self._lut_l)   # 폭 x1
        warp_r = cv2.LUT(self.warp(img_r, 1), self._lut_r)   # 폭 out_w-x0
        out = np.empty((self.out_h, self.out_w, 3), np.uint8)
        out[:, :x0] = warp_l[:, :x0]
        out[:, x1:] = warp_r[:, x1 - x0 :]
        out[:, x0:x1] = cv2.blendLinear(
            np.ascontiguousarray(warp_l[:, x0:x1]),
            np.ascontiguousarray(warp_r[:, : x1 - x0]),
            self.w_l_band, self.w_r_band)
        return out

    # ---------------------------------------------------------- 심 Y 정렬

    def _measure_seam_dy(self, img_l, img_r, band=90, margin=28, tile_h=120):
        """심 밴드에서 R 이 L 대비 세로로 얼마나 어긋났는지 (행별 dy, px) 측정.

        L 밴드를 템플릿으로 R 밴드(세로 여유 margin)에서 매칭. 반환 dy>0 은
        같은 내용이 R 에서 dy 픽셀 아래에 있음을 뜻한다.
        측정 밴드(±band)는 렌더용 크롭보다 넓으므로 float 맵에서 직접 워핑.
        """
        xc = self.out_w // 2
        lo, hi = max(0, xc - band), min(self.out_w, xc + band)
        warped = []
        for side, img in ((0, img_l), (1, img_r)):
            mx, my = self._float_maps[side]
            w = cv2.remap(img, np.ascontiguousarray(mx[:, lo:hi]),
                          np.ascontiguousarray(my[:, lo:hi]), cv2.INTER_LINEAR)
            warped.append(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY))
        gl, gr = warped
        xc = (hi - lo) // 2  # 이하 인덱싱은 밴드 로컬 좌표
        rows, dys, wts = [], [], []
        for y0 in range(0, self.out_h - tile_h, tile_h):
            y1 = y0 + tile_h
            tmpl = gl[y0 + margin : y1 - margin, xc - band : xc + band]
            if tmpl.std() < 4:   # 무늬 없는 잔디뿐이면 신뢰 불가
                continue
            # 가로 에지가 없으면 세로 매칭이 원리적으로 불가 (aperture) —
            # 심 근처의 방사형 잔디 줄무늬는 수직선이라 여기서 걸러진다
            if np.abs(cv2.Sobel(tmpl, cv2.CV_32F, 0, 1, ksize=3)).mean() < 6.0:
                continue
            region = gr[max(0, y0) : min(self.out_h, y1), xc - band : xc + band]
            if region.shape[0] < tmpl.shape[0] + 4:
                continue
            res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < 0.35:
                continue
            dy = float(loc[1] - margin)
            if abs(dy) >= margin - 2:    # 탐색 경계 포화 = 가짜 매칭
                continue
            dys.append(dy)
            rows.append((y0 + y1) / 2)
            wts.append(float(score))
        return np.array(rows), np.array(dys), np.array(wts)

    def refine_seam(self, img_l, img_r, taper_px=360, log=print):
        """심 밴드의 세로 어긋남을 실측해 R remap 을 심 주변에서만 국소 보정.

        시차 기인 오프셋이라 회전으로는 제거 불가 — R 소스 샘플 위치를
        행별 dy 만큼 이동시키되, 심에서 멀어질수록 0 으로 테이퍼.
        """
        rows, dys, wts = self._measure_seam_dy(img_l, img_r)
        if len(rows) < 4:
            log("[seam-refine] 측정 타일 부족 — 생략")
            return 0.0
        # 물리 모델 피팅: 시차는 거리 반비례 → 수평선 아래에서
        # dy = a + b·tan(-el), 위(원거리)에서는 상수 a.
        # 자유 다항식과 달리 pitch/세로범위가 바뀌어도 elevation 공간에서
        # 일관되고, 측정 없는 최하단으로도 물리적으로 타당하게 외삽된다.
        t1, t0 = np.tan(self.el1), np.tan(self.el0)
        t_rows = t1 + np.arange(self.out_h) / (self.out_h - 1) * (t0 - t1)
        depth_all = np.maximum(-t_rows, 0.0)          # ∝ 1/거리 (수평선 아래)
        depth_m = depth_all[rows.astype(int)]
        A = np.stack([np.ones_like(depth_m), depth_m], axis=1)
        sw = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], dys * sw, rcond=None)
        dy_fit = (coef[0] + coef[1] * depth_all).astype(np.float32)
        dy_fit = np.clip(dy_fit, -30 * self.scale - 5, 30 * self.scale + 5)
        rms0 = float(np.sqrt(np.average(dys**2, weights=wts)))

        taper = np.clip(1.0 - np.abs(np.arange(self.out_w) - self.out_w // 2)
                        / max(1, int(taper_px * self.scale)), 0.0, 1.0).astype(np.float32)
        # 출력 (x,y) 에서 R 소스를 (x, y + dy·taper) 의 기존 맵 값으로 샘플
        gx, gy = np.meshgrid(np.arange(self.out_w, dtype=np.float32),
                             np.arange(self.out_h, dtype=np.float32))
        gy = gy + dy_fit[:, None] * taper[None, :]
        mx, my = self._float_maps[1]
        mx2 = cv2.remap(mx, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        my2 = cv2.remap(my, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self._float_maps[1] = (mx2, my2)
        self._rebuild_cropped_maps()

        # 검증 재측정 — 악화되면 롤백 (부호/피팅 안전장치)
        rows2, dys2, wts2 = self._measure_seam_dy(img_l, img_r)
        rms1 = float(np.sqrt(np.average(dys2**2, weights=wts2))) if len(rows2) >= 3 else rms0
        if rms1 > rms0:
            self._float_maps[1] = (mx, my)
            self._rebuild_cropped_maps()
            log(f"[seam-refine] 개선 없음 (rms {rms0:.1f}→{rms1:.1f}px) — 롤백")
            return 0.0
        log(f"[seam-refine] 심 세로 어긋남 rms {rms0:.1f}px → {rms1:.1f}px")
        return rms0 - rms1
