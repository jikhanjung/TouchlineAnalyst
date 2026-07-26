# 080: NVENC 해상도 한계 — 파노라마는 hevc_nvenc, 판정은 실인코드로

- 날짜: 2026-07-26
- 종류: 버그픽스 (#11 WSL 실행 중 발견)

## 실측으로 확정한 사실들

#11(효창)을 WSL 에서 시작하며 3연속 실패 → 원인 사슬 전부 실측:

1. **h264 NVENC 는 폭 4096px 까지** — 5312px 테스트에서 정확히 배치와
   같은 'No capable devices found' 재현. 6196px 파노라마는 h264 NVENC
   로 인코드 자체가 불가. (--codec auto 가 h264_nvenc 를 골라 죽었다.)
2. **hevc_nvenc 는 8192px 까지** — 5312px 성공. 6K 파노라마의 GPU
   인코드 경로는 HEVC.
3. **WSL 도 NVENC 가 동작한다** (libnvidia-encode 존재, 최신 드라이버).
   반대로 Windows winget ffmpeg 는 드라이버 버전 불일치(API 13.1 요구
   vs 13.0)로 NVENC 고장 — '어느 쪽이 GPU 되는지'는 목록이 아니라
   런타임 판정이 필수.
4. 첫 프로브가 64px 로 테스트해 **NVENC 최소 해상도 미만 가짜 실패** —
   256px 로 수정.

## 수정

- `nvenc_works(encoder)` — 인코더별 1프레임 실인코드 프로브 (256px,
  lru_cache).
- `resolve_codec(codec, width=None)` — 폭 인지: ≤4096 h264_nvenc →
  ≤8192 hevc_nvenc → libx264.
- `export_pano` — out_w 확정 후 재판정: h264_nvenc 인데 폭 초과면
  hevc_nvenc/libx264 로 강등 + 로그 (강제 지정도 죽지 않게).
- GUI ProxyWorker·headless 프록시도 nvenc_works() 경유.
- **headless 스킵 가드**: 실패한 인코드가 남긴 0바이트 pano 를 '있음'
  으로 보고 건너뛰어 분석이 moov 없음으로 죽던 것(실측 2회) — 1MB
  미만이면 깨진 잔재로 삭제 후 재스티칭.

## 결과

#11 이 WSL 에서 hevc_nvenc 로 정상 가동: "[encode] 폭 6196px > h264
NVENC 한계 4096 → hevc_nvenc", 스티칭 ~20fps (29.9분 영상, 예상 45분).
테스트 151 통과.
