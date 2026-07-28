# -*- mode: python ; coding: utf-8 -*-
# TouchlineAnalyst onedir — PitchStitch.exe + PitchWatch.exe 를 **한
# COLLECT** 에 담는다 (P10). 별도 폴더 2개로 만들면 PyQt/cv2 가 중복되고
# 인스톨러도 갈라진다 — 한 폴더에 exe 2개면 torch 도 1벌만 실리고,
# PitchStitch 는 런타임에 torch 를 import 하지 않아 경량 기동이 유지된다.
#
# ultralytics 는 cfg yaml 등 데이터 파일을 런타임에 읽으므로 collect_all
# 필수. YOLO 가중치·ffmpeg 는 번들하지 않는다 (P10 결정: 가중치는
# ultralytics 자동 다운로드, ffmpeg 는 ffmpeg_bin() 탐색 + 설치 문서).
from PyInstaller.utils.hooks import collect_all

block_cipher = None

u_datas, u_bins, u_hidden = collect_all('ultralytics')
e_datas, e_bins, e_hidden = collect_all('easyocr')

common_datas = [('presets', 'presets')]

a_watch = Analysis(
    ['pitchwatch.py'],
    pathex=[],
    binaries=u_bins + e_bins,
    datas=common_datas + u_datas + e_datas,
    hiddenimports=['torch', 'torchvision'] + u_hidden + e_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

a_stitch = Analysis(
    ['pitchstitch.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # PitchStitch 는 설계상 torch 없이 기동 (pitchstitch.py docstring) —
    # 분석 스택이 스크립트 그래프에 끌려 들어오지 않게 명시 차단.
    excludes=['torch', 'torchvision', 'ultralytics', 'easyocr'],
    cipher=block_cipher,
    noarchive=False,
)

pyz_watch = PYZ(a_watch.pure, a_watch.zipped_data, cipher=block_cipher)
pyz_stitch = PYZ(a_stitch.pure, a_stitch.zipped_data, cipher=block_cipher)

exe_watch = EXE(
    pyz_watch,
    a_watch.scripts,
    [],
    exclude_binaries=True,
    name='PitchWatch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX 는 백신 오탐 (CTHarvester/Modan2 전례)
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_stitch = EXE(
    pyz_stitch,
    a_stitch.scripts,
    [],
    exclude_binaries=True,
    name='PitchStitch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_watch,
    exe_stitch,
    a_watch.binaries,
    a_watch.zipfiles,
    a_watch.datas,
    a_stitch.binaries,
    a_stitch.zipfiles,
    a_stitch.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TouchlineAnalyst',
)
