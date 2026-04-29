# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # путь к easyocr – замените на ваш реальный путь, если отличается
        ('build_env/Lib/site-packages/easyocr', 'easyocr'),
    ],
    hiddenimports=[
        'easyocr',
        'cv2',
        'torch',
        'sounddevice',
        'soundfile',          # добавлен soundfile
        'PIL',
        'requests',
        'win32com',
        'pythoncom',
        'numpy',
        'scipy',
        'skimage',
        'PyQt5',
        'PyQt5.sip',
        'queue',
        'difflib',
        'ctypes',
        'json',
        're',
        'os',
        'time',
        'threading',
        'traceback',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GenshinVoice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                # без окна командной строки
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GenshinVoice'
)