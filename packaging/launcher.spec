# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for the MyAIAssistant Launcher.
Creates a standalone MyAIAssistant.exe that orchestrates backend + frontend.
"""
import os
from pathlib import Path

PACKAGING_DIR = Path(os.path.abspath(SPECPATH))

block_cipher = None

a = Analysis(
    [str(PACKAGING_DIR / 'launcher.py')],
    pathex=[str(PACKAGING_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=[
        'urllib.request',
        'urllib.error',
        'webbrowser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MyAIAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console so user can see status
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PACKAGING_DIR / 'icon.ico') if (PACKAGING_DIR / 'icon.ico').exists() else None,
)
