# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for the MyAIAssistant Frontend (Flask).
Bundles all Python dependencies + HTML templates into a single folder with frontend.exe.
"""
import os
from pathlib import Path

# Paths
ROOT = Path(os.path.abspath(SPECPATH)).parent
FRONTEND_DIR = ROOT / "frontend"

block_cipher = None

a = Analysis(
    [str(FRONTEND_DIR / 'frontend_entry.py')],
    pathex=[str(FRONTEND_DIR)],
    binaries=[],
    datas=[
        # Include HTML templates
        (str(FRONTEND_DIR / 'templates'), 'templates'),
    ],
    hiddenimports=[
        'flask',
        'flask.json',
        'flask.templating',
        'jinja2',
        'jinja2.ext',
        'markupsafe',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.debug',
        'requests',
        'urllib3',
        'charset_normalizer',
        'certifi',
        'idna',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'jupyter',
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
    [],
    exclude_binaries=True,
    name='frontend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='frontend',
)
