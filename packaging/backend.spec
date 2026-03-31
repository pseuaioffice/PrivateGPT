# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for the MyAIAssistant Backend (FastAPI + Uvicorn).
Bundles all Python dependencies into a single folder with backend.exe.

IMPORTANT: The documents/ and chroma_db/ folders are intentionally NOT
bundled here.  They are created empty at install time by Inno Setup and
are written to at runtime (in the installed location, next to backend.exe).
Bundling them would embed the developer's own documents into every installer.
"""
import os
from pathlib import Path

# Paths
ROOT = Path(os.path.abspath(SPECPATH)).parent
BACKEND_DIR = ROOT / "backend"

block_cipher = None

# Collect all backend Python source files (no documents, no chroma_db)
backend_py_files = [
    (str(BACKEND_DIR / 'main.py'),             '.'),
    (str(BACKEND_DIR / 'config.py'),           '.'),
    (str(BACKEND_DIR / 'database.py'),         '.'),
    (str(BACKEND_DIR / 'chat_history.py'),     '.'),
    (str(BACKEND_DIR / 'document_loader.py'),  '.'),
    (str(BACKEND_DIR / 'embeddings.py'),       '.'),
    (str(BACKEND_DIR / 'indexer.py'),          '.'),
    (str(BACKEND_DIR / 'model_manager.py'),    '.'),
    (str(BACKEND_DIR / 'rag_chain.py'),        '.'),
    (str(BACKEND_DIR / 'vector_store.py'),     '.'),
    (str(BACKEND_DIR / 'find_working_model.py'), '.'),
]

a = Analysis(
    [str(BACKEND_DIR / 'backend_entry.py')],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=backend_py_files,   # .env is written by build.py separately; no docs/chroma bundled
    hiddenimports=[
        # FastAPI / Uvicorn
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'fastapi',
        'fastapi.middleware',
        'fastapi.middleware.cors',
        'starlette',
        'starlette.routing',
        'starlette.middleware',
        'starlette.responses',
        'starlette.websockets',
        'anyio._backends._asyncio',

        # Pydantic
        'pydantic',
        'pydantic.deprecated.decorator',

        # LangChain & Document Parsers
        'langchain',
        'langchain_core',
        'langchain_core.documents',
        'langchain_text_splitters',
        'langchain_community',
        'langchain_community.llms',
        'langchain_community.embeddings',
        'langchain_huggingface',

        # Document processing
        'pypdf',
        'docx2txt',
        'openpyxl',
        'openpyxl.cell',
        'openpyxl.styles',
        'openpyxl.workbook',
        'openpyxl.worksheet',
        'et_xmlfile',
        'xlrd',

        # Database
        'psycopg2',
        'psycopg2.extras',
        'psycopg2.extensions',
        'sqlalchemy',
        'sqlite3',

        # Watchdog
        'watchdog',
        'watchdog.observers',
        'watchdog.observers.polling',
        'watchdog.events',

        # Other
        'dotenv',
        'multipart',
        'python_multipart',
        'requests',
        'openai',

        # Our modules
        'config',
        'database',
        'chat_history',
        'document_loader',
        'embeddings',
        'indexer',
        'model_manager',
        'rag_chain',
        'vector_store',
        'find_working_model',
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
    name='backend',
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
    name='backend',
)
