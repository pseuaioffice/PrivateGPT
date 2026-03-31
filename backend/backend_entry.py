"""
backend_entry.py — Entry point for the PyInstaller-compiled backend.
Sets runtime paths correctly then starts uvicorn.

Path strategy (frozen exe):
  - exe lives at  {install_dir}/backend/backend.exe
  - documents →   {install_dir}/backend/documents/
  - chroma_db →   {install_dir}/backend/chroma_db/
  - .env      →   {install_dir}/backend/.env

We set DOCUMENTS_DIR and CHROMA_PERSIST_DIR as environment variables
with absolute paths BEFORE loading the .env file so that relative paths
in .env (./documents, ./chroma_db) are never used in a frozen build.
"""
import sys
import os
import shutil

# Suppress harmless urllib3/chardet dependency warnings that cause confusion
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")


def _setup_frozen_paths():
    """Configure all paths for a PyInstaller-frozen executable."""
    # Directory containing backend.exe (the real executable)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))

    # Change working directory so that relative imports still work
    os.chdir(exe_dir)

    # _MEIPASS is where PyInstaller unpacks the bundled Python files
    base_dir = getattr(sys, '_MEIPASS', exe_dir)

    # Ensure both are on sys.path so our modules can be imported
    for p in (base_dir, exe_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Set absolute runtime paths BEFORE loading .env.
    # We use os.environ[] (hard set) NOT setdefault() to ensure these
    # always point to the installed location — even if a stale Windows
    # environment variable or a previous test session left an old value.
    documents_dir = os.path.join(exe_dir, 'documents')
    chroma_dir    = os.path.join(exe_dir, 'chroma_db')
    logs_dir      = os.path.join(exe_dir, 'logs')

    os.environ['DOCUMENTS_DIR']      = documents_dir
    os.environ['CHROMA_PERSIST_DIR'] = chroma_dir

    # Ensure the directories exist
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(chroma_dir,    exist_ok=True)
    os.makedirs(logs_dir,      exist_ok=True)

    # Load .env — we use override=False so user-defined settings in .env
    # are respected but our critical path variables above are NOT overwritten.
    env_path = os.path.join(exe_dir, '.env')
    env_example_path = os.path.join(exe_dir, '.env.example')
    if not os.path.exists(env_path) and os.path.exists(env_example_path):
        shutil.copyfile(env_example_path, env_path)
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)  # override=False keeps our paths above

    # Re-stamp paths after dotenv load to be absolutely sure they weren't overwritten
    os.environ['DOCUMENTS_DIR']      = documents_dir
    os.environ['CHROMA_PERSIST_DIR'] = chroma_dir

    print(f"[backend_entry] exe_dir       = {exe_dir}")
    print(f"[backend_entry] documents_dir = {documents_dir}")
    print(f"[backend_entry] chroma_dir    = {chroma_dir}")


if getattr(sys, 'frozen', False):
    _setup_frozen_paths()

import uvicorn


def main():
    """Start the backend server."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    print(f"Starting MyAIAssistant Backend on {host}:{port}")
    print(f"  Documents dir : {os.getenv('DOCUMENTS_DIR', '(not set)')}")
    print(f"  Chroma dir    : {os.getenv('CHROMA_PERSIST_DIR', '(not set)')}")

    from main import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
