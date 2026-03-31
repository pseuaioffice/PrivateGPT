"""
Master build script for MyAIAssistant desktop packaging.

Builds:
- backend.exe
- frontend.exe
- MyAIAssistant.exe launcher
- clean packaged runtime config
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from runtime_config import write_env_file


ROOT_DIR = Path(__file__).parent.parent.resolve()
PACKAGING_DIR = ROOT_DIR / "packaging"
DIST_DIR = PACKAGING_DIR / "dist"
BUILD_DIR = PACKAGING_DIR / "build_temp"


def clean() -> None:
    """Remove previous build artifacts."""
    for directory in (DIST_DIR, BUILD_DIR):
        if not directory.exists():
            continue
        try:
            shutil.rmtree(directory)
            print(f"  Cleaned: {directory}")
        except (PermissionError, OSError) as exc:
            print(f"  WARNING: Could not fully clean {directory}. Error: {exc}")
            print("  Attempting to continue anyway...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def run_pyinstaller(spec_name: str, work_subdir: str) -> None:
    spec_file = PACKAGING_DIR / spec_name
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_file),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / work_subdir),
        "--noconfirm",
    ]
    subprocess.run(cmd, check=True)


def build_backend() -> None:
    print("\n" + "=" * 60)
    print("  BUILDING BACKEND (FastAPI + Uvicorn)")
    print("=" * 60)

    run_pyinstaller("backend.spec", "backend")

    env_dst = DIST_DIR / "backend" / ".env"
    env_example_dst = DIST_DIR / "backend" / ".env.example"
    write_env_file(env_dst)
    write_env_file(env_example_dst)
    print(f"  Wrote clean runtime config to {env_dst}")
    print(f"  Wrote editable template to {env_example_dst}")

    (DIST_DIR / "backend" / "documents").mkdir(exist_ok=True)
    (DIST_DIR / "backend" / "chroma_db").mkdir(exist_ok=True)
    print("  OK Backend build complete!")


def build_frontend() -> None:
    print("\n" + "=" * 60)
    print("  BUILDING FRONTEND (Flask)")
    print("=" * 60)

    run_pyinstaller("frontend.spec", "frontend")
    print("  OK Frontend build complete!")


def build_launcher() -> None:
    print("\n" + "=" * 60)
    print("  BUILDING LAUNCHER (MyAIAssistant.exe)")
    print("=" * 60)

    run_pyinstaller("launcher.spec", "launcher")
    print("  OK Launcher build complete!")


def copy_fallback_launcher() -> None:
    launcher_src = PACKAGING_DIR / "MyAIAssistant_Launcher.bat"
    launcher_dst = DIST_DIR / "MyAIAssistant_Launcher.bat"
    if launcher_src.exists():
        shutil.copy2(launcher_src, launcher_dst)
        print(f"  Copied batch launcher to {launcher_dst}")

    icon_src = PACKAGING_DIR / "icon.ico"
    if icon_src.exists():
        shutil.copy2(icon_src, DIST_DIR / "icon.ico")


def verify_build() -> bool:
    print("\n" + "=" * 60)
    print("  VERIFYING BUILD")
    print("=" * 60)

    checks = [
        DIST_DIR / "backend" / "backend.exe",
        DIST_DIR / "frontend" / "frontend.exe",
        DIST_DIR / "MyAIAssistant.exe",
        DIST_DIR / "backend" / ".env",
        DIST_DIR / "backend" / ".env.example",
        DIST_DIR / "backend" / "documents",
        DIST_DIR / "backend" / "chroma_db",
        DIST_DIR / "MyAIAssistant_Launcher.bat",
    ]

    all_ok = True
    for path in checks:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {status:<8} {path.relative_to(DIST_DIR)}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n  Build verification PASSED!")
    else:
        print("\n  Some files are missing. Check errors above.")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  MyAIAssistant Desktop Builder")
    print("=" * 60)

    clean()
    build_backend()
    build_frontend()
    build_launcher()
    copy_fallback_launcher()
    success = verify_build()

    if success:
        print("\n" + "=" * 60)
        print("  BUILD COMPLETE!")
        print(f"  Output: {DIST_DIR}")
        print("  Next: compile packaging\\installer.iss with Inno Setup.")
        print("=" * 60)
    else:
        sys.exit(1)
