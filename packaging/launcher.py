"""
MyAIAssistant desktop launcher.

This executable is the single entry point for packaged Windows installs.
It makes sure runtime config exists, starts backend/frontend processes with
the configured ports, waits for them to become healthy, and opens the browser.
"""
from __future__ import annotations

import ctypes
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from runtime_config import ensure_env_file, load_runtime_config


APP_NAME = "MyAIAssistant"
CREATE_NO_WINDOW = 0x08000000
BOOL_TRUE_VALUES = {"1", "true", "yes", "on"}

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent / "dist"

BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"
BACKEND_EXE = BACKEND_DIR / "backend.exe"
FRONTEND_EXE = FRONTEND_DIR / "frontend.exe"
ENV_PATH = BACKEND_DIR / ".env"
ENV_TEMPLATE_PATH = BACKEND_DIR / ".env.example"
LOG_DIR = BACKEND_DIR / "logs"

runtime_config: dict[str, str] = {}
backend_host = "127.0.0.1"
backend_port = 8000
frontend_host = "127.0.0.1"
frontend_port = 5000
backend_url = "http://127.0.0.1:8000"
frontend_url = "http://127.0.0.1:5000"
auto_open_browser = True
start_ollama_with_launcher = True

backend_proc: Optional[subprocess.Popen] = None
frontend_proc: Optional[subprocess.Popen] = None
backend_log_handle = None
frontend_log_handle = None
backend_managed = False
frontend_managed = False
shutdown_requested = False


def set_console_title(title: str) -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetConsoleTitleW(title)


def bool_from_string(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in BOOL_TRUE_VALUES


def int_from_string(value: str, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if 1 <= parsed <= 65535:
        return parsed
    return default


def load_runtime_settings() -> None:
    global runtime_config
    global backend_host, backend_port, frontend_host, frontend_port
    global backend_url, frontend_url, auto_open_browser, start_ollama_with_launcher

    ensure_env_file(ENV_PATH, ENV_TEMPLATE_PATH)
    runtime_config = dict(load_runtime_config(ENV_PATH))

    backend_host = runtime_config.get("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    frontend_host = runtime_config.get("FRONTEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    backend_port = int_from_string(runtime_config.get("BACKEND_PORT", "8000"), 8000)
    frontend_port = int_from_string(runtime_config.get("FRONTEND_PORT", "5000"), 5000)

    backend_url = f"http://{backend_host}:{backend_port}"
    frontend_url = f"http://{frontend_host}:{frontend_port}"
    auto_open_browser = bool_from_string(runtime_config.get("AUTO_OPEN_BROWSER", "true"), True)
    start_ollama_with_launcher = bool_from_string(
        runtime_config.get("START_OLLAMA_WITH_LAUNCHER", "true"),
        True,
    )


def tail_file(path: Path, max_chars: int = 1500) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def wait_for_http_ok(url: str, timeout: int = 60, path: str = "") -> bool:
    target = f"{url}{path}"
    deadline = time.time() + timeout
    while time.time() < deadline and not shutdown_requested:
        try:
            with urlopen(target, timeout=3) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(1)
    return False


def is_tcp_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def open_log_file(name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return (LOG_DIR / name).open("a", encoding="utf-8")


def print_banner() -> None:
    print("=" * 60)
    print("  MyAIAssistant Desktop Launcher")
    print("=" * 60)


def ensure_ollama_running() -> bool:
    provider = runtime_config.get("MODEL_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        print("  [0/4] Cloud provider selected in .env, skipping Ollama startup.")
        return True

    if not start_ollama_with_launcher:
        print("  [0/4] Ollama auto-start disabled in .env.")
        return True

    print("  [0/4] Checking Ollama service...")
    if wait_for_http_ok(runtime_config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), timeout=3, path="/api/tags"):
        print("  OK Ollama is already running.")
        return True

    print("  WARN Ollama is not running. Attempting to start background server...")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["ollama", "serve"], creationflags=CREATE_NO_WINDOW)

            if wait_for_http_ok(runtime_config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), timeout=15, path="/api/tags"):
                print("  OK Ollama started successfully.")
                return True
    except Exception as exc:
        print(f"  WARN Could not auto-start Ollama: {exc}")

    print("  ERROR Ollama is required but not running.")
    print("  Please install Ollama from https://ollama.com/download")
    print("  Then launch it once and re-run MyAIAssistant.")
    return False


def start_backend() -> bool:
    global backend_proc, backend_log_handle, backend_managed

    print("  [1/4] Starting backend server...")
    if not BACKEND_EXE.exists():
        print(f"  ERROR Backend executable not found: {BACKEND_EXE}")
        return False

    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "documents").mkdir(exist_ok=True)
    (BACKEND_DIR / "chroma_db").mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if wait_for_http_ok(backend_url, timeout=2, path="/health"):
        backend_managed = False
        print("  OK Backend is already running. Reusing existing instance.")
        return True

    if is_tcp_port_open(backend_host, backend_port):
        print(
            f"  ERROR Port {backend_port} is already in use, but it is not responding as the MyAIAssistant backend."
        )
        print("  Edit backend/.env and change BACKEND_PORT, then launch again.")
        return False

    env = os.environ.copy()
    env["HOST"] = backend_host
    env["PORT"] = str(backend_port)
    backend_log_handle = open_log_file("backend.log")
    backend_proc = subprocess.Popen(
        [str(BACKEND_EXE)],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=backend_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    backend_managed = True

    print("  [2/4] Waiting for backend health check...")
    if wait_for_http_ok(backend_url, timeout=90, path="/health"):
        print("  OK Backend is running.")
        return True

    print("  ERROR Backend did not become healthy in time.")
    backend_log_path = LOG_DIR / "backend.log"
    log_tail = tail_file(backend_log_path)
    if log_tail:
        print("  Recent backend log output:")
        print(log_tail)
    else:
        print(f"  Check log file: {backend_log_path}")
    return False


def start_frontend() -> bool:
    global frontend_proc, frontend_log_handle, frontend_managed

    print("  [3/4] Starting frontend server...")
    if not FRONTEND_EXE.exists():
        print(f"  ERROR Frontend executable not found: {FRONTEND_EXE}")
        return False

    if wait_for_http_ok(frontend_url, timeout=2):
        frontend_managed = False
        print("  OK Frontend is already running. Reusing existing instance.")
        return True

    if is_tcp_port_open(frontend_host, frontend_port):
        print(
            f"  ERROR Port {frontend_port} is already in use, but it is not responding as the MyAIAssistant frontend."
        )
        print("  Edit backend/.env and change FRONTEND_PORT, then launch again.")
        return False

    env = os.environ.copy()
    env["BACKEND_URL"] = backend_url
    env["HOST"] = frontend_host
    env["PORT"] = str(frontend_port)
    env["DEBUG"] = runtime_config.get("DEBUG", "false")

    frontend_log_handle = open_log_file("frontend.log")
    frontend_proc = subprocess.Popen(
        [str(FRONTEND_EXE)],
        cwd=str(FRONTEND_DIR),
        env=env,
        stdout=frontend_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    frontend_managed = True

    if wait_for_http_ok(frontend_url, timeout=60):
        print("  OK Frontend is running.")
        return True

    print("  ERROR Frontend did not become healthy in time.")
    frontend_log_path = LOG_DIR / "frontend.log"
    log_tail = tail_file(frontend_log_path)
    if log_tail:
        print("  Recent frontend log output:")
        print(log_tail)
    else:
        print(f"  Check log file: {frontend_log_path}")
    return False


def open_browser_window() -> None:
    if not auto_open_browser:
        print(f"  [4/4] Browser auto-open disabled. Open {frontend_url} manually.")
        return
    print(f"  [4/4] Opening browser at {frontend_url}")
    time.sleep(1)
    webbrowser.open(frontend_url)


def stop_process(name: str, proc: Optional[subprocess.Popen], managed: bool) -> None:
    if not managed or proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
        print(f"  OK {name} stopped.")
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  WARN {name} had to be force-killed.")
    except Exception as exc:
        print(f"  WARN Could not stop {name}: {exc}")


def cleanup() -> None:
    global shutdown_requested
    global backend_log_handle, frontend_log_handle

    shutdown_requested = True
    print("\n  Shutting down MyAIAssistant...")
    stop_process("Frontend", frontend_proc, frontend_managed)
    stop_process("Backend", backend_proc, backend_managed)

    for handle in (frontend_log_handle, backend_log_handle):
        if handle:
            try:
                handle.close()
            except Exception:
                pass


def signal_handler(_sig, _frame) -> None:
    cleanup()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    set_console_title(f"{APP_NAME} - Starting")
    load_runtime_settings()

    print_banner()
    print("  Starting MyAIAssistant with the following configuration:")
    print(f"    Backend : {backend_url}")
    print(f"    Frontend: {frontend_url}")
    print(f"    Provider: {runtime_config.get('MODEL_PROVIDER', 'ollama')}")
    print("")

    if not ensure_ollama_running():
        input("\n  Press Enter to exit...")
        cleanup()
        sys.exit(1)

    if not start_backend():
        input("\n  Press Enter to exit...")
        cleanup()
        sys.exit(1)

    if not start_frontend():
        input("\n  Press Enter to exit...")
        cleanup()
        sys.exit(1)

    open_browser_window()
    set_console_title(f"{APP_NAME} - Running")

    print("\n" + "=" * 60)
    print("  MyAIAssistant is running.")
    print(f"  App URL    : {frontend_url}")
    print(f"  Backend API: {backend_url}")
    print(f"  Logs       : {LOG_DIR}")
    print("=" * 60)
    print("  Keep this window open while using MyAIAssistant.")
    print("  Press Ctrl+C or close this window to stop it.\n")

    try:
        while not shutdown_requested:
            if backend_managed and backend_proc and backend_proc.poll() is not None:
                print("  WARN Backend exited unexpectedly. See backend.log for details.")
                break
            if frontend_managed and frontend_proc and frontend_proc.poll() is not None:
                print("  WARN Frontend exited unexpectedly. See frontend.log for details.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
