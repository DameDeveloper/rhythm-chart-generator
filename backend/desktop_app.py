"""Desktop launcher for the Rhythm Chart Generator.

Bundles the FastAPI backend and the pre-built React frontend into a single
process:

  * The existing API (``main.app``) is mounted under ``/api`` so the frontend's
    ``fetch('/api/...')`` calls work without a dev proxy.
  * The built frontend (``dist/``) is served as static files at ``/``.
  * A free local port is chosen, the server is started, and the default browser
    is opened automatically.

Run directly during development (``python desktop_app.py``) or frozen into a
single ``.exe`` with PyInstaller (see ``rhythm_chart_generator.spec``).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def resource_path(rel: str) -> Path:
    """Resolve a bundled resource, working both frozen and unfrozen.

    Frozen: looks inside the PyInstaller extraction dir (``sys._MEIPASS``).
    Dev: looks next to this file, then falls back to the source tree layout
    (e.g. the frontend build lives at ``../frontend/dist``).
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel
    here = Path(__file__).resolve().parent
    candidates = [here / rel]
    if rel == "dist":
        candidates.append(here.parent / "frontend" / "dist")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def writable_base() -> Path:
    """Directory the user can write to (next to the .exe, or the source dir)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# The API stores transient audio/song files under DATA.  In a frozen build the
# module lives in a temp extraction dir that is wiped on exit, so redirect it to
# a persistent folder beside the executable.
import main  # noqa: E402  (import after helpers so we can override DATA)

DATA_DIR = writable_base() / "data" / "downloads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
main.DATA = DATA_DIR

root = FastAPI(title="Rhythm Chart Generator (Desktop)")
root.mount("/api", main.app)

_DIST = resource_path("dist")
if _DIST.is_dir():
    root.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")


def find_free_port(preferred: tuple[int, ...] = (8000, 8001, 8080, 5000)) -> int:
    for port in preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _chromium_exe() -> str | None:
    """Locate an Edge/Chrome executable for standalone app-window mode."""
    candidates = [
        os.path.expandvars(p) for p in (
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
            r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
        )
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def open_app_window(url: str) -> None:
    """Open the app as a chromeless "web app" window when possible.

    Falls back to the default browser tab if no Chromium browser is found.
    """
    exe = _chromium_exe()
    if exe:
        profile = str(writable_base() / "data" / "browser-profile")
        try:
            subprocess.Popen([
                exe,
                f"--app={url}",
                f"--user-data-dir={profile}",
                "--new-window",
            ])
            return
        except OSError:
            pass
    webbrowser.open(url)


def run() -> None:
    import uvicorn

    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"

    print("=" * 48)
    print("  Rhythm Chart Generator")
    print(f"  주소: {url}")
    print("  창이 자동으로 열립니다. 종료하려면 이 콘솔 창을 닫으세요.")
    print("=" * 48)

    threading.Timer(1.5, lambda: open_app_window(url)).start()
    uvicorn.run(root, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run()
