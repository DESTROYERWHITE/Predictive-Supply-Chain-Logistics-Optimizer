from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


APP_PORT = 8501
APP_URL = f"http://localhost:{APP_PORT}"


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    project_root = base_dir()
    if not (project_root / "app.py").exists() and project_root.name.lower() == "dist":
        project_root = project_root.parent

    app_path = project_root / "app.py"
    python_executable = "python" if getattr(sys, "frozen", False) else sys.executable

    if not app_path.exists():
        print(f"Could not find dashboard app at {app_path}")
        print("Keep this launcher in the project folder beside app.py.")
        input("Press Enter to close...")
        return 1

    if not is_port_open(APP_PORT):
        subprocess.Popen(
            [
                python_executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless",
                "true",
                "--server.port",
                str(APP_PORT),
            ],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        for _ in range(40):
            if is_port_open(APP_PORT):
                break
            time.sleep(0.5)

    webbrowser.open(APP_URL)
    print(f"Dashboard is running at {APP_URL}")
    print("You can close this window after the browser opens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
