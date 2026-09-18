"""Run the VeriScope API and web app together for local development.

The process owns both child services. Press Ctrl+C once to stop them cleanly.
It deliberately injects only browser-safe local configuration into the Next.js
process; API secrets continue to be read by FastAPI from the root ``.env``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIRECTORY = ROOT / "apps" / "web"


def _command(name: str) -> str:
    """Return an executable name that works on Windows and Unix-like hosts."""

    if os.name == "nt":
        candidate = f"{name}.cmd"
        if shutil.which(candidate):
            return candidate
    if shutil.which(name):
        return name
    raise RuntimeError(f"Required command is not available on PATH: {name}")


def _stream_output(process: subprocess.Popen[str], label: str) -> None:
    """Forward one child process's output with an identifying prefix."""

    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=3000)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("CLASSICAL_MODEL_PATH", "models/classical/model.joblib")),
        help="Classical model artifact required for a complete local analysis.",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="Start the API even when no local model artifact is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    if not args.allow_missing_model and not model_path.is_file():
        raise RuntimeError(
            f"Model artifact not found: {model_path}. Train it first or pass "
            "--allow-missing-model to exercise the evidence-only state."
        )
    if not (WEB_DIRECTORY / "node_modules").is_dir():
        raise RuntimeError("Web dependencies are missing. Run npm install in apps/web first.")

    uv = _command("uv")
    npm = _command("npm")
    api_environment = os.environ.copy()
    api_environment.update(
        {
            "API_HOST": "127.0.0.1",
            "API_PORT": str(args.api_port),
            "CLASSICAL_MODEL_PATH": str(model_path),
            "CORS_ORIGINS": f"http://localhost:{args.web_port}",
        }
    )
    web_environment = os.environ.copy()
    web_environment.update(
        {
            "NEXT_PUBLIC_API_URL": f"http://localhost:{args.api_port}",
            "NEXT_PUBLIC_MIN_ARTICLE_LENGTH": api_environment.get(
                "MIN_ARTICLE_LENGTH", "100"
            ),
            "NEXT_PUBLIC_MAX_ARTICLE_LENGTH": api_environment.get(
                "MAX_ARTICLE_LENGTH", "20000"
            ),
        }
    )

    api = subprocess.Popen(
        [uv, "run", "uvicorn", "apps.api.main:app", "--reload", "--host", "127.0.0.1", "--port", str(args.api_port)],
        cwd=ROOT,
        env=api_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    web = subprocess.Popen(
        [npm, "run", "dev", "--", "--port", str(args.web_port)],
        cwd=WEB_DIRECTORY,
        env=web_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    processes = ((api, "api"), (web, "web"))
    threads = [
        threading.Thread(target=_stream_output, args=(process, label), daemon=True)
        for process, label in processes
    ]
    for thread in threads:
        thread.start()

    print(
        f"Local VeriScope is starting. Open http://localhost:{args.web_port} "
        "when the web service reports ready."
    )
    try:
        while all(process.poll() is None for process, _ in processes):
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nStopping local VeriScope services...")
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return max((process.returncode or 0 for process, _ in processes), default=0)


if __name__ == "__main__":
    raise SystemExit(main())
