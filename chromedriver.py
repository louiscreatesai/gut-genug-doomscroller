"""Finds or downloads a chromedriver matching the installed Chrome.

Bypasses Selenium's bundled auto-driver-manager, which has a known crash on
some macOS point releases — this does the same job (match installed Chrome's
major version, fetch the right platform's driver) directly against Google's
own Chrome-for-Testing index. Drivers are cached per Chrome major version, so
a Chrome auto-update fetches a fresh matching driver instead of reusing a
stale one that would refuse to start.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DRIVER_CACHE_DIR = Path(__file__).parent / ".driver"
MILESTONES_URL = "https://googlechromelabs.github.io/chrome-for-testing/latest-versions-per-milestone-with-downloads.json"

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # Windows
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",  # Windows
]


def chrome_major_version() -> str | None:
    for path in CHROME_PATHS:
        if not Path(path).exists():
            continue
        try:
            out = subprocess.check_output([path, "--version"], text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"(\d+)\.\d+\.\d+\.\d+", out)
        if m:
            return m.group(1)
    return None


def _driver_platform_key() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "mac-arm64" if machine in ("arm64", "aarch64") else "mac-x64"
    if system == "Windows":
        return "win64" if "64" in machine or machine == "amd64" else "win32"
    return "linux64"


def _download_url(major: str) -> str:
    with urllib.request.urlopen(MILESTONES_URL, timeout=10) as resp:
        data = json.load(resp)
    platform_key = _driver_platform_key()
    return next(
        d["url"] for d in data["milestones"][major]["downloads"]["chromedriver"]
        if d["platform"] == platform_key
    )


def ensure_chromedriver() -> str | None:
    """Returns a path to a chromedriver matching installed Chrome, downloading
    and caching it if needed, or None to let Selenium find one itself."""
    exe_name = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
    major = chrome_major_version()
    if major is None:
        return None
    cached = DRIVER_CACHE_DIR / major / exe_name
    if cached.exists():
        return str(cached)

    try:
        url = _download_url(major)
    except (OSError, KeyError, StopIteration, ValueError) as e:
        print(f"Could not resolve a matching chromedriver download: {e}")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "chromedriver.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            extracted = next(Path(tmp).rglob(exe_name))
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(extracted, cached)
        cached.chmod(0o755)
    except (OSError, zipfile.BadZipFile, StopIteration) as e:
        print(f"Could not download chromedriver: {e}")
        return None
    return str(cached)
