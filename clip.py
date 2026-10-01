"""Gets the Gut Genug clip ready to play.

The video isn't in the repo (it's KITSCHKRIEG's, not ours to redistribute),
so on first run it's downloaded with yt-dlp into assets/. You can also drop
your own copy at assets/gut-genug.mp4.

Chrome then plays it from a copy in the system temp folder: on macOS, Chrome
isn't allowed to read ~/Documents (or Desktop/Downloads) without a privacy
prompt, and a page that's waiting on that prompt just hangs.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

CLIP_URL = "https://www.youtube.com/shorts/NaOU8JZrjjg"
ASSETS_DIR = Path(__file__).parent / "assets"
CLIP_PATH = ASSETS_DIR / "gut-genug.mp4"
PLAYER_HTML = ASSETS_DIR / "player.html"
STAGE_DIR = Path(tempfile.gettempdir()) / "gut-genug-doomscroller"
TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
DOWNLOAD_TIMEOUT_SECONDS = 180

Downloader = Callable[[str, Path], None]


def find_tool(name: str) -> Optional[str]:
    """Finds a command-line tool, also checking Homebrew's folders (a
    double-clicked .command file may not have them on PATH)."""
    found = shutil.which(name)
    if found:
        return found
    return next((str(p) for p in (Path(d) / name for d in TOOL_DIRS) if p.exists()), None)


def download_command(yt_dlp: str, url: str, dest: Path, ffmpeg: Optional[str]) -> List[str]:
    # Best H.264 + AAC when ffmpeg can merge the streams, otherwise the best
    # single file that already has both.
    fmt = "bv*[vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4]/b" if ffmpeg else "b[ext=mp4]/b"
    cmd = [yt_dlp, "-q", "--no-warnings", "-f", fmt, "--merge-output-format", "mp4", "-o", str(dest), url]
    if ffmpeg:
        cmd[1:1] = ["--ffmpeg-location", ffmpeg]
    return cmd


def _yt_dlp_download(url: str, dest: Path) -> None:
    # The yt-dlp CLI rather than the Python package: YouTube changes often and
    # yt-dlp has dropped Python 3.9 (which mediapipe needs here), so the pip
    # version stuck in this venv would soon stop working.
    yt_dlp = find_tool("yt-dlp")
    if yt_dlp is None:
        raise RuntimeError("yt-dlp isn't installed (macOS: brew install yt-dlp)")
    cmd = download_command(yt_dlp, url, dest, find_tool("ffmpeg"))
    subprocess.run(cmd, check=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)


def ensure_clip(path: Path = CLIP_PATH, url: str = CLIP_URL,
                downloader: Downloader = _yt_dlp_download) -> Optional[Path]:
    """Returns the clip's path, downloading it first if it's missing.
    Returns None (after explaining why) if it can't be obtained."""
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading the Gut Genug clip (first run only) from {url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloader(url, path)
    except Exception as e:  # yt-dlp raises many types; any failure means "no clip"
        print(f"Could not download the clip: {e}")
    if path.exists() and path.stat().st_size > 0:
        print("Clip downloaded.")
        return path
    print(f"Put a copy of the video at {path} and run again.")
    return None


def _copy_if_changed(src: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == src.stat().st_size \
            and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    shutil.copy2(src, dest)


def stage_for_chrome(clip: Path, html: Path = PLAYER_HTML, stage_dir: Path = STAGE_DIR) -> Path:
    """Copies the player page and clip somewhere Chrome may read without a
    privacy prompt; returns the staged page's path."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_changed(html, stage_dir / "player.html")
    _copy_if_changed(clip, stage_dir / "gut-genug.mp4")
    return stage_dir / "player.html"
