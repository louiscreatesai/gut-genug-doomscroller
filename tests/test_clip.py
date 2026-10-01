from pathlib import Path

import clip
from clip import CLIP_URL, download_command, ensure_clip, find_tool, stage_for_chrome


class FakeDownloader:
    def __init__(self, writes: bool = True, raises: Exception = None):
        self.calls = []
        self.writes = writes
        self.raises = raises

    def __call__(self, url: str, dest: Path) -> None:
        self.calls.append((url, dest))
        if self.raises:
            raise self.raises
        if self.writes:
            dest.write_bytes(b"fake video")


def test_existing_clip_is_used_without_downloading(tmp_path):
    path = tmp_path / "gut-genug.mp4"
    path.write_bytes(b"video")
    downloader = FakeDownloader()

    assert ensure_clip(path, downloader=downloader) == path
    assert downloader.calls == []


def test_missing_clip_is_downloaded_from_the_youtube_short(tmp_path):
    path = tmp_path / "assets" / "gut-genug.mp4"
    downloader = FakeDownloader()

    assert ensure_clip(path, downloader=downloader) == path
    assert downloader.calls == [(CLIP_URL, path)]
    assert "NaOU8JZrjjg" in CLIP_URL


def test_empty_file_counts_as_missing(tmp_path):
    path = tmp_path / "gut-genug.mp4"
    path.write_bytes(b"")
    downloader = FakeDownloader()

    ensure_clip(path, downloader=downloader)

    assert len(downloader.calls) == 1


def test_download_error_returns_none_with_instructions(tmp_path, capsys):
    path = tmp_path / "gut-genug.mp4"

    result = ensure_clip(path, downloader=FakeDownloader(raises=RuntimeError("yt-dlp isn't installed")))

    assert result is None
    out = capsys.readouterr().out
    assert "yt-dlp isn't installed" in out and str(path) in out


def test_download_that_writes_nothing_returns_none(tmp_path):
    assert ensure_clip(tmp_path / "gut-genug.mp4", downloader=FakeDownloader(writes=False)) is None


def test_download_command_merges_best_h264_when_ffmpeg_available():
    cmd = download_command("/bin/yt-dlp", CLIP_URL, Path("/x/clip.mp4"), ffmpeg="/bin/ffmpeg")

    assert cmd[0] == "/bin/yt-dlp" and cmd[-1] == CLIP_URL
    assert cmd[cmd.index("--ffmpeg-location") + 1] == "/bin/ffmpeg"
    assert "avc1" in cmd[cmd.index("-f") + 1]
    assert cmd[cmd.index("-o") + 1] == "/x/clip.mp4"


def test_download_command_without_ffmpeg_uses_single_file_format():
    cmd = download_command("/bin/yt-dlp", CLIP_URL, Path("/x/clip.mp4"), ffmpeg=None)

    assert "--ffmpeg-location" not in cmd
    assert cmd[cmd.index("-f") + 1] == "b[ext=mp4]/b"


def test_find_tool_falls_back_to_homebrew_dirs(tmp_path, monkeypatch):
    fake_bin = tmp_path / "yt-dlp"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr(clip.shutil, "which", lambda name: None)
    monkeypatch.setattr(clip, "TOOL_DIRS", (str(tmp_path),))

    assert find_tool("yt-dlp") == str(fake_bin)
    assert find_tool("not-a-tool") is None


def test_stage_copies_page_and_clip_outside_protected_folders(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "player.html").write_text("<video>")
    (src / "gut-genug.mp4").write_bytes(b"video")
    stage = tmp_path / "stage"

    page = stage_for_chrome(src / "gut-genug.mp4", html=src / "player.html", stage_dir=stage)

    assert page == stage / "player.html"
    assert (stage / "gut-genug.mp4").read_bytes() == b"video"


def test_stage_refreshes_a_changed_clip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "player.html").write_text("<video>")
    clip_file = src / "gut-genug.mp4"
    clip_file.write_bytes(b"old")
    stage = tmp_path / "stage"
    stage_for_chrome(clip_file, html=src / "player.html", stage_dir=stage)

    clip_file.write_bytes(b"new and longer")
    stage_for_chrome(clip_file, html=src / "player.html", stage_dir=stage)

    assert (stage / "gut-genug.mp4").read_bytes() == b"new and longer"
