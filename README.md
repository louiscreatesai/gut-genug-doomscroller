# Gut Genug Doomscroller 🎤

A Python script that watches your eyes through your webcam. Look down at your phone and it pops up **"DU BIST GUT GENUG"** by KITSCHKRIEG, blasting from the top with sound, until you look back up. The point: make doomscrolling so loud you stop.

> **Credits:** built on [strawluck](https://github.com/strawluck)'s ideas — the original [gut-genug-doomscroller-tracker](https://github.com/strawluck/gut-genug-doomscroller-tracker) and [mcdonalds-application-doomscroller](https://github.com/strawluck/mcdonalds-application-doomscroller). This version reuses the upgraded McDonald's engine (per-person calibrated gaze model, instant pre-warmed window, freeze-free camera, pause button, tests) and swaps the punishment for the Gut Genug clip. The clip is [KITSCHKRIEG's](https://www.youtube.com/shorts/NaOU8JZrjjg) — it's **not** included in this repo; it's downloaded on your machine on first run.

Works on **macOS** (tested) and should work on Windows/Linux (Chrome-based player, no AppleScript).

---

## How it works

~30 times a second:

1. **Grab a frame** from your webcam (on a background thread — see `camera.py`).
2. **Measure four signals** with MediaPipe FaceMesh: eye openness, iris position between the eyelids, iris position below the eye corners, and head pitch. People look at phones differently — some drop their whole head, others keep it level and only lower their eyes — so no fixed threshold works for everyone.
3. **Score it against your own calibration.** On first launch a two-step calibration records you looking at the screen, then at your phone. Each signal is rescaled so 0 = your screen pose and 1 = your phone pose, then averaged, weighted by how clearly each one separated the two poses *for you*. Score above 0.5 for half a second → **the clip starts from 0:00 and the window pops up**. Look back up → it pauses and hides.

| File | Job |
|---|---|
| `main.py` | Main loop, preview UI, glues everything together |
| `gaze.py` | Signal extraction, per-person calibration, unit-tested state machine |
| `camera.py` | Threaded webcam reader; reopens the camera if it stops delivering frames |
| `player.py` | The video window: pre-warmed Chrome, clip buffered, shown/hidden from a worker thread |
| `clip.py` | Downloads the clip on first run (yt-dlp) and stages it where Chrome may read it |
| `chromedriver.py` | Downloads a chromedriver matching your installed Chrome (cached per version) |
| `pause_button.py` | Clickable Pause / Resume button drawn into the preview |
| `assets/player.html` | The black, full-window player page |

**Instant, freeze-free window:** Chrome is started once in the background when the tracker launches, with the clip already buffered, then minimized. Triggering just restarts the clip and restores the window (~0.02s); all browser work runs on a worker thread so the webcam preview never freezes.

**Why the clip is played from a temp folder:** on macOS, Chrome isn't allowed to read files in `~/Documents` (or Desktop/Downloads) without a privacy prompt, and a page waiting on that prompt just hangs. So at launch the player page and clip are copied to the system temp folder and played from there.

---

## How to run it

### One-time setup — macOS

1. Install **Google Chrome**, **yt-dlp** and **ffmpeg** (for the one-time clip download):
   ```
   brew install yt-dlp ffmpeg
   ```
2. Create a venv with **Python 3.9–3.11** (mediapipe 0.10.9 doesn't support 3.12+). If the project lives in an iCloud-synced folder like `~/Documents`, name the venv `.venv.nosync` so iCloud doesn't offload it (that makes launches hang for minutes), and symlink `.venv` to it:
   ```
   python3.9 -m venv .venv.nosync && ln -s .venv.nosync .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. **Grant camera permission** to Terminal: System Settings → Privacy & Security → Camera.

On first run the clip is downloaded to `assets/gut-genug.mp4`. No yt-dlp? Just put your own copy of the video there.

### Every time

Double-click **`run.command`** (opens in Terminal), or:
```
.venv/bin/python main.py
```

The first time, follow the two calibration prompts (look at the screen, then at your phone). After that it's saved to `calibration.json`.

| Control | Action |
|---|---|
| **PAUSE** button (bottom-right) / **p** / **space** | Pause / resume tracking (hides the clip) |
| **c** | Recalibrate |
| **ESC** | Quit |

---

## Tuning

The preview shows the live `down-score` (0 = your screen pose, 1 = your phone pose). Edit `GazeConfig` in `gaze.py`:

```python
grace_seconds: float = 0.5   # how long you must look down before it triggers
enter_score: float = 0.5     # lower = more sensitive
exit_score: float = 0.3      # score must drop below this to stop
```

## Tests

```
pip install -r requirements-dev.txt
pytest --cov=gaze --cov=player --cov=camera --cov=clip --cov=pause_button --cov=main
```

## Credits

Original ideas and code by [strawluck](https://github.com/strawluck). Clip: "DU BIST GUT GENUG" by [KITSCHKRIEG](https://www.youtube.com/shorts/NaOU8JZrjjg). Built by [louiscreatesai](https://github.com/louiscreatesai).
