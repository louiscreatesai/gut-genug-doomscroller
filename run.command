#!/bin/bash
# Double-click (or `open run.command`) to launch in Terminal.app, which
# triggers macOS's camera-permission prompt for Terminal the first time.
cd "$(dirname "$0")"
.venv/bin/python -u main.py 2>&1 | tee run.log
