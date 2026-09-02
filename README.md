# Yuzu

Play a Fruit Ninja–style game with your index finger using a webcam and real-time hand tracking.

## Requirements

- Python 3.11 or 3.12 (3.13+ is not supported yet — pygame has no wheels)
- A working webcam

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use `python3.11` instead if that is what you have installed.

## Run

```bash
python main.py
```

Opens a `yuzu` game window and draws your index fingertip as a circle.
Uses the Mac's built-in camera (Continuity Camera / iPhone is skipped).
The window is resizable. Press `q` or `Esc` to quit.

If the camera fails to open on macOS, enable **Camera** access for Terminal (or Cursor) under **System Settings → Privacy & Security → Camera**, then rerun.
