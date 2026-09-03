# Yuzu

Fruit Ninja with your webcam. Point a finger to slash; MediaPipe tracks your hand in real time.

## Stack

- **Python** 3.11 or 3.12
- **pygame** — window, sprites, HUD
- **MediaPipe** — hand landmarks
- **OpenCV** — camera capture
- **NumPy** — rendering and physics helpers

Uses the laptop webcam on Windows and Mac.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.13+ won’t work yet — pygame doesn’t ship wheels for it.

## Play

```bash
python main.py
```

Point a finger at the yuzu on the title screen and slash to start. Swipe through fruit for points. Combos raise a multiplier; three or more in one slash awards a bonus. Miss the bombs — slicing one ends the round. Open palm pauses (or press `P`); point again to resume, or make a fist to return to the menu. `R` / Space restarts after a round. `Q` / Esc quits.

If the camera won’t open, allow Camera access for your terminal in system privacy settings, then run again.
