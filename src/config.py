"""Tunable constants for camera, tracking, and gameplay."""

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

GAME_WIDTH = 1280
GAME_HEIGHT = 720

TARGET_FPS = 60

# Hand tracking — lower thresholds hold up better during fast swipes.
HAND_MIN_DETECTION = 0.3
HAND_MIN_PRESENCE = 0.3
HAND_MIN_TRACKING = 0.3
# Keep using the last tip briefly when MediaPipe drops a few frames.
HAND_COAST_MS = 180
# Infer closer to full res so upright fingertips stay sharp.
HAND_INFER_MAX_WIDTH = 1280
# Tip EMA; lower = smoother (helps upright / foreshortened fingers).
HAND_TIP_SMOOTHING = 0.45
# Ignore single-frame tip teleports (pixels at camera resolution).
HAND_MAX_TIP_JUMP = 120.0
# Cap coast velocity so flicker doesn't fling the cursor.
HAND_MAX_COAST_SPEED = 2500.0
# Stick to the current pointer finger unless another is clearly more extended.
HAND_POINTER_LOCK_FRAMES = 10
HAND_POINTER_SWITCH_MARGIN = 0.35
# Soft preference for the anatomical index when scores are close.
HAND_INDEX_BIAS = 12.0
# Curl vs extend: tip must reach this far past the PIP (ratio tip-mcp / pip-mcp).
HAND_EXTEND_RATIO = 1.45
# PIP joint must bend past this cos threshold to count as extended (more negative = straighter).
HAND_EXTEND_PIP_COS = -0.15

SMOOTHING_ALPHA = 0.4

MIN_SWIPE_VELOCITY = 500

BLADE_HISTORY_MS = 150

FRUIT_MIN_SPAWN_INTERVAL = 1.0
FRUIT_MAX_SPAWN_INTERVAL = 2.0

GRAVITY = 1100

DEBUG = True

ACTIVE_REGION = {
    "left": 0.10,
    "right": 0.90,
    "top": 0.10,
    "bottom": 0.90,
}
