"""Tunable constants for camera, tracking, and gameplay."""

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

GAME_WIDTH = 1280
GAME_HEIGHT = 720

TARGET_FPS = 60

# Hand tracking. Detection is permissive so fast swipes re-acquire quickly;
# presence/tracking are stricter so half-baked landmarks are rejected.
HAND_MIN_DETECTION = 0.3
HAND_MIN_PRESENCE = 0.5
HAND_MIN_TRACKING = 0.4
# Keep using the last tip briefly when MediaPipe drops a few frames.
HAND_COAST_MS = 200
# Cap coast velocity so a noisy last frame doesn't fling the cursor.
HAND_MAX_COAST_SPEED = 2000.0
# Coast velocity decays with this time constant, bounding overshoot to about
# speed * decay when the hand reverses during dropped frames.
HAND_COAST_DECAY_MS = 70
# Infer below native 1280 so MediaPipe stays smooth on a laptop camera loop.
HAND_INFER_MAX_WIDTH = 960

# When the full-frame detector loses the hand (typically in front of the face
# or hair), re-detect inside a crop around the last known hand for this long.
HAND_SEARCH_MS = 700
# Crop is this many times the last hand's bounding box, at least this big.
HAND_SEARCH_SCALE = 2.5
HAND_SEARCH_MIN_PX = 360
# The crop has little clutter, so accept slightly weaker landmark presence.
HAND_ROI_MIN_PRESENCE = 0.4

# One Euro filter on the fingertip (camera pixels). Lower min_cutoff = steadier
# at rest; higher beta = follows fast swipes more tightly.
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.035
# Velocity estimate cutoff; higher tracks direction changes faster.
ONE_EURO_D_CUTOFF = 5.0
# How much the filter opens up sideways relative to along the swipe (0..1).
# Lower = straighter swipes; too low and curved swipes feel stiff.
ONE_EURO_LATERAL_SCALE = 0.2
# Cross-track deviation (camera px) at which sideways motion is treated as a
# real turn rather than wobble and gets full responsiveness.
ONE_EURO_LATERAL_GATE_PX = 12.0

# Palm-anchored tip. The tip is tracked as palm center + offset; the offset
# cutoff falls from SLOW (finger free to bend at rest) to FAST (offset nearly
# frozen) as speed approaches SPEED_FULL px/s, so a blurred tip landmark that
# drops toward the palm mid-swipe doesn't yank the cursor.
TIP_ANCHOR_CUTOFF_SLOW = 4.0
TIP_ANCHOR_CUTOFF_FAST = 0.5
TIP_ANCHOR_SPEED_FULL = 1000.0

# Pointer finger: a finger only takes over when its tip reaches this many
# times farther from the wrist than the next-best finger, for this many
# consecutive frames. Returning to index only needs one clear frame.
HAND_POINTER_DOMINANCE = 1.25
HAND_POINTER_SWITCH_FRAMES = 5

# Open-palm pause: every non-thumb finger must reach at least this far
# (tip-to-wrist / palm size). A pointing finger is typically ~1.8–2.5, but
# the other three stay curled (~1.0), so a point never counts as a palm.
HAND_PALM_MIN_REACH = 1.55
# Longest finger may only be this times the shortest; blocks a single point.
HAND_PALM_EVENNESS = 1.35
# Consecutive confirmed-palm / not-palm frames before pause / resume.
PALM_PAUSE_FRAMES = 6
PALM_RESUME_FRAMES = 4

# Legacy exponential smoother alpha (kept for debug comparisons).
SMOOTHING_ALPHA = 0.4

# Blade becomes active at or above this path speed (game px / s).
MIN_SWIPE_VELOCITY = 500
# Keep the blade lit briefly after speed drops so a continuous slash doesn't
# flicker off between camera frames.
SWIPE_HOLD_MS = 80

BLADE_HISTORY_MS = 150
BLADE_THICKNESS = 3
# Skip trail points that barely moved (duplicate CV frames).
BLADE_MIN_STEP_PX = 1.5
# Interpolated sub-steps between trail samples when drawing the curve.
BLADE_CURVE_SEGMENTS = 8
# Extra radius on the blade when testing fruit hits (px).
BLADE_HIT_RADIUS = 4.0
# Tapered slash: pixels at the trailing end vs the leading tip.
BLADE_TAIL_WIDTH = 1.5
BLADE_HEAD_WIDTH = 12.0

# Spawn cadence at the start of a round, then ramps toward *_END by time's up.
FRUIT_MIN_SPAWN_INTERVAL = 1.0
FRUIT_MAX_SPAWN_INTERVAL = 2.0
FRUIT_MIN_SPAWN_INTERVAL_END = 0.35
FRUIT_MAX_SPAWN_INTERVAL_END = 0.75
# After this fraction of the round, tosses can throw two fruits at once.
FRUIT_DOUBLE_SPAWN_START = 0.45
FRUIT_DOUBLE_SPAWN_CHANCE = 0.35
# Peak height as a fraction of the window, measured from the top.
FRUIT_PEAK_MIN = 0.18
FRUIT_PEAK_MAX = 0.42
# Keep the whole sprite inside this many pixels of the left/right edges.
FRUIT_SIDE_PAD = 28.0
# Spawn x as a fraction of width around the center (0.5 = full center).
FRUIT_SPAWN_HALF_SPAN = 0.18
# Peak x stays this close to the screen center (fraction of width).
FRUIT_PEAK_HALF_SPAN = 0.12
# Spawn a bit below the bottom edge so fruits rise into view.
FRUIT_SPAWN_MARGIN_Y = 40.0
# How hard halves fly apart along the cut normal (px/s).
SLICE_SEPARATION_SPEED = 340.0
SLICE_SPIN = 520.0

# Consecutive slices within this many seconds keep the combo streak alive.
COMBO_WINDOW = 1.6
# 3+ fruits in one swipe awards a COMBO bonus, Fruit Ninja classic style.
SWIPE_COMBO_MIN = 3
MAX_MULTIPLIER = 8

ROUND_SECONDS = 60

GRAVITY = 1100

DEBUG = False

ACTIVE_REGION = {
    "left": 0.10,
    "right": 0.90,
    "top": 0.10,
    "bottom": 0.90,
}
