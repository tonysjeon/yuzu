"""Run yuzu: webcam hand tracking driving the game window."""

from __future__ import annotations

import sys
import time

from game.game import Game
from src import config
from src.blade import Blade
from src.camera import Camera
from src.coordinate_mapper import CoordinateMapper
from src.hand_tracker import HandTracker
from src.swipe_detector import SwipeDetector


def main() -> int:
    try:
        camera = Camera()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        tracker = HandTracker()
    except FileNotFoundError as exc:
        camera.release()
        print(exc, file=sys.stderr)
        return 1

    game = Game()
    mapper = CoordinateMapper(
        camera_width=camera.width,
        camera_height=camera.height,
        game_width=game.width,
        game_height=game.height,
        active_region=config.ACTIVE_REGION,
    )
    blade = Blade()
    swipe = SwipeDetector()

    try:
        while game.running:
            game.handle_events()
            mapper.set_game_size(game.width, game.height)

            frame = camera.read()
            if frame is None:
                print(
                    "Failed to read from webcam. Check that nothing else is "
                    "using the camera and that Camera permission is enabled.",
                    file=sys.stderr,
                )
                return 1
            # Stamp after the blocking read so timing matches the frame.
            now = time.perf_counter()

            hand = tracker.process(frame)
            live = bool(hand["detected"] and not hand.get("coasted"))
            if live:
                game.set_palm(bool(hand.get("palm")), fist=bool(hand.get("fist")))
            else:
                # Lost or coasting: keep pause as-is instead of auto-resuming.
                game.set_palm(None)

            if (
                hand["detected"]
                and hand["index_tip"] is not None
                and not hand.get("palm")
                and not game.paused
            ):
                tip_x, tip_y = hand["index_tip"]
                mapped = mapper.map(tip_x, tip_y)
                game.set_fingertip(
                    mapped,
                    pointer_finger=hand.get("pointer_finger"),
                )
                blade.add(
                    mapped[0],
                    mapped[1],
                    timestamp=now,
                    confidence=float(hand.get("confidence") or 0.0),
                )
            else:
                game.set_fingertip(None)
                blade.expire(now)
                swipe.reset()

            swipe_result = swipe.update(blade.points, now=now)
            if game.paused:
                game.set_blade_points([], active=False, velocity=0.0, segments=[])
            else:
                game.set_blade_points(
                    blade.curve() if swipe_result.active else [],
                    active=swipe_result.active,
                    velocity=swipe_result.velocity,
                    segments=blade.polyline() if swipe_result.active else [],
                )
            game.update()
            game.render()
            game.tick()
    finally:
        tracker.close()
        camera.release()
        game.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
