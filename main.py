"""Run yuzu: webcam hand tracking driving the game window."""

from __future__ import annotations

import sys

from game.game import Game
from src import config
from src.camera import Camera
from src.coordinate_mapper import CoordinateMapper
from src.hand_tracker import HandTracker


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

            hand = tracker.process(frame)
            if hand["detected"] and hand["index_tip"] is not None:
                tip_x, tip_y = hand["index_tip"]
                game.set_fingertip(
                    mapper.map(tip_x, tip_y),
                    pointer_finger=hand.get("pointer_finger"),
                )
            else:
                game.set_fingertip(None)

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
