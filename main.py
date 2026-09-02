"""Run yuzu: webcam hand tracking driving the game window."""

from __future__ import annotations

import sys

from game.game import Game
from src.camera import Camera
from src.hand_tracker import HandTracker


def _map_to_game(
    tip: tuple[float, float],
    camera_width: int,
    camera_height: int,
    game_width: int,
    game_height: int,
) -> tuple[float, float]:
    """Simple scale from camera pixels into the game surface."""
    x = tip[0] / max(camera_width, 1) * game_width
    y = tip[1] / max(camera_height, 1) * game_height
    x = min(max(x, 0.0), game_width - 1.0)
    y = min(max(y, 0.0), game_height - 1.0)
    return x, y


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
    try:
        while game.running:
            game.handle_events()

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
                game.set_fingertip(
                    _map_to_game(
                        hand["index_tip"],
                        camera.width,
                        camera.height,
                        game.width,
                        game.height,
                    ),
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
