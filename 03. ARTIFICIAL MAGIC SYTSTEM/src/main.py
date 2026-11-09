import argparse
import time

import cv2
import numpy as np

from config import CHARACTERS
from hand_tracking import HandTracker
from vfx import MagicRenderer

def draw_hud(frame, character, hands, fps, show_help=True):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 82), (12, 10, 24), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, "MYSTIC MOTION", (18, 31),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, (235, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, character.name, (18, 61),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, character.accent, 2, cv2.LINE_AA)
    status = f"HANDS: {len(hands)}   FPS: {fps:.0f}"
    cv2.putText(frame, status, (w - 230, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 245), 1, cv2.LINE_AA)
    if show_help:
        help_text = "1 Strange  2 Wanda  3 Agatha  4 Loki  5 Thor  6 Hela  7 Wong  8 Ancient One  9 Wiccan"
        cv2.putText(frame, help_text, (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 215, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q: quit | H: hide help | S: show/hide hand skeleton",
                    (16, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 185, 205), 1, cv2.LINE_AA)

def draw_skeleton(frame, points):
    h, w = frame.shape[:2]
    p = [(int(x * w), int(y * h)) for x, y, _ in points]
    connections = [
        (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    for a, b in connections:
        cv2.line(frame, p[a], p[b], (110, 210, 180), 1, cv2.LINE_AA)
    for point in p:
        cv2.circle(frame, point, 2, (220, 245, 235), -1, cv2.LINE_AA)


import time
import cv2

from config import CHARACTERS
from hand_tracking import HandTracker
from vfx import MagicRenderer


def run(
    camera_index=0,
    width=1280,
    height=720,
    mirror=True,
    show_skeleton=False
):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check macOS camera permission or try camera_index=1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracker = None

    try:
        tracker = HandTracker(max_hands=2)
        renderer = MagicRenderer()

        character_key = "1"
        show_help = True
        previous = time.perf_counter()
        fps = 0.0

        print("\nMYSTIC MOTION started.")
        print("Select characters using keys 1-9.")
        print("H = Help | S = Skeleton | Q = Quit\n")

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Camera frame could not be read. Stopping.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - previous, 1e-6)
            previous = now

            current_fps = 1.0 / dt
            fps = (
                0.9 * fps + 0.1 * current_fps
                if fps else current_fps
            )

            elapsed = now

            hands = tracker.process(frame)
            character = CHARACTERS[character_key]

            for hand_id, hand in enumerate(hands):
                if show_skeleton:
                    draw_skeleton(frame, hand["points"])

                renderer.render_hand(
                    frame,
                    character,
                    hand,
                    elapsed,
                    hand_id
                )

            if len(hands) >= 2:
                renderer.render_two_hand_field(
                    frame,
                    character,
                    hands,
                    elapsed
                )

            draw_hud(
                frame,
                character,
                hands,
                fps,
                show_help
            )

            cv2.imshow(
                "MYSTIC MOTION | Live Magic Experience",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("Closing Mystic Motion...")
                break

            elif key in (ord("h"), ord("H")):
                show_help = not show_help

            elif key in (ord("s"), ord("S")):
                show_skeleton = not show_skeleton

            elif ord("1") <= key <= ord("9"):
                selected_key = chr(key)

                if selected_key in CHARACTERS:
                    character_key = selected_key
                    print(
                        "Selected:",
                        CHARACTERS[character_key].name
                    )

                else:
                    print(
                        f"Character {selected_key} is not configured."
                    )

    finally:
        if tracker is not None:
            tracker.close()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mystic Motion real-time hand VFX")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--skeleton", action="store_true")
    args = parser.parse_args()
    run(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        show_skeleton=args.skeleton,
    )
