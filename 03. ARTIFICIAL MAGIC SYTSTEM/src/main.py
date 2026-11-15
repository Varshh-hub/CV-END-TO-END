import argparse
import sys
import time

import cv2

from config import CHARACTERS
from hand_tracking import HandTracker
from vfx import MagicRenderer


def draw_hud(frame, character, hands, fps, show_help=True):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 88), (12, 10, 24), -1)
    cv2.addWeighted(panel, 0.74, frame, 0.26, 0, frame)
    cv2.putText(frame, "MYSTIC MOTION", (18, 31),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, (235, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, character.name, (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, character.accent, 2, cv2.LINE_AA)
    cv2.putText(frame, f"HANDS: {len(hands)}   FPS: {fps:.0f}", (w - 230, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 245), 1, cv2.LINE_AA)
    if show_help:
        cv2.putText(frame, "1 Strange  2 Wanda  3 Agatha  4 Loki  5 Thor  6 Wong  7 Ancient One",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (210, 215, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q: quit | H: help | Magic is palm-centred; move your hand to change its size",
                    (16, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (180, 185, 205), 1, cv2.LINE_AA)


def run(camera_index=0, width=1280, height=720, mirror=True):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check camera permissions or try --camera 1."
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

        print("\\nMYSTIC MOTION started.")
        print("Select characters using keys 1-7.")
        print("Magic is automatic whenever a hand is detected; no gesture is required.")
        print("Effects stay on the palm, contain no finger-connecting lines, and scale with movement.")
        print("H = Help | Q = Quit\\n")

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
            fps = 0.9 * fps + 0.1 * current_fps if fps else current_fps
            hands = tracker.process(frame)
            character = CHARACTERS[character_key]

            # Each detected hand gets an independent palm-anchored effect.
            for hand_id, hand in enumerate(hands):
                renderer.render_hand(frame, character, hand, now, hand_id)

            draw_hud(frame, character, hands, fps, show_help)
            cv2.imshow("MYSTIC MOTION | Live Magic Experience", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Closing Mystic Motion...")
                break
            elif key in (ord("h"), ord("H")):
                show_help = not show_help
            elif ord("1") <= key <= ord("7"):
                character_key = chr(key)
                print("Selected:", CHARACTERS[character_key].name)

    finally:
        if tracker is not None:
            tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mystic Motion real-time palm VFX")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    run(camera_index=args.camera, width=args.width, height=args.height,
        mirror=not args.no_mirror)
