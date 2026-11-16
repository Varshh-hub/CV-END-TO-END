import argparse
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

    cv2.putText(
        frame, "MYSTIC MOTION", (18, 31),
        cv2.FONT_HERSHEY_DUPLEX, 0.78,
        (235, 235, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        frame, character.name, (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        character.accent, 2, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"HANDS: {len(hands)}   FPS: {fps:.0f}",
        (w - 230, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (235, 235, 245), 1, cv2.LINE_AA
    )

    if show_help:
        cv2.putText(
            frame,
            "1 Strange  2 Wanda  3 Agatha  4 Loki  5 Thor",
            (16, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (210, 215, 235), 1, cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Q: Quit | H: Help | Magic activates automatically on detected palms",
            (16, h - 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            (180, 185, 205), 1, cv2.LINE_AA
        )


def run(camera_index=0, width=1280, height=720, mirror=True):
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
        print("1 Doctor Strange")
        print("2 Scarlet Witch")
        print("3 Agatha Harkness")
        print("4 Loki")
        print("5 Thor")
        print("H = Help | Q = Quit")

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Camera frame could not be read.")
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

            # Pose-dependent formations:
            # 1) Palms turned inward -> ONLY the shared central force orb.
            # 2) Both palms presented to camera -> palm effects + a third
            #    character-specific magical formation in the middle.
            # 3) One hand / other poses -> palm-centered magic only.

            inward_pose = (
                len(hands) >= 2
                and renderer.palms_facing_each_other(hands[0], hands[1])
            )

            camera_pose = (
                len(hands) >= 2
                and renderer.palms_facing_camera(hands[0], hands[1])
            )

            if inward_pose:
                # Avoid individual palm effects and avoid straight connecting
                # lines: render the existing character-specific orb only.
                renderer.render_two_hand_field(
                    frame, character, hands, now
                )

            else:
                # Individual effects on visible palms; size adapts to apparent
                # palm scale (near/far from camera) and motion.
                for hand_id, hand in enumerate(hands):
                    renderer.render_hand(
                        frame, character, hand, now, hand_id
                    )

                # When both palms face the camera, add a distinct third
                # magical formation centered between them, no connector lines.
                if camera_pose:
                    renderer.render_two_hand_field(
                        frame, character, hands, now, force=True
                    )

            draw_hud(frame, character, hands, fps, show_help)

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

            elif ord("1") <= key <= ord("5"):
                character_key = chr(key)
                print("Selected:", CHARACTERS[character_key].name)

    finally:
        if tracker is not None:
            tracker.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mystic Motion real-time palm VFX"
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")

    args = parser.parse_args()

    run(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror
    )
