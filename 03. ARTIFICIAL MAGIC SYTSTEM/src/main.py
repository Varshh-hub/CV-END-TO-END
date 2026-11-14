import argparse
import sys
import time
import random

import cv2
import numpy as np

from config import CHARACTERS, INPAINT_DOWNSCALE
from hand_tracking import HandTracker, PersonSegmenter
from vfx import MagicRenderer


class SnapDetector:
    """Scale-invariant thumb-index pinch and flick detector."""

    def __init__(self):
        self.pinched_at = {}
        self.last_snap = 0.0
        self.previous_distance = {}

    def update(self, hands, now):
        snapped = False
        for hand_id, hand in enumerate(hands):
            points = hand["points"]
            # Scale-invariant normalization: hand scale is distance from wrist (0) to middle MCP (9)
            hand_scale = float(np.linalg.norm(points[0, :2] - points[9, :2])) + 1e-6
            rel_distance = float(np.linalg.norm(points[4, :2] - points[8, :2])) / hand_scale

            was_pinched = hand_id in self.pinched_at
            if rel_distance < 0.22:
                self.pinched_at[hand_id] = now
            elif was_pinched:
                held = now - self.pinched_at.pop(hand_id)
                previous = self.previous_distance.get(hand_id, rel_distance)
                quick_release = rel_distance > 0.45 and held < 0.65
                sharp_motion = (rel_distance - previous) > 0.15
                if quick_release and sharp_motion and (now - self.last_snap) > 1.0:
                    snapped = True
                    self.last_snap = now
            self.previous_distance[hand_id] = rel_distance

        # Forget stale pinches when a hand disappears from camera
        active_ids = set(range(len(hands)))
        for hand_id in list(self.pinched_at):
            if hand_id not in active_ids and now - self.pinched_at[hand_id] > 0.6:
                self.pinched_at.pop(hand_id, None)
        return snapped


def draw_hud(frame, character, hands, fps, show_help=True, vanished=False, has_bg=False):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 88), (12, 10, 24), -1)
    cv2.addWeighted(panel, 0.74, frame, 0.26, 0, frame)
    cv2.putText(frame, "MYSTIC MOTION", (18, 31),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, (235, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, character.name, (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, character.accent, 2, cv2.LINE_AA)
    status = f"HANDS: {len(hands)}   FPS: {fps:.0f}"
    cv2.putText(frame, status, (w - 230, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 245), 1, cv2.LINE_AA)
    if character.name == "THANOS":
        state = "SNAPPED / VANISHED" if vanished else "READY - PINCH & FLICK TO SNAP"
        if has_bg:
            state += " [BG CAPTURED]"
        cv2.putText(frame, state, (w - 460, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (185, 100, 245) if vanished else (225, 220, 245), 1, cv2.LINE_AA)
    if show_help:
        help_text = "1 Strange  2 Wanda  3 Agatha  4 Loki  5 Thor  6 Thanos"
        cv2.putText(frame, help_text, (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 215, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q: quit | H: help | S: hand skeleton | B: capture clean background",
                    (16, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 185, 205), 1, cv2.LINE_AA)


def draw_skeleton(frame, points):
    h, w = frame.shape[:2]
    p = [(int(x * w), int(y * h)) for x, y, _ in points]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]
    for a, b in connections:
        cv2.line(frame, p[a], p[b], (110, 210, 180), 1, cv2.LINE_AA)
    for point in p:
        cv2.circle(frame, point, 2, (220, 245, 235), -1, cv2.LINE_AA)


def apply_snap_disappearance(frame, segmenter, progress, elapsed, background_plate=None, inpaint_downscale=INPAINT_DOWNSCALE):
    """Blend the segmented person into a clean background plate or high-speed downscaled inpainting."""
    mask_float = segmenter.mask(frame)
    if mask_float is None:
        return frame
    person_mask = (mask_float > 0.35).astype(np.uint8) * 255
    if cv2.countNonZero(person_mask) == 0:
        return frame

    h, w = frame.shape[:2]
    if background_plate is not None:
        clean_background = background_plate
    else:
        # High-performance downscaled inpainting: runs ~16x faster than full 720p inpaint
        dw = max(1, int(w * inpaint_downscale))
        dh = max(1, int(h * inpaint_downscale))
        small_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(person_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        small_inpaint = cv2.inpaint(small_frame, small_mask, 3, cv2.INPAINT_TELEA)
        clean_background = cv2.resize(small_inpaint, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = (mask_float * progress).clip(0.0, 1.0)[..., None]
    result = (frame.astype(np.float32) * (1.0 - alpha) +
              clean_background.astype(np.float32) * alpha).astype(np.uint8)

    # Cosmic dissolving dust particles drifting upward
    if progress > 0.02:
        rng = random.Random(int(elapsed * 100))
        ys, xs = np.where(person_mask > 0)
        if len(xs):
            overlay = np.zeros_like(frame)
            count = min(300, max(50, len(xs) // 100))
            indices = rng.sample(range(len(xs)), min(count, len(xs)))
            for idx in indices:
                x, y = int(xs[idx]), int(ys[idx])
                float_up = int(progress * 18 + rng.uniform(-4, 4))
                y = max(0, y - float_up)
                radius = rng.choice((1, 2, 3))
                color = rng.choice(((155, 75, 220), (210, 165, 65), (235, 220, 255)))
                cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
            result = cv2.addWeighted(result, 1.0, overlay, 0.9, 0)
    return result


def run(camera_index=0, width=1280, height=720, mirror=True, show_skeleton=False):
    # Platform-specific camera initialization for low latency on Windows
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check camera permissions or try --camera 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracker = None
    segmenter = None
    background_plate = None

    try:
        tracker = HandTracker(max_hands=2)
        renderer = MagicRenderer()
        snap_detector = SnapDetector()

        character_key = "1"
        show_help = True
        previous = time.perf_counter()
        fps = 0.0
        vanished = False
        transition_start = None
        transition_from = 0.0
        transition_to = 0.0
        transition_duration = 1.25

        print("\nMYSTIC MOTION started.")
        print("Select characters using keys 1-6.")
        print("In Thanos mode: pinch thumb + index finger, then quickly flick apart.")
        print("Press 'B' anytime to capture a clean room background for instant Thanos dissolution.")
        print("H = Help | S = Skeleton | B = Background | Q = Quit\n")

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
            elapsed = now

            hands = tracker.process(frame)
            character = CHARACTERS[character_key]

            if character.name == "THANOS":
                if segmenter is None:
                    segmenter = PersonSegmenter()
                if snap_detector.update(hands, now):
                    transition_from = 1.0 if vanished else 0.0
                    transition_to = 0.0 if vanished else 1.0
                    transition_start = now
                    vanished = not vanished

                if transition_start is not None:
                    fraction = min(1.0, (now - transition_start) / transition_duration)
                    eased = fraction * fraction * (3.0 - 2.0 * fraction)
                    progress = transition_from + (transition_to - transition_from) * eased
                    frame = apply_snap_disappearance(frame, segmenter, progress, elapsed, background_plate)
                    if fraction >= 1.0:
                        transition_start = None
                elif vanished:
                    frame = apply_snap_disappearance(frame, segmenter, 1.0, elapsed, background_plate)
            else:
                vanished = False
                transition_start = None

            # Render skeleton if requested
            if show_skeleton:
                for hand in hands:
                    draw_skeleton(frame, hand["points"])

            # High-performance unified rendering of all hand VFX and particles
            renderer.render_all(frame, character, hands, elapsed, dt)

            draw_hud(frame, character, hands, fps, show_help, vanished, has_bg=(background_plate is not None))
            cv2.imshow("MYSTIC MOTION | Live Magic Experience", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Closing Mystic Motion...")
                break
            elif key in (ord("h"), ord("H")):
                show_help = not show_help
            elif key in (ord("s"), ord("S")):
                show_skeleton = not show_skeleton
            elif key in (ord("b"), ord("B")):
                background_plate = frame.copy()
                print("Captured clean background plate!")
            elif ord("1") <= key <= ord("6"):
                character_key = chr(key)
                print("Selected:", CHARACTERS[character_key].name)
                if CHARACTERS[character_key].name != "THANOS":
                    vanished = False
                    transition_start = None

    finally:
        if tracker is not None:
            tracker.close()
        if segmenter is not None:
            segmenter.close()
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
