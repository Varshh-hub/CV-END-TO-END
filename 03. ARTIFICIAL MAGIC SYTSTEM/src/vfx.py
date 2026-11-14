import math
import random
from collections import deque

import cv2
import numpy as np

from config import Character


# ---------------------------------------------------------------------
# Shared drawing primitives. Each character uses its own visual language;
# there is deliberately no generic finger-to-finger lightning/line effect.
# ---------------------------------------------------------------------

def palm_center(points):
    return np.mean(points[[0, 5, 9, 13, 17], :2], axis=0)


def to_px(point, width, height):
    return (
        int(np.clip(point[0] * width, 0, width - 1)),
        int(np.clip(point[1] * height, 0, height - 1)),
    )


def hand_is_open(points):
    wrist = points[0, :2]
    checks = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = sum(
        np.linalg.norm(points[tip, :2] - wrist)
        > np.linalg.norm(points[pip, :2] - wrist) * 1.12
        for tip, pip in checks
    )
    return extended >= 3


def is_pinching(points, threshold=0.060):
    return float(np.linalg.norm(points[4, :2] - points[8, :2])) < threshold


def poly(layer, points, color, thickness=2, closed=False):
    if len(points) >= 2:
        pts = np.asarray(points, dtype=np.int32)
        cv2.polylines(layer, [pts], closed, color, thickness, cv2.LINE_AA)


def glow(layer, center, radius, color, core=(245, 248, 255), strength=1.0):
    x, y = center
    cv2.circle(layer, (x, y), max(2, int(radius * 2.8)), color, -1, cv2.LINE_AA)
    cv2.circle(layer, (x, y), max(1, int(radius)), core, -1, cv2.LINE_AA)


def bloom_composite(frame, overlay, blur=13, bloom_strength=0.8, detail_strength=0.95):
    soft = cv2.GaussianBlur(overlay, (0, 0), blur)
    frame[:] = cv2.addWeighted(frame, 1.0, soft, bloom_strength, 0)
    frame[:] = cv2.addWeighted(frame, 1.0, overlay, detail_strength, 0)


def mandala(layer, center, radius, color, accent, t):
    """Layered, symmetrical sorcerer mandala: clean geometry, ticks and runes."""
    cx, cy = center
    radius = max(34, int(radius))
    # Fine concentric rings with intentional gaps.
    for i, scale in enumerate((1.00, 0.91, 0.78, 0.64, 0.48, 0.30)):
        r = int(radius * scale)
        start = int((t * (18 if i % 2 else -14) + i * 17) % 360)
        cv2.ellipse(layer, center, (r, r), 0, start, start + 350,
                    color if i % 2 == 0 else accent, 2 if i < 3 else 1, cv2.LINE_AA)

    # Radial divisions and small rune-like rectangular marks.
    for i in range(48):
        a = math.tau * i / 48 + t * 0.025
        r1 = radius * (0.82 if i % 4 else 0.75)
        r2 = radius * (0.91 if i % 4 else 1.04)
        p1 = (int(cx + math.cos(a) * r1), int(cy + math.sin(a) * r1))
        p2 = (int(cx + math.cos(a) * r2), int(cy + math.sin(a) * r2))
        cv2.line(layer, p1, p2, accent if i % 4 == 0 else color,
                 2 if i % 4 == 0 else 1, cv2.LINE_AA)

    # Nested rotated squares / octagonal geometry.
    for sides, scale, rotation in ((4, 0.69, 45), (4, 0.52, 0), (8, 0.39, 22.5)):
        pts = []
        for i in range(sides):
            a = math.tau * i / sides + math.radians(rotation) + t * 0.035
            r = radius * scale
            pts.append((int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)))
        poly(layer, pts, color, 2, True)

    # Short rune strokes around the outer band; kept separate, not connected.
    for i in range(24):
        a = math.tau * i / 24 + t * 0.018
        r = radius * 0.96
        x, y = int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)
        tx, ty = -math.sin(a), math.cos(a)
        length = 5 if i % 3 else 10
        cv2.line(layer, (int(x - tx * length), int(y - ty * length)),
                 (int(x + tx * length), int(y + ty * length)),
                 accent, 1, cv2.LINE_AA)

    # Bright central seal.
    cv2.circle(layer, center, max(8, int(radius * 0.16)), color, 2, cv2.LINE_AA)
    cv2.circle(layer, center, max(3, int(radius * 0.07)), accent, 2, cv2.LINE_AA)

def electric_bolt(layer, start, end, color, seed, amplitude=16, branches=2, thickness=2):
    """Jagged branching lightning with a bright white-hot core."""
    rng = random.Random(seed)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = b - a
    length = np.linalg.norm(delta) + 1e-6
    perp = np.array([-delta[1], delta[0]]) / length
    count = max(5, int(length / 24))
    pts = [tuple(a.astype(int))]
    for i in range(1, count):
        f = i / count
        offset = rng.uniform(-amplitude, amplitude)
        q = a + delta * f + perp * offset
        pts.append(tuple(q.astype(int)))
    pts.append(tuple(b.astype(int)))
    poly(layer, pts, color, thickness)
    poly(layer, pts, (245, 250, 255), max(1, thickness // 2))
    if branches:
        for j in range(branches):
            idx = rng.randint(1, max(2, len(pts) - 2))
            origin = pts[idx]
            branch_end = (
                int(origin[0] + rng.uniform(-35, 35)),
                int(origin[1] + rng.uniform(-42, 42)),
            )
            electric_bolt(layer, origin, branch_end, color, seed + j + 91,
                          amplitude * 0.45, 0, max(1, thickness - 1))


def energy_wisp(layer, start, end, color, t, seed=0, width=3):
    """Soft flowing red/purple energy filament, not a straight connector."""
    rng = random.Random(seed)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = b - a
    length = np.linalg.norm(delta) + 1e-6
    perp = np.array([-delta[1], delta[0]]) / length
    pts = [tuple(a.astype(int))]
    count = max(8, int(length / 18))
    phase = rng.random() * math.tau
    for i in range(1, count):
        f = i / count
        envelope = math.sin(math.pi * f)
        offset = math.sin(t * 5.0 + f * 15.0 + phase) * 13 * envelope
        offset += math.sin(t * 8.0 + f * 29.0 + phase) * 4 * envelope
        q = a + delta * f + perp * offset
        pts.append(tuple(q.astype(int)))
    pts.append(tuple(b.astype(int)))
    poly(layer, pts, color, width + 3)
    poly(layer, pts, (255, 235, 245), max(1, width - 1))


class MagicRenderer:
    """Six distinct, webcam-driven magic styles based on the supplied references.

    These are procedural real-time VFX approximations, not original movie assets.
    """

    def __init__(self):
        self.history = {}

    def _remember(self, key, point, size=12):
        trail = self.history.setdefault(key, deque(maxlen=size))
        trail.append(point)
        return list(trail)

    def _render_strange(self, overlay, palm, elapsed, power):
        radius = int(78 + 20 * power)
        mandala(overlay, palm, radius, (0, 155, 255), (80, 220, 255), elapsed)
        # Sparse sparks on the circumference only.
        for i in range(8):
            a = elapsed * 0.22 + i * math.tau / 8
            q = (int(palm[0] + math.cos(a) * radius * 1.08),
                 int(palm[1] + math.sin(a) * radius * 1.08))
            cv2.circle(overlay, q, 2, (90, 225, 255), -1, cv2.LINE_AA)
        glow(overlay, palm, 7, (0, 165, 255), (255, 245, 220))

    def _render_wanda(self, overlay, palm, tips, elapsed, open_hand):
        # Scarlet Witch: a turbulent crimson core with rotating chaos sigils.
        pulse = 1.0 + 0.22 * math.sin(elapsed * 8.0)
        glow(overlay, palm, int(24 * pulse), (0, 0, 245), (255, 225, 245))
        cv2.circle(overlay, palm, 34, (0, 0, 205), 2, cv2.LINE_AA)
        cv2.ellipse(overlay, palm, (49, 31), elapsed * 34, 18, 315,
                    (0, 0, 245), 2, cv2.LINE_AA)
        cv2.ellipse(overlay, palm, (57, 40), -elapsed * 27, 205, 500,
                    (35, 25, 220), 2, cv2.LINE_AA)

        # Red tendrils flare outward, then curl back toward the palm.
        active = tips[1:] if open_hand else [tips[1]]
        for i, tip in enumerate(active):
            energy_wisp(overlay, palm, tip, (0, 0, 245),
                        elapsed + i * 0.12, seed=40 + i, width=4)
        for i in range(7):
            angle = elapsed * 0.9 + i * math.tau / 7
            end = (int(palm[0] + math.cos(angle) * 62),
                   int(palm[1] + math.sin(angle) * 47))
            energy_wisp(overlay, palm, end, (0, 0, 205),
                        elapsed + i * 0.08, seed=120 + i, width=2)
        for i in range(12):
            angle = elapsed * 0.45 + i * math.tau / 12
            x = int(palm[0] + math.cos(angle) * 68)
            y = int(palm[1] + math.sin(angle) * 53)
            cv2.circle(overlay, (x, y), 2, (65, 55, 255), -1, cv2.LINE_AA)

    def _render_agatha(self, overlay, palm, tips, elapsed, open_hand):
        # Agatha: deep violet witchfire, layered coils and a dark-purple corona.
        pulse = 1.0 + 0.15 * math.sin(elapsed * 5.5)
        glow(overlay, palm, int(20 * pulse), (175, 25, 205), (255, 220, 255))
        cv2.ellipse(overlay, palm, (38, 25), elapsed * 25, 0, 330,
                    (205, 45, 235), 3, cv2.LINE_AA)
        cv2.ellipse(overlay, palm, (52, 37), -elapsed * 19, 30, 350,
                    (145, 25, 195), 2, cv2.LINE_AA)
        for i, tip in enumerate(tips[1:]):
            if open_hand or i == 0:
                energy_wisp(overlay, palm, tip, (190, 35, 225),
                            elapsed + i * 0.18, seed=220 + i, width=3)
        # Separate floating sparks create a smoky, spell-casting halo.
        for i in range(9):
            angle = elapsed * 0.55 + i * math.tau / 9
            radius_x = 58 + 7 * math.sin(elapsed * 2 + i)
            radius_y = 43 + 5 * math.cos(elapsed * 2.3 + i)
            q = (int(palm[0] + math.cos(angle) * radius_x),
                 int(palm[1] + math.sin(angle) * radius_y))
            cv2.circle(overlay, q, 2 + (i % 2), (225, 95, 255), -1, cv2.LINE_AA)
        glow(overlay, palm, 5, (205, 45, 245), (255, 245, 255))

    def _render_loki(self, overlay, palm, tips, elapsed, open_hand):
        # Loki: elegant emerald illusion ribbons, not generic lightning bolts.
        glow(overlay, palm, 16, (35, 175, 45), (220, 255, 220))
        cv2.ellipse(overlay, palm, (43, 28), elapsed * 32, 25, 320,
                    (45, 210, 70), 2, cv2.LINE_AA)
        cv2.ellipse(overlay, palm, (58, 39), -elapsed * 23, 70, 355,
                    (25, 145, 45), 2, cv2.LINE_AA)
        selected = tips[1:] if open_hand else [tips[1]]
        for i, tip in enumerate(selected):
            energy_wisp(overlay, palm, tip, (45, 220, 70),
                        elapsed + i * 0.16, seed=320 + i, width=3)
        # Small rotating diamond sigils reinforce the illusion/mischief look.
        for i in range(4):
            angle = elapsed * 0.7 + i * math.tau / 4
            cx = int(palm[0] + math.cos(angle) * 54)
            cy = int(palm[1] + math.sin(angle) * 40)
            r = 7
            diamond = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
            poly(overlay, diamond, (75, 235, 100), 2, True)
        for i in range(5):
            angle = elapsed * 0.6 + i * math.tau / 5
            end = (int(palm[0] + math.cos(angle) * 66),
                   int(palm[1] + math.sin(angle) * 48))
            energy_wisp(overlay, palm, end, (30, 155, 45),
                        elapsed + i * 0.1, seed=360 + i, width=2)

    def _render_thor(self, overlay, palm, tips, elapsed, open_hand, fist):
        # Thor: a concentrated storm core with branching blue-white lightning.
        pulse = 1.0 + 0.18 * math.sin(elapsed * 10.0)
        glow(overlay, palm, int(19 * pulse), (255, 170, 35), (255, 255, 255))
        selected = tips if open_hand else [tips[1]]
        for i, tip in enumerate(selected):
            electric_bolt(overlay, palm, tip, (255, 205, 65),
                          int(elapsed * 28) + i * 37, 22, 3, 4)
        # A dominant upward strike and side forks make the effect read as thunder.
        targets = [
            (palm[0] - 18, max(8, palm[1] - 150)),
            (palm[0] + 8, max(8, palm[1] - 185)),
            (palm[0] + 38, max(8, palm[1] - 135)),
        ]
        for i, end in enumerate(targets):
            electric_bolt(overlay, palm, end, (255, 225, 125),
                          int(elapsed * 31) + 520 + i * 19, 27, 3, 4)
        # Short electric arcs orbit the wrist/palm between larger strikes.
        for i in range(3):
            angle = elapsed * 1.4 + i * math.tau / 3
            end = (int(palm[0] + math.cos(angle) * 57),
                   int(palm[1] + math.sin(angle) * 42))
            electric_bolt(overlay, palm, end, (255, 185, 65),
                          int(elapsed * 24) + 610 + i, 10, 1, 2)

    def _render_thanos(self, overlay, palm, tips, elapsed, open_hand, points, width, height):
        """Purple cosmic energy and an Infinity-Gauntlet-inspired hand overlay."""
        pulse = 1.0 + 0.12 * math.sin(elapsed * 4.5)
        glow(overlay, palm, int(22 * pulse), (155, 65, 205), (235, 205, 255))

        # Six Infinity-Stone coloured lights orbit the palm.
        stone_colors = [
            (30, 30, 235), (30, 190, 245), (210, 55, 190),
            (55, 210, 70), (220, 150, 35), (190, 80, 220)
        ]
        for i, color in enumerate(stone_colors):
            angle = elapsed * 0.45 + i * math.tau / 6
            x = int(palm[0] + math.cos(angle) * 43)
            y = int(palm[1] + math.sin(angle) * 34)
            glow(overlay, (x, y), 5, color, (255, 255, 255))

        # Purple energy filaments extend from the palm toward the fingers.
        selected = tips[1:] if open_hand else [tips[1]]
        for i, tip in enumerate(selected):
            energy_wisp(overlay, palm, tip, (165, 75, 225),
                        elapsed + i * 0.12, seed=820 + i, width=3)

        # A gold cuff at the wrist gives the hand a gauntlet-like silhouette.
        wrist = to_px(points[0, :2], width, height)
        cv2.circle(overlay, wrist, 15, (35, 175, 235), 4, cv2.LINE_AA)
        cv2.circle(overlay, wrist, 10, (80, 215, 255), 2, cv2.LINE_AA)
        glow(overlay, palm, 5, (175, 80, 225), (255, 245, 255))

    def render_hand(self, frame, character: Character, hand, elapsed, hand_id=0):
        height, width = frame.shape[:2]
        points = hand["points"]
        palm = to_px(palm_center(points), width, height)
        tips = [to_px(points[i, :2], width, height) for i in (4, 8, 12, 16, 20)]
        open_hand = hand.get("gesture") == "Open_Palm" or hand_is_open(points)
        fist = hand.get("gesture") == "Closed_Fist"
        pinch = is_pinching(points)
        power = 1.0 if open_hand else (0.82 if fist else 0.68)
        overlay = np.zeros_like(frame)

        name = character.name
        if name == "DOCTOR STRANGE":
            self._render_strange(overlay, palm, elapsed, power)
        elif name == "SCARLET WITCH":
            self._render_wanda(overlay, palm, tips, elapsed, open_hand)
        elif name == "AGATHA HARKNESS":
            self._render_agatha(overlay, palm, tips, elapsed, open_hand)
        elif name == "LOKI":
            self._render_loki(overlay, palm, tips, elapsed, open_hand)
        elif name == "THOR":
            self._render_thor(overlay, palm, tips, elapsed, open_hand, fist)
        elif name == "THANOS":
            self._render_thanos(overlay, palm, tips, elapsed, open_hand, points, width, height)

        # Pinch focuses the existing effect at the fingertips; it does not add
        # another generic ring around the hand.
        if pinch:
            focus = ((tips[0][0] + tips[1][0]) // 2,
                     (tips[0][1] + tips[1][1]) // 2)
            glow(overlay, focus, 5, character.accent, (255, 255, 255))

        bloom_composite(frame, overlay, blur=12, bloom_strength=0.72,
                        detail_strength=0.92)

    def render_two_hand_field(self, frame, character: Character, hands, elapsed):
        if len(hands) < 2:
            return

        height, width = frame.shape[:2]
        p1, p2 = hands[0]["points"], hands[1]["points"]
        c1 = to_px(palm_center(p1), width, height)
        c2 = to_px(palm_center(p2), width, height)
        midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
        distance = math.dist(c1, c2)
        overlay = np.zeros_like(frame)
        name = character.name

        if name == "AGATHA HARKNESS":
            # Main purple beam between both palms, with a luminous core and
            # two restrained flowing edge filaments.
            cv2.line(overlay, c1, c2, (220, 35, 245), 23, cv2.LINE_AA)
            cv2.line(overlay, c1, c2, (245, 120, 255), 10, cv2.LINE_AA)
            cv2.line(overlay, c1, c2, (255, 245, 255), 3, cv2.LINE_AA)
            energy_wisp(overlay, c1, c2, (230, 65, 255), elapsed, 701, 3)
            energy_wisp(overlay, c2, c1, (210, 45, 245), elapsed + 0.5, 702, 2)
            glow(overlay, c1, 13, (220, 35, 245))
            glow(overlay, c2, 13, (220, 35, 245))

        elif name == "SCARLET WITCH":
            # Red energy surges between the hands as a turbulent crimson stream.
            cv2.line(overlay, c1, c2, (0, 0, 190), 19, cv2.LINE_AA)
            energy_wisp(overlay, c1, c2, (0, 0, 245), elapsed, 711, 5)
            energy_wisp(overlay, c2, c1, (0, 0, 220), elapsed + 0.4, 712, 3)
            glow(overlay, c1, 16, (0, 0, 245))
            glow(overlay, c2, 16, (0, 0, 245))

        elif name == "THOR":
            # Lightning arcs leap through the space between the hands.
            for i in range(5):
                electric_bolt(overlay, c1, c2, (255, 215, 95),
                              int(elapsed * 24) + 800 + i * 17,
                              25 + i * 3, 2, 3 if i == 0 else 2)
            glow(overlay, c1, 13, (255, 195, 65))
            glow(overlay, c2, 13, (255, 195, 65))

        elif name == "LOKI":
            for i in range(3):
                electric_bolt(overlay, c1, c2, (45, 210, 65),
                              int(elapsed * 17) + 900 + i * 23,
                              19 + i * 4, 2, 2)
            glow(overlay, midpoint, 8, (45, 220, 65))

        elif name == "DOCTOR STRANGE":
            radius = int(np.clip(distance * 0.48, 55, min(width, height) * 0.34))
            mandala(overlay, midpoint, radius, (0, 155, 255),
                    (80, 220, 255), elapsed)

        elif name == "THANOS":
            for i, color in enumerate(((155, 65, 205), (80, 120, 235), (35, 190, 225))):
                energy_wisp(overlay, c1, c2, color, elapsed + i * 0.22, 850 + i, 3)
            glow(overlay, midpoint, 18, (155, 65, 205), (245, 220, 255))

        bloom_composite(frame, overlay, blur=16, bloom_strength=0.85,
                        detail_strength=0.90)
