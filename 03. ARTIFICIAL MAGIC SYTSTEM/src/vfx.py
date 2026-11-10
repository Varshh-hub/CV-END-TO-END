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


def mandala(layer, center, radius, color, accent, t, ancient=False):
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

    if ancient:
        # Additional fine petal geometry, characteristic of the Ancient One's
        # layered golden-green spell language.
        for i in range(12):
            a = math.tau * i / 12 + t * 0.02
            p1 = (int(cx + math.cos(a) * radius * 0.26),
                  int(cy + math.sin(a) * radius * 0.26))
            p2 = (int(cx + math.cos(a + 0.20) * radius * 0.57),
                  int(cy + math.sin(a + 0.20) * radius * 0.57))
            p3 = (int(cx + math.cos(a + 0.40) * radius * 0.26),
                  int(cy + math.sin(a + 0.40) * radius * 0.26))
            poly(layer, (p1, p2, p3), accent, 1)


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
    """Seven distinct, webcam-driven magic styles based on the supplied references.

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
        # Crimson source orb and diffuse red aura.
        pulse = 1.0 + 0.16 * math.sin(elapsed * 7.0)
        glow(overlay, palm, int(19 * pulse), (0, 0, 245), (255, 235, 245))
        cv2.circle(overlay, palm, 31, (0, 0, 190), 2, cv2.LINE_AA)
        # Short, turbulent tendrils emerging from the palm/fingertips.
        active = tips[1:] if open_hand else [tips[1]]
        for i, tip in enumerate(active):
            energy_wisp(overlay, palm, tip, (0, 0, 235), elapsed,
                        seed=40 + i, width=3)
        # Red wisps curl around the hand instead of drawing a web of lines.
        for i in range(5):
            a = elapsed * 0.75 + i * math.tau / 5
            end = (int(palm[0] + math.cos(a) * 48),
                   int(palm[1] + math.sin(a) * 38))
            energy_wisp(overlay, palm, end, (0, 0, 185), elapsed,
                        seed=100 + i, width=2)

    def _render_agatha(self, overlay, palm, tips, elapsed, open_hand):
        # Purple energy gathers in the palm and coils around the fingertips.
        glow(overlay, palm, 16, (220, 35, 235), (255, 245, 255))
        for i, tip in enumerate(tips[1:]):
            if open_hand or i == 0:
                energy_wisp(overlay, palm, tip, (225, 45, 245),
                            elapsed + i * 0.2, seed=200 + i, width=2)
        # Small purple corona around the palm; no generic circular shield.
        for i in range(3):
            a = elapsed * 0.7 + i * math.tau / 3
            q = (int(palm[0] + math.cos(a) * 27),
                 int(palm[1] + math.sin(a) * 23))
            cv2.circle(overlay, q, 4, (245, 90, 255), -1, cv2.LINE_AA)

    def _render_loki(self, overlay, palm, tips, elapsed, open_hand):
        # Emerald electrical discharge; irregular bolts radiate from the palm.
        glow(overlay, palm, 13, (35, 205, 45), (230, 255, 225))
        selected = tips[1:] if open_hand else [tips[1]]
        for i, tip in enumerate(selected):
            electric_bolt(overlay, palm, tip, (35, 225, 65),
                          int(elapsed * 18) + i * 13, 13, 2, 3)
        # A few detached arcs add the green electrical atmosphere.
        for i in range(4):
            a = elapsed * 0.9 + i * math.tau / 4
            end = (int(palm[0] + math.cos(a) * 52),
                   int(palm[1] + math.sin(a) * 44))
            electric_bolt(overlay, palm, end, (55, 190, 35),
                          int(elapsed * 15) + 300 + i, 9, 1, 2)

    def _render_thor(self, overlay, palm, tips, elapsed, open_hand, fist):
        # Cold blue-white lightning, with a bright central discharge.
        glow(overlay, palm, 16 if fist else 11, (255, 175, 35), (255, 255, 255))
        selected = tips if open_hand else [tips[1]]
        for i, tip in enumerate(selected):
            electric_bolt(overlay, palm, tip, (255, 205, 70),
                          int(elapsed * 22) + i * 31, 19, 3, 3)
        # Upward forks suggest the large storm-lightning look in the reference.
        for i in range(3):
            end = (palm[0] + (i - 1) * 35, max(8, palm[1] - 115 - i * 17))
            electric_bolt(overlay, palm, end, (255, 225, 145),
                          int(elapsed * 20) + 500 + i, 22, 2, 3)

    def _render_wong(self, overlay, palm, elapsed, power):
        # Distinct orange-gold circular shield with segmented outer rim.
        radius = int(76 + 18 * power)
        mandala(overlay, palm, radius, (0, 135, 255), (60, 205, 255), elapsed)
        # Stronger segmented shield border.
        for i in range(16):
            start = int(i * 22.5 + elapsed * 9)
            cv2.ellipse(overlay, palm, (radius + 9, radius + 9), 0,
                        start, start + 13, (0, 175, 255), 3, cv2.LINE_AA)
        glow(overlay, palm, 5, (0, 150, 255), (255, 250, 225))

    def _render_ancient_one(self, overlay, palm, elapsed, power):
        radius = int(83 + 15 * power)
        mandala(overlay, palm, radius, (70, 190, 130), (175, 245, 190),
                elapsed, ancient=True)
        # Delicate luminous inner spokes, not long finger connections.
        for i in range(12):
            a = elapsed * 0.08 + i * math.tau / 12
            p1 = (int(palm[0] + math.cos(a) * radius * 0.18),
                  int(palm[1] + math.sin(a) * radius * 0.18))
            p2 = (int(palm[0] + math.cos(a) * radius * 0.34),
                  int(palm[1] + math.sin(a) * radius * 0.34))
            cv2.line(overlay, p1, p2, (190, 255, 205), 2, cv2.LINE_AA)
        glow(overlay, palm, 6, (90, 205, 145), (240, 255, 235))

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
        elif name == "WONG":
            self._render_wong(overlay, palm, elapsed, power)
        elif name == "THE ANCIENT ONE":
            self._render_ancient_one(overlay, palm, elapsed, power)

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

        elif name in ("DOCTOR STRANGE", "WONG", "THE ANCIENT ONE"):
            radius = int(np.clip(distance * 0.48, 55, min(width, height) * 0.34))
            if name == "DOCTOR STRANGE":
                mandala(overlay, midpoint, radius, (0, 155, 255),
                        (80, 220, 255), elapsed)
            elif name == "WONG":
                mandala(overlay, midpoint, radius, (0, 135, 255),
                        (60, 205, 255), elapsed)
            else:
                mandala(overlay, midpoint, radius, (70, 190, 130),
                        (175, 245, 190), elapsed, ancient=True)

        bloom_composite(frame, overlay, blur=16, bloom_strength=0.85,
                        detail_strength=0.90)
