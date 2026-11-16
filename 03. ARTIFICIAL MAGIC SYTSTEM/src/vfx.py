import math
import random

import cv2
import numpy as np

from config import Character


# ============================================================
# 1. PALM UTILITIES
# ============================================================

def palm_center(points):
    """Calculate palm centre using wrist and MCP landmarks."""
    return np.mean(points[[0, 5, 9, 13, 17], :2], axis=0)


def to_px(point, width, height):
    """Convert normalized MediaPipe coordinates to pixels."""
    return (
        int(np.clip(point[0] * width, 0, width - 1)),
        int(np.clip(point[1] * height, 0, height - 1)),
    )


def palm_normal(points):
    """
    Estimate palm orientation from 3D landmarks.
    This is an approximate orientation estimate.
    """
    wrist = points[0, :3]
    index_base = points[5, :3]
    pinky_base = points[17, :3]

    v1 = index_base - wrist
    v2 = pinky_base - wrist

    normal = np.cross(v1, v2)
    length = np.linalg.norm(normal)

    if length < 1e-7:
        return None

    return normal / length


def facing_each_other(hand_a, hand_b):
    """
    Estimate whether two palms face one another.

    MediaPipe's normalized 3D landmarks are approximate,
    so this condition may need tuning for different camera angles.
    """
    points_a = hand_a["points"]
    points_b = hand_b["points"]

    normal_a = palm_normal(points_a)
    normal_b = palm_normal(points_b)

    if normal_a is None or normal_b is None:
        return False

    center_a = palm_center(points_a)
    center_b = palm_center(points_b)

    center_a_3d = np.array([
        center_a[0],
        center_a[1],
        np.mean(points_a[[0, 5, 9, 13, 17], 2])
    ], dtype=np.float32)

    center_b_3d = np.array([
        center_b[0],
        center_b[1],
        np.mean(points_b[[0, 5, 9, 13, 17], 2])
    ], dtype=np.float32)

    direction_ab = center_b_3d - center_a_3d
    distance = np.linalg.norm(direction_ab)

    if distance < 1e-6:
        return False

    direction_ab /= distance
    direction_ba = -direction_ab

    alignment_a = float(np.dot(normal_a, direction_ab))
    alignment_b = float(np.dot(normal_b, direction_ba))

    return alignment_a > 0.25 and alignment_b > 0.25


def palms_face_camera(hand_a, hand_b, threshold=0.55):
    """Approximate whether both palms are presented toward the camera."""
    normal_a = palm_normal(hand_a["points"])
    normal_b = palm_normal(hand_b["points"])
    if normal_a is None or normal_b is None:
        return False
    # In MediaPipe camera coordinates, a strong z component approximates
    # a palm plane directed toward the camera. This is a heuristic.
    return abs(float(normal_a[2])) > threshold and abs(float(normal_b[2])) > threshold


# ============================================================
# 2. GLOW AND COMPOSITING
# ============================================================

def glow(layer, center, radius, color, core=(245, 248, 255)):
    x, y = center
    radius = max(2, int(radius))

    cv2.circle(
        layer, (x, y), max(3, int(radius * 2.8)),
        color, -1, cv2.LINE_AA
    )

    cv2.circle(
        layer, (x, y), radius,
        core, -1, cv2.LINE_AA
    )


def bloom_composite(
    frame,
    overlay,
    blur=15,
    bloom_strength=0.9,
    detail_strength=0.95
):
    soft = cv2.GaussianBlur(overlay, (0, 0), blur)

    frame[:] = cv2.addWeighted(
        frame, 1.0, soft, bloom_strength, 0
    )

    frame[:] = cv2.addWeighted(
        frame, 1.0, overlay, detail_strength, 0
    )


def ring(
    layer,
    center,
    radius,
    color,
    thickness=2,
    start=0,
    end=350
):
    cv2.ellipse(
        layer,
        center,
        (max(2, int(radius)), max(2, int(radius))),
        0,
        start,
        end,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 3. DOCTOR STRANGE MANDALA
# ============================================================

def mandala(layer, center, radius, color, accent, t, dense=1.0):
    """Rotating sorcerer mandala centred on the palm."""

    cx, cy = center
    radius = max(25, int(radius))

    scales = (1.0, 0.91, 0.79, 0.66, 0.53, 0.39, 0.25)

    for i, scale in enumerate(scales):
        r = int(radius * scale)

        start = int(
            (t * (22 if i % 2 else -17) + i * 19) % 360
        )

        ring(
            layer,
            center,
            r,
            color if i % 2 == 0 else accent,
            3 if i < 2 else 1,
            start,
            start + 342
        )

    count = int(48 * dense)

    for i in range(count):
        a = math.tau * i / count + t * 0.035
        r = radius * (0.82 if i % 4 else 0.72)

        x = int(cx + math.cos(a) * r)
        y = int(cy + math.sin(a) * r)

        length = 5 if i % 3 else 12

        tx = -math.sin(a)
        ty = math.cos(a)

        cv2.line(
            layer,
            (int(x - tx * length), int(y - ty * length)),
            (int(x + tx * length), int(y + ty * length)),
            accent if i % 4 == 0 else color,
            1,
            cv2.LINE_AA
        )

    for sides, scale, rotation in (
        (8, 0.58, 22.5),
        (4, 0.40, 45),
        (6, 0.29, 0)
    ):
        pts = []

        for i in range(sides):
            a = (
                math.tau * i / sides
                + math.radians(rotation)
                + t * 0.025
            )

            pts.append((
                int(cx + math.cos(a) * radius * scale),
                int(cy + math.sin(a) * radius * scale)
            ))

        cv2.polylines(
            layer,
            [np.asarray(pts, np.int32)],
            True,
            accent,
            2,
            cv2.LINE_AA
        )

    cv2.circle(
        layer, center, max(5, int(radius * 0.12)),
        accent, 2, cv2.LINE_AA
    )

    cv2.circle(
        layer, center, max(3, int(radius * 0.055)),
        (255, 255, 255), -1, cv2.LINE_AA
    )


# ============================================================
# 4. ENERGY PARTICLES AND PALM ARCS
# ============================================================

def particle_cloud(
    layer,
    center,
    radius,
    color,
    accent,
    t,
    count=100,
    seed=0
):
    """Dense animated particles around the palm."""

    cx, cy = center
    rng = random.Random(int(t * 15) + seed)

    for i in range(count):
        a = (
            rng.random() * math.tau
            + t * (0.35 + (i % 4) * 0.08)
        )

        r = radius * (0.25 + 0.85 * rng.random())

        x = int(cx + math.cos(a) * r)
        y = int(cy + math.sin(a) * r * 0.78)

        cv2.circle(
            layer,
            (x, y),
            rng.choice((1, 1, 2, 3)),
            accent if i % 5 == 0 else color,
            -1,
            cv2.LINE_AA
        )


def palm_arcs(
    layer,
    center,
    radius,
    color,
    accent,
    t,
    count=9
):
    """Broken rotating energy arcs around the palm."""

    for i in range(count):
        start = int(
            (t * 80 + i * 360 / count) % 360
        )

        extent = 22 + (i % 3) * 12
        r = radius * (0.72 + 0.28 * (i % 2))

        cv2.ellipse(
            layer,
            center,
            (max(2, int(r)), max(2, int(r * 0.78))),
            0,
            start,
            start + extent,
            color if i % 2 else accent,
            2 if i % 3 else 3,
            cv2.LINE_AA
        )


# ============================================================
# 5. MAGIC RENDERER
# ============================================================

class MagicRenderer:
    """
    Six palm-centred procedural magic styles.

    Character effects:
    1. Doctor Strange - orange sorcerer mandalas
    2. Scarlet Witch - red chaos energy
    3. Agatha - purple witchfire
    4. Loki - green illusion energy
    5. Thor - blue-white lightning
    6. Thanos - purple cosmic energy

    Effects are automatic when hands are detected.
    No finger skeletons or landmark lines are drawn.
    """

    def __init__(self):
        self.motion = {}

    # --------------------------------------------------------
    # MOTION ADAPTIVE SCALE
    # --------------------------------------------------------

    def _motion_scale(self, points, width, height, hand_id):

        center = palm_center(points)

        center_px = np.array(
            [center[0] * width, center[1] * height],
            dtype=np.float32
        )

        palm_width = (
            np.linalg.norm(points[5, :2] - points[17, :2])
            * width
        )

        palm_height = (
            np.linalg.norm(points[0, :2] - points[9, :2])
            * height
        )

        hand_size = max(
            18.0,
            (palm_width + palm_height) * 0.5
        )

        old = self.motion.get(hand_id)

        if old is None:
            smooth_speed = 0.0

        else:
            old_center, old_time, old_speed = old

            dt = max(
                1 / 120,
                min(0.15, self._now - old_time)
            )

            speed = float(
                np.linalg.norm(center_px - old_center) / dt
            )

            smooth_speed = (
                0.72 * old_speed + 0.28 * speed
            )

        self.motion[hand_id] = (
            center_px,
            self._now,
            smooth_speed
        )

        movement_boost = float(
            np.clip(smooth_speed / 900.0, 0.0, 0.75)
        )

        # Apparent palm size changes as the hand moves toward/away from
        # the camera. Keep a wider useful range than the original so this
        # perspective change is visible in the spell size.
        base = float(
            np.clip(hand_size * 1.08, 28.0, 145.0)
        )

        # Movement adds a modest energetic expansion, while the landmark
        # scale itself handles forward/backward movement.
        dynamic_scale = 1.0 + movement_boost * 0.85

        return (
            int(np.clip(base * dynamic_scale, 28.0, 190.0)),
            movement_boost
        )

    # --------------------------------------------------------
    # RENDER MAGIC ON ONE PALM
    # --------------------------------------------------------

    def render_hand(
        self,
        frame,
        character: Character,
        hand,
        elapsed,
        hand_id=0
    ):

        h, w = frame.shape[:2]

        points = hand["points"]

        palm = to_px(
            palm_center(points),
            w,
            h
        )

        self._now = elapsed

        radius, motion_boost = self._motion_scale(
            points, w, h, hand_id
        )

        overlay = np.zeros_like(frame)

        name = character.name

        pulse = 1.0 + 0.08 * math.sin(elapsed * 7.0)

        r = int(radius * pulse)

        # ----------------------------------------------------
        # DOCTOR STRANGE
        # ----------------------------------------------------

        if name == "DOCTOR STRANGE":

            mandala(
                overlay, palm, int(r * 1.30),
                (0, 155, 255), (80, 220, 255),
                elapsed, 1.25
            )

            palm_arcs(
                overlay, palm, r * 0.83,
                (0, 175, 255), (100, 235, 255),
                elapsed, 12
            )

            particle_cloud(
                overlay, palm, r * 0.85,
                (0, 150, 255), (130, 235, 255),
                elapsed, 65, 1
            )

            glow(
                overlay, palm, 9,
                (0, 175, 255), (255, 250, 225)
            )

        # ----------------------------------------------------
        # SCARLET WITCH
        # ----------------------------------------------------

        elif name == "SCARLET WITCH":

            glow(
                overlay, palm, int(r * 0.82),
                (0, 0, 205), (255, 245, 255)
            )

            for i in range(7):

                a = (
                    elapsed * (1.8 if i % 2 else -1.4)
                    + i * math.tau / 7
                )

                axes = (
                    int(r * (0.72 + i * 0.045)),
                    int(r * (0.48 + i * 0.035))
                )

                cv2.ellipse(
                    overlay, palm, axes,
                    math.degrees(a), 12, 338,
                    (0, 0, 190 + (i % 2) * 45),
                    3, cv2.LINE_AA
                )

            palm_arcs(
                overlay, palm, r * 0.95,
                (0, 0, 220), (95, 70, 255),
                elapsed, 14
            )

            particle_cloud(
                overlay, palm, r * 1.15,
                (0, 0, 220), (100, 80, 255),
                elapsed, 145, 2
            )

            glow(
                overlay, palm, int(r * 0.22),
                (0, 0, 245), (255, 255, 255)
            )

        # ----------------------------------------------------
        # AGATHA HARKNESS
        # ----------------------------------------------------

        elif name == "AGATHA HARKNESS":

            glow(
                overlay, palm, int(r * 0.88),
                (175, 20, 205), (255, 235, 255)
            )

            for i in range(9):

                a = (
                    elapsed * (1.2 if i % 2 else -1.5)
                    + i * 20
                )

                cv2.ellipse(
                    overlay,
                    palm,
                    (
                        int(r * (0.62 + i * 0.055)),
                        int(r * (0.42 + i * 0.045))
                    ),
                    math.degrees(a),
                    0, 335,
                    (195, 35 + i % 2 * 25, 235),
                    3,
                    cv2.LINE_AA
                )

            palm_arcs(
                overlay, palm, r * 0.95,
                (185, 25, 220), (245, 135, 255),
                elapsed, 16
            )

            particle_cloud(
                overlay, palm, r * 1.18,
                (185, 25, 220), (245, 150, 255),
                elapsed, 155, 3
            )

            glow(
                overlay, palm, int(r * 0.24),
                (205, 45, 240), (255, 255, 255)
            )

        # ----------------------------------------------------
        # LOKI
        # ----------------------------------------------------

        elif name == "LOKI":

            glow(
                overlay, palm, int(r * 0.75),
                (25, 145, 20), (220, 255, 220)
            )

            for i in range(8):

                angle = (
                    elapsed * (1.35 if i % 2 else -1.1)
                    + i * 23
                )

                cv2.ellipse(
                    overlay,
                    palm,
                    (
                        int(r * (0.58 + i * 0.06)),
                        int(r * (0.35 + i * 0.045))
                    ),
                    math.degrees(angle),
                    25, 330,
                    (25 + i * 2, 145 + (i % 2) * 65, 25),
                    3,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, palm, r * 1.12,
                (30, 165, 25), (155, 255, 155),
                elapsed, 125, 4
            )

            palm_arcs(
                overlay, palm, r * 0.9,
                (30, 180, 35), (155, 255, 155),
                elapsed, 10
            )

            glow(
                overlay, palm, int(r * 0.20),
                (45, 195, 45), (255, 255, 255)
            )

        # ----------------------------------------------------
        # THOR
        # ----------------------------------------------------

        elif name == "THOR":

            glow(
                overlay, palm, int(r * 0.72),
                (255, 150, 30), (255, 255, 255)
            )

            palm_arcs(
                overlay, palm, r * 1.05,
                (255, 205, 65), (255, 250, 225),
                elapsed, 13
            )

            rng = random.Random(int(elapsed * 28))

            for i in range(12):

                a = elapsed * 0.9 + i * math.tau / 12

                reach = r * (0.65 + 0.4 * rng.random())

                end = (
                    int(palm[0] + math.cos(a) * reach),
                    int(palm[1] + math.sin(a) * reach * 0.78)
                )

                mid = (
                    (palm[0] + end[0]) // 2 + rng.randint(-8, 8),
                    (palm[1] + end[1]) // 2 + rng.randint(-8, 8)
                )

                cv2.polylines(
                    overlay,
                    [np.asarray([palm, mid, end], np.int32)],
                    False,
                    (255, 215, 105),
                    2,
                    cv2.LINE_AA
                )

                cv2.circle(
                    overlay, end, 2,
                    (255, 255, 255), -1, cv2.LINE_AA
                )

            particle_cloud(
                overlay, palm, r * 1.1,
                (255, 185, 45), (255, 255, 255),
                elapsed, 100, 5
            )

        # ----------------------------------------------------
        # THANOS
        # ----------------------------------------------------

        elif name == "THANOS":

            glow(
                overlay, palm, int(r * 0.95),
                (170, 35, 195), (255, 220, 255)
            )

            for i in range(9):

                angle = (
                    elapsed * (1.1 if i % 2 else -1.3)
                    + i * 23
                )

                cv2.ellipse(
                    overlay,
                    palm,
                    (
                        int(r * (0.55 + i * 0.065)),
                        int(r * (0.38 + i * 0.045))
                    ),
                    math.degrees(angle),
                    0, 340,
                    (180, 55 + (i % 2) * 35, 220),
                    2,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, palm, r * 1.25,
                (180, 45, 210), (255, 195, 130),
                elapsed, 180, 6
            )

            palm_arcs(
                overlay, palm, r * 1.0,
                (180, 50, 215), (255, 190, 100),
                elapsed, 16
            )

            glow(
                overlay, palm, int(r * 0.25),
                (190, 45, 220), (255, 255, 255)
            )

        # ----------------------------------------------------
        # COMPOSITE PALM EFFECT
        # ----------------------------------------------------

        bloom_composite(
            frame,
            overlay,
            blur=max(10, int(r * 0.18)),
            bloom_strength=0.95 + motion_boost * 0.12,
            detail_strength=0.92
        )

    # --------------------------------------------------------
    # POSE CLASSIFIERS
    # --------------------------------------------------------

    def palms_facing_each_other(self, hand_a, hand_b):
        return facing_each_other(hand_a, hand_b)

    def palms_facing_camera(self, hand_a, hand_b, threshold=0.55):
        normal_a = palm_normal(hand_a["points"])
        normal_b = palm_normal(hand_b["points"])
        if normal_a is None or normal_b is None:
            return False
        return (
            abs(float(normal_a[2])) > threshold
            and abs(float(normal_b[2])) > threshold
        )

    # ========================================================
    # 6. THIRD MAGIC FORMATION BETWEEN TWO FACING PALMS
    # ========================================================

    def render_two_hand_field(
        self,
        frame,
        character,
        hands,
        elapsed,
        force=False
    ):
        """
        Creates a third energy orb between two facing palms.

        Both individual palm effects are rendered separately
        by render_hand().

        The central orb scales with palm distance.
        """

        if len(hands) < 2:
            return

        hand_a = hands[0]
        hand_b = hands[1]

        if not force and not facing_each_other(hand_a, hand_b):
            return

        h, w = frame.shape[:2]

        center_a = palm_center(hand_a["points"])
        center_b = palm_center(hand_b["points"])

        a = np.array(
            [center_a[0] * w, center_a[1] * h],
            dtype=np.float32
        )

        b = np.array(
            [center_b[0] * w, center_b[1] * h],
            dtype=np.float32
        )

        midpoint = tuple(
            ((a + b) / 2).astype(int).tolist()
        )

        distance = float(np.linalg.norm(b - a))

        # Central formation responds to distance between palms.
        radius = int(
            np.clip(distance * 0.22, 30, 105)
        )

        pulse = 1.0 + 0.12 * math.sin(elapsed * 6.0)

        radius = int(radius * pulse)

        overlay = np.zeros_like(frame)

        name = character.name

        # ----------------------------------------------------
        # DOCTOR STRANGE CENTRAL MANDALA
        # ----------------------------------------------------

        if name == "DOCTOR STRANGE":

            mandala(
                overlay, midpoint, int(radius * 1.35),
                (0, 155, 255), (80, 220, 255),
                elapsed, 1.7
            )

            particle_cloud(
                overlay, midpoint, radius * 1.2,
                (0, 155, 255), (100, 230, 255),
                elapsed, 180, 20
            )

            glow(
                overlay, midpoint, int(radius * 0.30),
                (0, 170, 255), (255, 255, 255)
            )

        # ----------------------------------------------------
        # SCARLET WITCH CENTRAL CHAOS SPHERE
        # ----------------------------------------------------

        elif name == "SCARLET WITCH":

            glow(
                overlay, midpoint, radius,
                (0, 0, 210), (255, 245, 255)
            )

            for i in range(12):

                angle = (
                    elapsed * (1.4 if i % 2 else -1.5)
                    + i * 30
                )

                cv2.ellipse(
                    overlay,
                    midpoint,
                    (
                        int(radius * (0.65 + i * 0.045)),
                        int(radius * (0.45 + i * 0.035))
                    ),
                    math.degrees(angle),
                    0, 340,
                    (0, 0, 230),
                    3,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, midpoint, radius * 1.4,
                (0, 0, 220), (100, 70, 255),
                elapsed, 220, 21
            )

        # ----------------------------------------------------
        # AGATHA CENTRAL WITCHFIRE SPHERE
        # ----------------------------------------------------

        elif name == "AGATHA HARKNESS":

            glow(
                overlay, midpoint, radius,
                (175, 20, 205), (255, 235, 255)
            )

            for i in range(12):

                angle = (
                    elapsed * (1.2 if i % 2 else -1.4)
                    + i * 25
                )

                cv2.ellipse(
                    overlay,
                    midpoint,
                    (
                        int(radius * (0.60 + i * 0.05)),
                        int(radius * (0.40 + i * 0.04))
                    ),
                    math.degrees(angle),
                    0, 340,
                    (195, 35, 235),
                    3,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, midpoint, radius * 1.4,
                (185, 25, 220), (245, 145, 255),
                elapsed, 220, 22
            )

        # ----------------------------------------------------
        # LOKI CENTRAL ILLUSION SPHERE
        # ----------------------------------------------------

        elif name == "LOKI":

            glow(
                overlay, midpoint, radius,
                (25, 145, 20), (220, 255, 220)
            )

            for i in range(12):

                angle = (
                    elapsed * (1.3 if i % 2 else -1.1)
                    + i * 25
                )

                cv2.ellipse(
                    overlay,
                    midpoint,
                    (
                        int(radius * (0.60 + i * 0.05)),
                        int(radius * (0.40 + i * 0.04))
                    ),
                    math.degrees(angle),
                    0, 340,
                    (35, 175, 35),
                    3,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, midpoint, radius * 1.4,
                (30, 165, 25), (155, 255, 155),
                elapsed, 190, 23
            )

        # ----------------------------------------------------
        # THOR CENTRAL LIGHTNING SPHERE
        # ----------------------------------------------------

        elif name == "THOR":

            glow(
                overlay, midpoint, radius,
                (255, 165, 35), (255, 255, 255)
            )

            rng = random.Random(int(elapsed * 30))

            for i in range(18):

                angle = (
                    elapsed * 0.8 + i * math.tau / 18
                )

                reach = radius * (
                    0.65 + rng.random() * 0.45
                )

                end = (
                    int(midpoint[0] + math.cos(angle) * reach),
                    int(midpoint[1] + math.sin(angle) * reach)
                )

                mid = (
                    (midpoint[0] + end[0]) // 2
                    + rng.randint(-12, 12),

                    (midpoint[1] + end[1]) // 2
                    + rng.randint(-12, 12)
                )

                cv2.polylines(
                    overlay,
                    [np.asarray(
                        [midpoint, mid, end],
                        np.int32
                    )],
                    False,
                    (255, 225, 140),
                    2,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, midpoint, radius * 1.3,
                (255, 190, 50), (255, 255, 255),
                elapsed, 160, 24
            )

        # ----------------------------------------------------
        # THANOS CENTRAL COSMIC SPHERE
        # ----------------------------------------------------

        elif name == "THANOS":

            glow(
                overlay, midpoint, radius,
                (170, 35, 195), (255, 225, 255)
            )

            for i in range(12):

                angle = (
                    elapsed * (1.0 if i % 2 else -1.3)
                    + i * 20
                )

                cv2.ellipse(
                    overlay,
                    midpoint,
                    (
                        int(radius * (0.6 + i * 0.045)),
                        int(radius * (0.4 + i * 0.035))
                    ),
                    math.degrees(angle),
                    0, 340,
                    (180, 50 + (i % 2) * 30, 220),
                    2,
                    cv2.LINE_AA
                )

            particle_cloud(
                overlay, midpoint, radius * 1.5,
                (180, 45, 210), (255, 190, 100),
                elapsed, 240, 25
            )

            glow(
                overlay, midpoint, int(radius * 0.32),
                (190, 45, 220), (255, 255, 255)
            )

        # ----------------------------------------------------
        # COMPOSITE CENTRAL ENERGY FORMATION
        # ----------------------------------------------------

        bloom_composite(
            frame,
            overlay,
            blur=max(12, int(radius * 0.25)),
            bloom_strength=1.15,
            detail_strength=1.0
        )
