import math
import random
from collections import deque

import cv2
import numpy as np

from config import Character, BLOOM_BLUR, BLOOM_STRENGTH, DETAIL_STRENGTH


def palm_center(points):
    """Palm anchor from wrist and four MCP landmarks; fingers are not rendered."""
    return np.mean(points[[0, 5, 9, 13, 17], :2], axis=0)


def to_px(point, width, height):
    return (
        int(np.clip(point[0] * width, 0, width - 1)),
        int(np.clip(point[1] * height, 0, height - 1)),
    )


def glow(layer, center, radius, color, core=(245, 248, 255)):
    x, y = center
    radius = max(2, int(radius))
    cv2.circle(layer, (x, y), max(3, int(radius * 2.8)), color, -1, cv2.LINE_AA)
    cv2.circle(layer, (x, y), radius, core, -1, cv2.LINE_AA)


def bloom_composite(frame, overlay, blur=15, bloom_strength=0.9, detail_strength=0.95):
    soft = cv2.GaussianBlur(overlay, (0, 0), blur)
    frame[:] = cv2.addWeighted(frame, 1.0, soft, bloom_strength, 0)
    frame[:] = cv2.addWeighted(frame, 1.0, overlay, detail_strength, 0)


def ring(layer, center, radius, color, thickness=2, start=0, end=350):
    cv2.ellipse(layer, center, (max(2, int(radius)), max(2, int(radius))),
                0, start, end, color, thickness, cv2.LINE_AA)


def mandala(layer, center, radius, color, accent, t, dense=1.0):
    """Palm-centred sorcerer mandala. No landmarks or lines are drawn on fingers."""
    cx, cy = center
    radius = max(25, int(radius))
    scales = (1.0, .91, .79, .66, .53, .39, .25)
    for i, scale in enumerate(scales):
        r = int(radius * scale)
        start = int((t * (22 if i % 2 else -17) + i * 19) % 360)
        ring(layer, center, r, color if i % 2 == 0 else accent,
             3 if i < 2 else 1, start, start + 342)
    count = int(48 * dense)
    for i in range(count):
        a = math.tau * i / count + t * .035
        r = radius * (.82 if i % 4 else .72)
        x, y = int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)
        length = 5 if i % 3 else 12
        tx, ty = -math.sin(a), math.cos(a)
        cv2.line(layer, (int(x-tx*length), int(y-ty*length)),
                 (int(x+tx*length), int(y+ty*length)),
                 accent if i % 4 == 0 else color, 1, cv2.LINE_AA)
    for sides, scale, rotation in ((8, .58, 22.5), (4, .40, 45), (6, .29, 0)):
        pts = []
        for i in range(sides):
            a = math.tau * i / sides + math.radians(rotation) + t * .025
            pts.append((int(cx + math.cos(a)*radius*scale),
                        int(cy + math.sin(a)*radius*scale)))
        cv2.polylines(layer, [np.asarray(pts, np.int32)], True, accent, 2, cv2.LINE_AA)
    cv2.circle(layer, center, max(5, int(radius*.12)), accent, 2, cv2.LINE_AA)
    cv2.circle(layer, center, max(3, int(radius*.055)), (255,255,255), -1, cv2.LINE_AA)


def particle_cloud(layer, center, radius, color, accent, t, count=100, seed=0):
    """Dense floating energy particles contained around the palm."""
    cx, cy = center
    rng = random.Random(int(t * 15) + seed)
    for i in range(count):
        a = rng.random() * math.tau + t * (0.35 + (i % 4) * .08)
        r = radius * (0.25 + 0.85 * rng.random())
        x = int(cx + math.cos(a) * r)
        y = int(cy + math.sin(a) * r * .78)
        cv2.circle(layer, (x, y), rng.choice((1, 1, 2, 3)),
                   accent if i % 5 == 0 else color, -1, cv2.LINE_AA)


def palm_arcs(layer, center, radius, color, accent, t, count=9):
    """Short broken arcs around the palm only (never connected to fingertips)."""
    for i in range(count):
        start = int((t * 80 + i * 360 / count) % 360)
        extent = 22 + (i % 3) * 12
        r = radius * (0.72 + .28 * ((i % 2)))
        cv2.ellipse(layer, center, (int(r), int(r*.78)), 0,
                    start, start + extent, color if i % 2 else accent,
                    2 if i % 3 else 3, cv2.LINE_AA)


class MagicRenderer:
    """Seven palm-centred procedural VFX styles with motion-adaptive scale.

    MediaPipe landmarks estimate palm location, hand size and movement.
    They are not drawn. These are real-time visual approximations, not movie footage.
    """
    def __init__(self):
        self.motion = {}

    def _motion_scale(self, points, width, height, hand_id):
        center = palm_center(points)
        center_px = np.array([center[0] * width, center[1] * height], dtype=np.float32)
        # Palm width gives a scale that adapts to the hand's distance from camera.
        palm_width = np.linalg.norm(points[5, :2] - points[17, :2]) * width
        palm_height = np.linalg.norm(points[0, :2] - points[9, :2]) * height
        hand_size = max(18.0, (palm_width + palm_height) * .5)
        old = self.motion.get(hand_id)
        if old is None:
            speed = 0.0
            smooth_speed = 0.0
        else:
            old_center, old_time, old_speed = old
            dt = max(1/120, min(.15, self._now - old_time))
            speed = float(np.linalg.norm(center_px - old_center) / dt)
            smooth_speed = .72 * old_speed + .28 * speed
        self.motion[hand_id] = (center_px, self._now, smooth_speed)
        # Motion increases the energy radius; hand size controls the base radius.
        movement_boost = np.clip(smooth_speed / 900.0, 0.0, .75)
        base = np.clip(hand_size * .92, 32.0, 105.0)
        return int(base * (1.0 + movement_boost)), float(movement_boost)

    def render_hand(self, frame, character: Character, hand, elapsed, hand_id=0):
        h, w = frame.shape[:2]
        points = hand["points"]
        palm = to_px(palm_center(points), w, h)
        self._now = elapsed
        radius, motion_boost = self._motion_scale(points, w, h, hand_id)
        # Keep the visual anchored at the palm. Finger tips are deliberately unused.
        overlay = np.zeros_like(frame)
        name = character.name
        pulse = 1.0 + .08 * math.sin(elapsed * 7.0)
        r = int(radius * pulse)

        if name == "DOCTOR STRANGE":
            mandala(overlay, palm, int(r * 1.30), (0, 155, 255), (80, 220, 255), elapsed, 1.25)
            palm_arcs(overlay, palm, r * .83, (0, 175, 255), (100, 235, 255), elapsed, 12)
            particle_cloud(overlay, palm, r*.85, (0, 150, 255), (130, 235, 255), elapsed, 65, 1)
            glow(overlay, palm, 9, (0, 175, 255), (255, 250, 225))

        elif name == "SCARLET WITCH":
            # Dense crimson chaos energy concentrated around the palm.
            glow(overlay, palm, int(r*.82), (0, 0, 205), (255, 245, 255))
            for i in range(7):
                a = elapsed * (1.8 if i % 2 else -1.4) + i * math.tau/7
                axes = (int(r*(.72+i*.045)), int(r*(.48+i*.035)))
                cv2.ellipse(overlay, palm, axes, math.degrees(a), 12, 338,
                            (0, 0, 190 + (i%2)*45), 3, cv2.LINE_AA)
            palm_arcs(overlay, palm, r*.95, (0,0,220), (95,70,255), elapsed, 14)
            particle_cloud(overlay, palm, r*1.15, (0,0,220), (100,80,255), elapsed, 145, 2)
            glow(overlay, palm, int(r*.22), (0,0,245), (255,255,255))

        elif name == "AGATHA HARKNESS":
            # Layered violet witchfire with dense lavender particles and smoky coils.
            glow(overlay, palm, int(r*.88), (175, 20, 205), (255, 235, 255))
            for i in range(9):
                a = elapsed * (1.2 if i%2 else -1.5) + i*20
                cv2.ellipse(overlay, palm, (int(r*(.62+i*.055)), int(r*(.42+i*.045))),
                            math.degrees(a), 0, 335, (195, 35+i%2*25, 235), 3, cv2.LINE_AA)
            palm_arcs(overlay, palm, r*.95, (185,25,220), (245,135,255), elapsed, 16)
            particle_cloud(overlay, palm, r*1.18, (185,25,220), (245,150,255), elapsed, 155, 3)
            glow(overlay, palm, int(r*.24), (205,45,240), (255,255,255))

        elif name == "LOKI":
            # Emerald illusion magic: elegant elliptical ribbons and green motes.
            glow(overlay, palm, int(r*.75), (25,145,20), (220,255,220))
            for i in range(8):
                angle = elapsed * (1.35 if i%2 else -1.1) + i*23
                cv2.ellipse(overlay, palm, (int(r*(.58+i*.06)), int(r*(.35+i*.045))),
                            math.degrees(angle), 25, 330,
                            (25+i*2, 145+(i%2)*65, 25), 3, cv2.LINE_AA)
            particle_cloud(overlay, palm, r*1.12, (30,165,25), (155,255,155), elapsed, 125, 4)
            palm_arcs(overlay, palm, r*.9, (30,180,35), (155,255,155), elapsed, 10)
            glow(overlay, palm, int(r*.20), (45,195,45), (255,255,255))

        elif name == "THOR":
            # Blue-white lightning arcs orbit and branch locally around the palm.
            glow(overlay, palm, int(r*.72), (255,150,30), (255,255,255))
            palm_arcs(overlay, palm, r*1.05, (255,205,65), (255,250,225), elapsed, 13)
            rng = random.Random(int(elapsed*28))
            for i in range(12):
                a = elapsed*.9 + i*math.tau/12
                reach = r*(.65 + .4*rng.random())
                end = (int(palm[0]+math.cos(a)*reach),
                       int(palm[1]+math.sin(a)*reach*.78))
                # Short local zig-zag arcs are confined to the palm aura.
                mid = ((palm[0]+end[0])//2 + rng.randint(-8,8),
                       (palm[1]+end[1])//2 + rng.randint(-8,8))
                cv2.polylines(overlay, [np.asarray([palm, mid, end], np.int32)],
                              False, (255,215,105), 2, cv2.LINE_AA)
                cv2.circle(overlay, end, 2, (255,255,255), -1, cv2.LINE_AA)
            particle_cloud(overlay, palm, r*1.1, (255,185,45), (255,255,255), elapsed, 100, 5)

        elif name == "WONG":
            mandala(overlay, palm, int(r*1.12), (0,135,245), (130,220,255), elapsed, 1.1)
            palm_arcs(overlay, palm, r*.9, (0,155,255), (190,235,255), elapsed, 10)
            particle_cloud(overlay, palm, r*.9, (0,135,245), (190,235,255), elapsed, 85, 6)

        elif name == "THE ANCIENT ONE":
            mandala(overlay, palm, int(r*1.38), (65,185,220), (190,245,255), elapsed, 1.5)
            palm_arcs(overlay, palm, r*1.05, (65,185,220), (220,250,255), elapsed, 16)
            particle_cloud(overlay, palm, r*1.12, (65,185,220), (220,250,255), elapsed, 120, 7)

        bloom_composite(frame, overlay, blur=max(10, int(r*.18)),
                        bloom_strength=.95 + motion_boost*.12, detail_strength=.92)

    def render_two_hand_field(self, frame, character, hands, elapsed):
        """No hand-to-hand connector: each palm gets its own independent VFX."""
        return
