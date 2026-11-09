import math
import random
from collections import deque

import cv2
import numpy as np

from config import Character

def palm_center(p):
    return np.mean(p[[0, 5, 9, 13, 17], :2], axis=0)

def to_px(point, w, h):
    return int(np.clip(point[0] * w, 0, w - 1)), int(np.clip(point[1] * h, 0, h - 1))

def hand_is_open(p):
    wrist = p[0, :2]
    checks = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = sum(
        np.linalg.norm(p[tip, :2] - wrist) >
        np.linalg.norm(p[pip, :2] - wrist) * 1.12
        for tip, pip in checks
    )
    return extended >= 3

def is_pinching(p, threshold=0.060):
    return float(np.linalg.norm(p[4, :2] - p[8, :2])) < threshold

def draw_polyline(layer, points, color, thickness=2):
    if len(points) >= 2:
        cv2.polylines(layer, [np.asarray(points, np.int32)], False,
                      color, thickness, cv2.LINE_AA)

def glow(layer, center, radius, color, intensity=1.0):
    x, y = center
    cv2.circle(layer, (x, y), max(2, int(radius * 2.8)),
               color, -1, cv2.LINE_AA)
    cv2.circle(layer, (x, y), max(1, int(radius)),
               (245, 250, 255), -1, cv2.LINE_AA)

def ring(layer, center, radius, color, angle, count=3, aspect=0.72):
    x, y = center
    for i in range(count):
        r = max(8, int(radius * (0.70 + 0.16 * i)))
        cv2.ellipse(layer, (x, y), (r, max(5, int(r * aspect))),
                    int(angle * (1 if i % 2 else -1) + i * 37),
                    0, 350, color, max(1, 3 - i // 2), cv2.LINE_AA)

def arc(layer, a, b, color, t, amplitude=8, thickness=2, branches=5):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a
    norm = np.linalg.norm(delta) + 1e-6
    perpendicular = np.array([-delta[1], delta[0]]) / norm
    pts = [tuple(a.astype(int))]
    for i in range(1, branches):
        f = i / branches
        base = a + delta * f
        jitter = math.sin(t * 17 + i * 5.7) * amplitude
        pts.append(tuple((base + perpendicular * jitter).astype(int)))
    pts.append(tuple(b.astype(int)))
    draw_polyline(layer, pts, color, thickness)
    draw_polyline(layer, pts, (235, 245, 255), 1)

import math
class ParticleField:
    def __init__(self, limit=420):
        self.particles = deque(maxlen=limit)

    def emit(self, origin, color, count=5, speed=2.0, spread=math.tau):
        ox, oy = origin
        for _ in range(count):
            angle = random.random() * spread
            velocity = random.uniform(0.5, speed)
            self.particles.append({
                "x": float(ox), "y": float(oy),
                "vx": math.cos(angle) * velocity,
                "vy": math.sin(angle) * velocity,
                "life": random.randint(14, 34),
                "max_life": 34,
                "r": random.randint(1, 3),
                "color": color,
            })

    def update_draw(self, layer):
        alive = deque(maxlen=self.particles.maxlen)
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.985
            p["vy"] *= 0.985
            p["life"] -= 1
            if p["life"] <= 0:
                continue
            fade = p["life"] / p["max_life"]
            color = tuple(int(c * fade) for c in p["color"])
            cv2.circle(layer, (int(p["x"]), int(p["y"])),
                       max(1, int(p["r"] * fade)), color, -1, cv2.LINE_AA)
            alive.append(p)
        self.particles = alive

from config import Character
class MagicRenderer:
    """Procedural, character-specific VFX inspired by each character's film look.

    These are original real-time approximations, not extracted or copied film assets.
    """

    def __init__(self):
        self.particles = ParticleField()
        self.trails = {}

    def _trail(self, key, point, maxlen=24):
        trail = self.trails.setdefault(key, deque(maxlen=maxlen))
        trail.append(point)
        return list(trail)

    def render_hand(self, frame, character: Character, hand, elapsed, hand_id=0):
        h, w = frame.shape[:2]
        p = hand["points"]
        gesture = hand["gesture"]
        confidence = hand["confidence"]
        palm = to_px(palm_center(p), w, h)
        tips = [to_px(p[i, :2], w, h) for i in (4, 8, 12, 16, 20)]
        index_tip = tips[1]
        open_hand = gesture == "Open_Palm" or hand_is_open(p)
        fist = gesture == "Closed_Fist"
        pinch = is_pinching(p)
        power = 1.0 if open_hand else (0.8 if fist else 0.65)
        color, accent = character.bgr, character.accent

        overlay = np.zeros_like(frame)
        self.particles.emit(palm, color, count=2 if open_hand else 1, speed=2.4)

        trail = self._trail((hand_id, character.name), index_tip)
        if len(trail) > 2:
            for i in range(1, len(trail)):
                strength = i / len(trail)
                c = tuple(int(v * strength) for v in accent)
                cv2.line(overlay, trail[i - 1], trail[i], c,
                         max(1, int(1 + strength * 5)), cv2.LINE_AA)

        if character.name == "DOCTOR STRANGE":
            radius = int(54 + 35 * power)
            ring(overlay, palm, radius, color, elapsed * 78, 5, 0.62)
            ring(overlay, palm, radius * 0.70, accent, -elapsed * 115, 4, 0.82)
            for i in range(32):
                a = elapsed * 0.8 + i * math.tau / 32
                r1, r2 = radius * 0.72, radius * (0.91 if i % 4 else 1.08)
                q1 = (int(palm[0] + math.cos(a) * r1),
                      int(palm[1] + math.sin(a) * r1))
                q2 = (int(palm[0] + math.cos(a) * r2),
                      int(palm[1] + math.sin(a) * r2))
                cv2.line(overlay, q1, q2, accent if i % 4 == 0 else color, 2, cv2.LINE_AA)
            for tip in tips:
                arc(overlay, palm, tip, accent, elapsed, 5, 2)
            if pinch:
                ring(overlay, index_tip, 28, accent, elapsed * 130, 3)

        elif character.name == "SCARLET WITCH":
            radius = int(20 + 22 * power + 8 * math.sin(elapsed * 5))
            glow(overlay, palm, radius, color)
            for i, tip in enumerate(tips):
                if open_hand or i == 1:
                    arc(overlay, palm, tip, color, elapsed + i, 15, 4, 8)
                    arc(overlay, palm, tip, accent, elapsed + i * 0.4, 7, 1, 8)
            for i in range(9):
                a = elapsed * 1.8 + i * math.tau / 9
                end = (int(palm[0] + math.cos(a) * (55 + 12 * math.sin(elapsed * 3 + i))),
                       int(palm[1] + math.sin(a) * (45 + 10 * math.cos(elapsed * 2 + i))))
                arc(overlay, palm, end, color, elapsed + i, 12, 3, 7)

        elif character.name == "AGATHA HARKNESS":
            ring(overlay, palm, 42 + int(16 * power), color, elapsed * 90, 3, 0.55)
            for i, tip in enumerate(tips):
                phase = elapsed * 2.0 + i * 1.3
                mid = (int((palm[0] + tip[0]) / 2 + 22 * math.sin(phase)),
                       int((palm[1] + tip[1]) / 2 + 22 * math.cos(phase)))
                draw_polyline(overlay, [palm, mid, tip], accent, 3)
                glow(overlay, tip, 4, color)
            for i in range(7):
                a = -elapsed * 1.3 + i * math.tau / 7
                q = (int(palm[0] + 58 * math.cos(a)),
                     int(palm[1] + 38 * math.sin(a)))
                arc(overlay, palm, q, color, elapsed + i, 10, 2)

        elif character.name == "LOKI":
            for offset in (-18, 0, 18):
                pts = [(x + offset, y - offset // 2) for x, y in trail[-12:]]
                if len(pts) > 1:
                    draw_polyline(overlay, pts, color, 2)
            for i in range(3):
                radius = 22 + i * 18
                ring(overlay, palm, radius, color if i != 1 else accent,
                     elapsed * (65 if i % 2 else -80) + i * 40, 2, 0.9)
            for i, tip in enumerate(tips):
                if open_hand or i == 1:
                    q = (tip[0] + int(18 * math.sin(elapsed * 3 + i)),
                         tip[1] - int(12 * math.cos(elapsed * 2 + i)))
                    arc(overlay, tip, q, accent, elapsed + i, 8, 2)

        elif character.name == "THOR":
            for tip in tips:
                arc(overlay, palm, tip, accent, elapsed, 13, 3, 9)
                if open_hand:
                    for branch in range(2):
                        bx = int((palm[0] + tip[0]) / 2 + random.randint(-25, 25))
                        by = int((palm[1] + tip[1]) / 2 + random.randint(-25, 25))
                        arc(overlay, (bx, by), tip, color, elapsed + branch, 8, 2, 6)
            glow(overlay, palm, 11 if fist else 7, color)
            if fist:
                ring(overlay, palm, 48, accent, elapsed * 120, 3)

        elif character.name == "HELA":
            ring(overlay, palm, 30, color, -elapsed * 35, 2, 0.45)
            blade_count = 5 if not open_hand else 8
            for i in range(blade_count):
                a = elapsed * 0.28 + i * math.tau / blade_count
                length = 80 + 18 * math.sin(elapsed * 2 + i)
                tip = (int(palm[0] + math.cos(a) * length),
                       int(palm[1] + math.sin(a) * length * 0.55))
                side = (int(palm[0] + math.cos(a + 0.12) * length * 0.72),
                        int(palm[1] + math.sin(a + 0.12) * length * 0.40))
                draw_polyline(overlay, [palm, side, tip], color, 5)
                draw_polyline(overlay, [palm, tip], accent, 1)
                glow(overlay, tip, 3, accent)

        elif character.name == "WONG":
            radius = int(58 + 28 * power)
            ring(overlay, palm, radius, color, elapsed * 45, 6, 0.78)
            for i in range(12):
                a = elapsed * 0.22 + i * math.tau / 12
                inner = (int(palm[0] + math.cos(a) * radius * 0.35),
                         int(palm[1] + math.sin(a) * radius * 0.35))
                outer = (int(palm[0] + math.cos(a) * radius),
                         int(palm[1] + math.sin(a) * radius))
                cv2.line(overlay, inner, outer, accent, 2, cv2.LINE_AA)
                cv2.circle(overlay, outer, 4, color, 1, cv2.LINE_AA)
            if fist:
                ring(overlay, palm, radius + 12, accent, -elapsed * 55, 2)

        elif character.name == "THE ANCIENT ONE":
            ring(overlay, palm, 66, color, elapsed * 24, 7, 0.88)
            for i in range(24):
                a = -elapsed * 0.18 + i * math.tau / 24
                r1, r2 = 49, 68 if i % 3 == 0 else 59
                p1 = (int(palm[0] + math.cos(a) * r1),
                      int(palm[1] + math.sin(a) * r1))
                p2 = (int(palm[0] + math.cos(a) * r2),
                      int(palm[1] + math.sin(a) * r2))
                cv2.line(overlay, p1, p2, accent, 2, cv2.LINE_AA)
            for tip in tips:
                arc(overlay, palm, tip, color, -elapsed, 5, 2)

        if fist:
            ring(overlay, palm, max(16, int(34 - 8 * math.sin(elapsed * 4))),
                 accent, elapsed * 150, 3)
            glow(overlay, palm, 10, color)
        elif pinch:
            focus = ((tips[0][0] + tips[1][0]) // 2,
                     (tips[0][1] + tips[1][1]) // 2)
            ring(overlay, focus, 22, accent, elapsed * 170, 3)
            glow(overlay, focus, 5, color)
        elif open_hand:
            ring(overlay, palm, 74, color, elapsed * 38, 2)

        self.particles.update_draw(overlay)
        bloom = cv2.GaussianBlur(overlay, (0, 0), 9)
        frame[:] = cv2.addWeighted(frame, 1.0, bloom, 0.48, 0)
        frame[:] = cv2.addWeighted(frame, 1.0, overlay, 0.82, 0)

    def render_two_hand_field(self, frame, character: Character, hands, elapsed):
        if len(hands) < 2:
            return
        h, w = frame.shape[:2]
        p1, p2 = hands[0]["points"], hands[1]["points"]
        c1, c2 = palm_center(p1), palm_center(p2)
        distance = float(np.linalg.norm(c1 - c2))
        if distance < 0.16:
            return
        center = to_px((c1 + c2) / 2, w, h)
        radius = int(np.clip(distance * w * 0.52, 55, min(w, h) * 0.42))
        overlay = np.zeros_like(frame)
        color, accent = character.bgr, character.accent

        if character.name in ("DOCTOR STRANGE", "WONG", "THE ANCIENT ONE"):
            ring(overlay, center, radius, color, elapsed * 55, 7, 0.72)
            ring(overlay, center, int(radius * 0.64), accent, -elapsed * 75, 5, 0.95)
            for i in range(24):
                a = elapsed * 0.35 + i * math.tau / 24
                a1 = (int(center[0] + math.cos(a) * radius * 0.42),
                      int(center[1] + math.sin(a) * radius * 0.42))
                a2 = (int(center[0] + math.cos(a) * radius * 0.92),
                      int(center[1] + math.sin(a) * radius * 0.92))
                cv2.line(overlay, a1, a2, color, 2, cv2.LINE_AA)
        elif character.name == "THOR":
            for i in range(18):
                a = i * math.tau / 18
                q = (int(center[0] + math.cos(a) * radius),
                     int(center[1] + math.sin(a) * radius))
                arc(overlay, to_px(c1, w, h), q, accent, elapsed + i, 16, 3, 8)
                arc(overlay, to_px(c2, w, h), q, color, elapsed - i, 12, 2, 8)
        elif character.name == "HELA":
            for i in range(12):
                a = elapsed * 0.15 + i * math.tau / 12
                q = (int(center[0] + math.cos(a) * radius),
                     int(center[1] + math.sin(a) * radius))
                draw_polyline(overlay, [to_px(c1, w, h), q, to_px(c2, w, h)], color, 4)
        else:
            ring(overlay, center, radius, color, elapsed * 40, 4, 0.6)
            for i in range(12):
                a = elapsed + i * math.tau / 12
                q = (int(center[0] + math.cos(a) * radius),
                     int(center[1] + math.sin(a) * radius))
                arc(overlay, to_px(c1, w, h), q, accent, elapsed + i, 14, 3)

        bloom = cv2.GaussianBlur(overlay, (0, 0), 12)
        frame[:] = cv2.addWeighted(frame, 1.0, bloom, 0.52, 0)
        frame[:] = cv2.addWeighted(frame, 1.0, overlay, 0.78, 0)
