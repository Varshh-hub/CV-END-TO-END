from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import CHARACTERS
from hand_tracking import HandTracker
from vfx import MagicRenderer

app = Flask(__name__)
tracker = None
renderer = None
model_lock = threading.Lock()
started_at = time.perf_counter()
frame_count = 0
fps_value = 0.0
last_fps_update = time.perf_counter()

def get_model():
    global tracker, renderer
    if tracker is None:
        tracker = HandTracker(max_hands=2)
        renderer = MagicRenderer()
    return tracker, renderer

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/status")
def status():
    return jsonify({
        "status": "ready",
        "characters": [
            {"key": key, "name": char.name, "accent": {
                "1": "#ff9b36", "2": "#ff405f", "3": "#c56cff",
                "4": "#6ce58a", "5": "#ffd36a"
            }[key]}
            for key, char in CHARACTERS.items()
        ]
    })

@app.post("/api/process")
def process_frame():
    global frame_count, fps_value, last_fps_update
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    character_key = request.form.get("character", "1")
    if character_key not in CHARACTERS:
        character_key = "1"

    raw = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode camera frame"}), 400

    # The browser mirrors the captured frame before upload, matching the notebook UI.

    try:
        with model_lock:
            hand_tracker, magic_renderer = get_model()
            hands = hand_tracker.process(frame)
            elapsed = time.perf_counter() - started_at

            inward_pose = (
                len(hands) >= 2
                and magic_renderer.palms_facing_each_other(hands[0], hands[1])
            )
            camera_pose = (
                len(hands) >= 2
                and magic_renderer.palms_facing_camera(hands[0], hands[1])
            )

            if inward_pose or camera_pose:
                magic_renderer.render_two_hand_field(
                    frame, CHARACTERS[character_key], hands, elapsed, force=True
                )
            else:
                for hand_id, hand in enumerate(hands):
                    magic_renderer.render_hand(
                        frame, CHARACTERS[character_key], hand, elapsed, hand_id
                    )

            now = time.perf_counter()
            frame_count += 1
            if now - last_fps_update >= 1:
                fps_value = frame_count / (now - last_fps_update)
                frame_count = 0
                last_fps_update = now

            # Compact HUD, preserving the notebook's original visual identity.
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 58), (16, 12, 30), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.putText(frame, "MYSTIC MOTION", (18, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (245, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, CHARACTERS[character_key].name, (18, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CHARACTERS[character_key].accent, 1, cv2.LINE_AA)
            cv2.putText(frame, f"HANDS: {len(hands)}  |  {fps_value:.0f} FPS",
                        (w - 190, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (235, 230, 245), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return jsonify({"error": "Frame encoding failed"}), 500
        return Response(encoded.tobytes(), mimetype="image/jpeg",
                        headers={"X-Hand-Count": str(len(hands)), "X-FPS": str(round(fps_value))})
    except Exception as exc:
        app.logger.exception("Frame processing failed")
        return jsonify({"error": str(exc)}), 500

@app.post("/api/shutdown")
def shutdown():
    global tracker, renderer
    with model_lock:
        if tracker is not None:
            tracker.close()
            tracker = None
            renderer = None
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    # Use threaded=False so camera/model processing remains sequential and safe.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
