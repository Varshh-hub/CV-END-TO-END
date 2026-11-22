from pathlib import Path
from urllib.request import urlretrieve
import time

import cv2
import mediapipe as mp
import numpy as np

from config import MODEL_PATH, MODEL_URL

class HandTracker:
    """Pretrained MediaPipe Gesture Recognizer; no project-specific training."""

    def __init__(self, max_hands: int = 2):
        model_path = Path(MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            print("Downloading pretrained MediaPipe Gesture Recognizer...")
            urlretrieve(MODEL_URL, model_path)

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)
        self.last_timestamp_ms = 0

    def process(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            points = np.asarray(
                [[p.x, p.y, p.z] for p in landmarks], dtype=np.float32
            )
            gesture = "None"
            score = 0.0
            if index < len(result.gestures) and result.gestures[index]:
                gesture = result.gestures[index][0].category_name
                score = float(result.gestures[index][0].score)

            handedness = "Unknown"
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name

            hands.append({
                "points": points,
                "gesture": gesture,
                "confidence": score,
                "handedness": handedness,
            })
        return hands

    def close(self):
        self.recognizer.close()


class PersonSegmenter:
    """Segments the person in the webcam frame for the Thanos disappearance effect."""

    def __init__(self):
        selfie = getattr(mp, "solutions", None)
        if selfie is None or not hasattr(selfie, "selfie_segmentation"):
            raise RuntimeError(
                "This MediaPipe build does not expose selfie_segmentation. "
                "Install mediapipe==0.10.14 and restart the kernel."
            )
        self.segmenter = selfie.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    def mask(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        if result.segmentation_mask is None:
            return None
        return result.segmentation_mask

    def close(self):
        self.segmenter.close()
