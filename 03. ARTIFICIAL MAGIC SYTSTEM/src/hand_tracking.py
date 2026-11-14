from pathlib import Path
from urllib.request import urlretrieve
import time

import cv2
import mediapipe as mp
import numpy as np

from config import MODEL_PATH, MODEL_URL, SEGMENTER_MODEL_PATH, SEGMENTER_MODEL_URL, SMOOTHING_ALPHA


class HandTracker:
    """Pretrained MediaPipe Gesture Recognizer with temporal landmark smoothing."""

    def __init__(self, max_hands: int = 2, smoothing_alpha: float = SMOOTHING_ALPHA):
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
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_hands = {}

    def process(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands = []
        active_keys = set()

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

            # Landmark temporal smoothing to eliminate high-frequency jitter
            hand_key = f"{handedness}_{index}"
            active_keys.add(hand_key)
            if hand_key in self.smoothed_hands:
                prev_points = self.smoothed_hands[hand_key]
                # If hand jumped too far (teleported), reset smoothing
                if np.linalg.norm(points[0, :2] - prev_points[0, :2]) < 0.25:
                    points = self.smoothing_alpha * points + (1.0 - self.smoothing_alpha) * prev_points
            self.smoothed_hands[hand_key] = points

            hands.append({
                "points": points,
                "gesture": gesture,
                "confidence": score,
                "handedness": handedness,
            })

        # Purge stale smoothing entries
        for key in list(self.smoothed_hands.keys()):
            if key not in active_keys:
                del self.smoothed_hands[key]

        return hands

    def close(self):
        self.recognizer.close()


class PersonSegmenter:
    """Segments the person in the webcam frame for the Thanos disappearance effect.
    
    Supports modern MediaPipe Tasks API (ImageSegmenter) with graceful fallback
    to legacy mp.solutions.selfie_segmentation for cross-version compatibility.
    """

    def __init__(self):
        self.use_tasks_api = False
        self.segmenter = None

        # 1. First attempt: Modern MediaPipe Tasks API
        try:
            model_path = Path(SEGMENTER_MODEL_PATH)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if not model_path.exists():
                print("Downloading MediaPipe Selfie Segmenter model...")
                urlretrieve(SEGMENTER_MODEL_URL, model_path)

            BaseOptions = mp.tasks.BaseOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self.segmenter = ImageSegmenter.create_from_options(options)
            self.use_tasks_api = True
            print("PersonSegmenter: initialized using MediaPipe Tasks ImageSegmenter.")
        except Exception as e:
            print(f"Tasks ImageSegmenter unavailable ({e}). Falling back to solutions API...")

        # 2. Fallback attempt: Legacy solutions API
        if not self.use_tasks_api:
            selfie = getattr(mp, "solutions", None)
            if selfie is not None and hasattr(selfie, "selfie_segmentation"):
                self.segmenter = selfie.selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
                print("PersonSegmenter: initialized using legacy mp.solutions.selfie_segmentation.")
            else:
                raise RuntimeError(
                    "Neither MediaPipe Tasks ImageSegmenter nor mp.solutions.selfie_segmentation "
                    "could be initialized. Please check mediapipe installation."
                )

    def mask(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.segmenter.segment(mp_image)
            if result.confidence_masks:
                # Confidence mask index 0 corresponds to person foreground
                mask = result.confidence_masks[0].numpy_view()
                return mask
            return None
        else:
            result = self.segmenter.process(rgb)
            if result.segmentation_mask is None:
                return None
            return result.segmentation_mask

    def close(self):
        if self.segmenter is not None:
            self.segmenter.close()
