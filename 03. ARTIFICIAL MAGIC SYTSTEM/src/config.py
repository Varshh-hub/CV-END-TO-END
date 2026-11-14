from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Character:
    key: str
    name: str
    bgr: tuple[int, int, int]
    accent: tuple[int, int, int]

# Six character modes. OpenCV uses BGR colour order.
CHARACTERS = {
    "1": Character("1", "DOCTOR STRANGE", (0, 155, 255), (80, 220, 255)),
    "2": Character("2", "SCARLET WITCH", (0, 0, 235), (70, 35, 255)),
    "3": Character("3", "AGATHA HARKNESS", (220, 35, 235), (245, 145, 255)),
    "4": Character("4", "LOKI", (45, 205, 55), (145, 255, 155)),
    "5": Character("5", "THOR", (255, 205, 65), (255, 250, 225)),
    "6": Character("6", "THANOS", (150, 80, 205), (205, 155, 255)),
}

# MediaPipe Gesture Recognizer & Segmenter model configurations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/"
    "gesture_recognizer.task"
)

SEGMENTER_MODEL_PATH = MODEL_DIR / "selfie_segmenter.tflite"
SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/"
    "selfie_segmenter.tflite"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Performance & Rendering tuning
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
BLOOM_BLUR = 15
BLOOM_STRENGTH = 0.75
DETAIL_STRENGTH = 0.92
INPAINT_DOWNSCALE = 0.25  # Inpaint at 1/4 resolution for 16x speedup
SMOOTHING_ALPHA = 0.65    # EMA smoothing factor for landmarks (0.0 = freeze, 1.0 = raw)