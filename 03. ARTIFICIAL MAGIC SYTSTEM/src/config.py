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

# MediaPipe Gesture Recognizer model configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/"
    "gesture_recognizer.task"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
