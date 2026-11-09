from dataclasses import dataclass

@dataclass(frozen=True)
class Character:
    key: str
    name: str
    bgr: tuple[int, int, int]
    accent: tuple[int, int, int]

CHARACTERS = {
    "1": Character("1", "DOCTOR STRANGE", (35, 155, 255), (210, 235, 255)),
    "2": Character("2", "SCARLET WITCH", (35, 35, 245), (120, 110, 255)),
    "3": Character("3", "AGATHA HARKNESS", (190, 65, 220), (245, 175, 255)),
    "4": Character("4", "LOKI", (65, 205, 85), (175, 255, 190)),
    "5": Character("5", "THOR", (255, 215, 95), (245, 250, 255)),
    "6": Character("6", "HELA", (45, 145, 65), (165, 245, 175)),
    "7": Character("7", "WONG", (30, 145, 245), (190, 225, 255)),
    "8": Character("8", "THE ANCIENT ONE", (125, 205, 95), (225, 245, 175)),
}

# MediaPipe Gesture Recognizer model configuration
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/"
    "gesture_recognizer.task"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Ninth character selection
CHARACTERS["9"] = Character("9", "WICCAN", (190, 80, 235), (245, 190, 255))
