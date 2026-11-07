from pathlib import Path
from urllib.request import urlretrieve
import time

import cv2
import mediapipe as mp
import numpy as np

from config import MODEL_PATH, MODEL_URL
