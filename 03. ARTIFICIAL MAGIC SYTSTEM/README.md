# Mystic Motion — Real-Time Marvel Magic Computer Vision System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)

A real-time webcam augmented reality experience that tracks hand gestures and renders procedural, comic-accurate magical visual effects (VFX) in 60 FPS using MediaPipe and OpenCV.

---

## ✨ Features & Character Modes

| Key | Character | Magic Style & Visual Language |
| :---: | :--- | :--- |
| **`1`** | **Doctor Strange** | Eldritch magic: rotating sacred geometry mandalas, concentric rune bands, and perimeter drifting sparks. |
| **`2`** | **Scarlet Witch (Wanda)** | Chaos magic: pulsing crimson core sigil with undulating energy wisps wrapping around fingertips. |
| **`3`** | **Agatha Harkness** | Dark witchfire: layered violet coils, floating spell halo, and high-intensity two-hand purple energy beam. |
| **`4`** | **Loki** | Mischief & illusions: emerald green ribbons with orbiting diamond glyphs. |
| **`5`** | **Thor** | Thunder God: concentrated storm core with jagged branching blue-white electric arcs and upward lightning strikes. |
| **`6`** | **Thanos** | Cosmic energy & Infinity Gauntlet: six orbiting Infinity Stones and a **finger-snap gesture** to disintegrate into ash! |

---

## 🌐 Deploy to Vercel (Web / Mobile Live Experience)

This repository is configured for **1-click direct deployment to Vercel**! The web edition runs Google MediaPipe via WebAssembly and WebGL directly in the user's browser, giving 60 FPS performance with zero server compute costs.

### Deploy in 30 Seconds:
1. Push this repository to GitHub.
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. Leave all build settings default (root directory) and click **"Deploy"**!

### Or Deploy via Vercel CLI:
```bash
npx vercel
```

Your live web app will be instantly accessible at `https://your-project.vercel.app` on any device with a webcam!

---

## 🚀 Getting Started (Desktop Python App)

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/mystic-motion.git
cd "CV-END-TO-END/03. ARTIFICIAL MAGIC SYTSTEM"

pip install -r requirements.txt
```

*(Pretrained MediaPipe gesture and selfie models are automatically downloaded to `models/` on first launch).*

### 2. Run Live Webcam Experience

#### Option A: Via Command Line (Recommended)
```bash
python src/main.py
```

Optional CLI flags:
```bash
python src/main.py --camera 0 --width 1280 --height 720
python src/main.py --skeleton      # Show MediaPipe hand tracking skeleton
python src/main.py --no-mirror     # Disable horizontal frame flipping
```

#### Option B: Via Jupyter Notebook
Open `model.ipynb` in VS Code or Jupyter Lab and execute the cells.

---

## 🎮 Controls & Interactions

- **`1` – `6`**: Switch character mode.
- **`B`**: **Capture Clean Background Plate.** Step aside for a second and press `B`. When you snap as Thanos, your dissolution will blend into this clean room background at a silky-smooth 60 FPS!
- **`S`**: Toggle 21-point hand skeleton visualization.
- **`H`**: Toggle HUD help overlay.
- **`Q` or `ESC`**: Quit application.

### Thanos Snap Gesture
1. Select mode **`6`** (Thanos).
2. Pinch your **thumb and index finger** together.
3. Quickly **flick them apart**.
4. The system segments your silhouette and dissolves your person into cosmic dust!
5. Snap again to rematerialize.

---

## 🛠️ Architecture & Performance Improvements

1. **Unified Frame Rendering & Downscaled Bloom Composite:**
   - Previous versions created multiple full-resolution blank layers and applied Gaussian blur passes per-hand, bottlenecking multi-hand FPS.
   - The current engine composites all hand effects, two-hand fields, and particles onto a single pass, downsampling by 2x for bloom before in-place blending.
2. **Exponential Moving Average (EMA) Landmark Smoothing:**
   - Eliminates high-frequency webcam landmark jitter, locking rune mandalas and lightning arcs firmly in place.
3. **Scale-Invariant & Aspect-Corrected Pinch Detection:**
   - Gesture distance thresholds are normalized relative to hand span (wrist to middle MCP), ensuring consistent snap detection whether you are close to or far from the webcam.
4. **Dual MediaPipe Architecture:**
   - Supports the modern **MediaPipe Tasks Vision API** (`ImageSegmenter` & `GestureRecognizer`) with automatic fallback to legacy `mp.solutions.selfie_segmentation` for backward compatibility across Python versions.
5. **High-Speed Downscaled Inpainting Fallback:**
   - Telea inpainting runs at 1/4th resolution when no clean background plate is provided, delivering a **16x speedup** over full 720p inpainting.

---

## 📁 Repository Structure

```
├── models/
│   ├── gesture_recognizer.task   # MediaPipe gesture model (auto-downloaded)
│   └── selfie_segmenter.tflite   # MediaPipe segmentation model (auto-downloaded)
├── src/
│   ├── config.py                 # Character palettes, model URLs, and tuning params
│   ├── hand_tracking.py          # HandTracker & PersonSegmenter with smoothing
│   ├── vfx.py                    # MagicRenderer, ParticleEmitter, and bloom composite
│   └── main.py                   # Main event loop, snap detector, and HUD
├── model.ipynb                   # Interactive Jupyter notebook
├── requirements.txt              # Core Python dependencies
├── .gitignore                    # Ignore binaries, caches, and checkpoints
└── README.md                     # Documentation
```
