# Mystic Motion — Web UI

A browser-based interface for the existing Mystic Motion MediaPipe + OpenCV project.

## Run locally

1. Extract the ZIP.
2. Open a terminal in the extracted `mystic_motion_web` folder.
3. Create and activate a virtual environment:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the server:
   ```bash
   python app.py
   ```
6. Open **http://127.0.0.1:5000** in Chrome or Edge and click **Activate camera**.
7. Allow camera permission. Choose powers using the character cards or keyboard keys `1`–`5`.

The first run downloads the MediaPipe Gesture Recognizer task file automatically into `models/`.

## How it works

- HTML/CSS/JS provide the interface and browser webcam capture.
- The browser sends small JPEG frames to the Flask endpoint `/api/process`.
- The Python backend runs the notebook's `HandTracker` and `MagicRenderer`.
- The processed frame is returned and displayed in the live preview.

## Important notes

- This is a local development deployment. For a public deployment, use a host that supports Python, install the requirements, and serve the app over HTTPS. Browsers generally require HTTPS (or localhost) for webcam access.
- A normal static hosting service (GitHub Pages, Netlify static hosting, etc.) cannot run the Python model. You need a Python backend.
- The backend processes one frame at a time to protect the MediaPipe video recognizer state.
- Performance depends on your CPU/GPU and network/browser setup. The UI sends frames at a modest rate to keep the experience responsive.
- The original notebook's `src/main.py` desktop entry point is included too; `app.py` is the web entry point.
