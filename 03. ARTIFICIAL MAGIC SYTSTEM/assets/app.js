// ============================================================================
// MYSTIC MOTION — Main Web Application Controller (ES Module)
// Powered by MediaPipe Tasks Vision & HTML5 Canvas
// ============================================================================

import {
  FilesetResolver,
  GestureRecognizer
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

import {
  CHARACTERS,
  MagicRenderer,
  handScalePx,
  toPx
} from "./vfx.js";

// DOM Elements
const video = document.getElementById("webcam-video");
const canvas = document.getElementById("output-canvas");
const ctx = canvas.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingDesc = document.getElementById("loading-desc");
const errorOverlay = document.getElementById("error-overlay");
const errorMsg = document.getElementById("error-msg");
const handsCountEl = document.getElementById("hands-count");
const fpsCounterEl = document.getElementById("fps-counter");
const activeCharNameEl = document.getElementById("active-character-name");
const thanosBanner = document.getElementById("thanos-banner");
const thanosStatusText = document.getElementById("thanos-status-text");
const charButtons = document.querySelectorAll(".char-btn");
const btnBg = document.getElementById("btn-bg");
const btnSkeleton = document.getElementById("btn-skeleton");
const btnHelp = document.getElementById("btn-help");
const helpModal = document.getElementById("help-modal");
const btnCloseHelp = document.getElementById("btn-close-help");
const btnRetry = document.getElementById("btn-retry");

// Application State
let gestureRecognizer = null;
let running = false;
let activeCharKey = "1";
let showSkeleton = false;
let mirror = true;
let backgroundPlate = null;

// Thanos Snap State
let isVanished = false;
let transitionStart = null;
let transitionFrom = 0.0;
let transitionTo = 0.0;
const transitionDuration = 1.2;

// Snap Detector State
const pinchedAt = new Map();
let lastSnapTime = 0;
const previousDistance = new Map();

// Performance & Animation
const renderer = new MagicRenderer();
let lastFrameTime = performance.now();
let fps = 60;
let lastVideoTime = -1;
let lastInferenceTimestamp = 0;

// Smoothing State
const smoothedHands = new Map();
const SMOOTHING_ALPHA = 0.70;

// ----------------------------------------------------------------------------
// MediaPipe Initialization (Pinned to 0.10.14 for Rock-Solid Reliability)
// ----------------------------------------------------------------------------
async function initVision() {
  try {
    loadingDesc.textContent = "Loading WebAssembly vision runtime from CDN...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    loadingDesc.textContent = "Loading MediaPipe Gesture Recognizer...";
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minTrackingConfidence: 0.50
    });

    loadingDesc.textContent = "Connecting to webcam...";
    await startCamera();
    loadingOverlay.classList.add("hidden");
    running = true;
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Initialization error:", err);
    loadingOverlay.classList.add("hidden");
    errorMsg.textContent = err.message || "Failed to load camera or MediaPipe models.";
    errorOverlay.classList.remove("hidden");
  }
}

// ----------------------------------------------------------------------------
// Camera Stream Initialization with Fallback & Re-entrance Protection
// ----------------------------------------------------------------------------
async function startCamera() {
  // Release any existing tracks so webcam isn't locked on page refresh
  if (video.srcObject) {
    try {
      video.srcObject.getTracks().forEach(t => t.stop());
    } catch (_) {}
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: false
  });

  video.srcObject = stream;

  // Ensure playback starts without hanging on autoplay policy
  try {
    await video.play();
  } catch (playErr) {
    console.warn("Autoplay notice, will resume on user interaction:", playErr);
  }

  // Robust dimensions resolution: handles instant cache and slow event dispatch
  await new Promise(resolve => {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      return resolve();
    }
    const onLoaded = () => {
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      video.removeEventListener("loadeddata", onLoaded);
      resolve();
    };
    video.addEventListener("loadeddata", onLoaded);
    // 800ms safety timeout fallback
    setTimeout(onLoaded, 800);
  });
}

// ----------------------------------------------------------------------------
// Snap Detector (Pinch & Flick with hand scale normalization)
// ----------------------------------------------------------------------------
function updateSnapDetector(hands, nowSec) {
  let snapped = false;

  hands.forEach((hand, handId) => {
    const points = hand.points;
    const p0 = points[0];
    const p9 = points[9];
    const handScale = Math.hypot(p9.x - p0.x, p9.y - p0.y) + 1e-5;

    const p4 = points[4];
    const p8 = points[8];
    const relDist = Math.hypot(p4.x - p8.x, p4.y - p8.y) / handScale;

    const wasPinched = pinchedAt.has(handId);
    if (relDist < 0.22) {
      pinchedAt.set(handId, nowSec);
    } else if (wasPinched) {
      const held = nowSec - pinchedAt.get(handId);
      pinchedAt.delete(handId);
      const prev = previousDistance.get(handId) || relDist;
      const quickRelease = relDist > 0.45 && held < 0.65;
      const sharpMotion = (relDist - prev) > 0.15;

      if (quickRelease && sharpMotion && (nowSec - lastSnapTime) > 1.0) {
        snapped = true;
        lastSnapTime = nowSec;
      }
    }
    previousDistance.set(handId, relDist);
  });

  for (const [id, time] of pinchedAt.entries()) {
    if (id >= hands.length || (nowSec - time) > 0.6) {
      pinchedAt.delete(id);
    }
  }

  return snapped;
}

// ----------------------------------------------------------------------------
// Draw Hand Skeleton
// ----------------------------------------------------------------------------
function drawSkeleton(ctx, points, width, height) {
  const p = points.map(pt => toPx(pt, width, height));
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
  ];

  ctx.save();
  ctx.strokeStyle = "rgba(110, 210, 180, 0.75)";
  ctx.lineWidth = 1.8;
  connections.forEach(([a, b]) => {
    ctx.beginPath();
    ctx.moveTo(p[a].x, p[a].y);
    ctx.lineTo(p[b].x, p[b].y);
    ctx.stroke();
  });

  ctx.fillStyle = "#e0f5ed";
  p.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

// ----------------------------------------------------------------------------
// Main Real-Time Render Loop (Protected Against Monotonic Timestamp Crashes)
// ----------------------------------------------------------------------------
function renderLoop(currentTime) {
  if (!running) return;

  const width = canvas.width;
  const height = canvas.height;
  const nowSec = currentTime / 1000;
  const dt = Math.max(1e-4, (currentTime - lastFrameTime) / 1000);
  lastFrameTime = currentTime;

  const currentFps = 1 / dt;
  fps = Math.round(0.9 * fps + 0.1 * currentFps);
  fpsCounterEl.textContent = fps;

  // Draw current webcam video feed onto canvas
  ctx.save();
  if (mirror) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();

  // Run MediaPipe Gesture Recognition safely
  const detectedHands = [];
  if (gestureRecognizer && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // MediaPipe strictly enforces monotonically increasing timestamps
    const nowMs = Math.round(performance.now());
    const timestampMs = Math.max(lastInferenceTimestamp + 1, nowMs);
    lastInferenceTimestamp = timestampMs;

    try {
      const results = gestureRecognizer.recognizeForVideo(video, timestampMs);
      if (results && results.landmarks) {
        results.landmarks.forEach((landmarks, index) => {
          let pts = landmarks.map(p => ({
            x: mirror ? 1.0 - p.x : p.x,
            y: p.y,
            z: p.z || 0
          }));

          const key = `hand_${index}`;
          if (smoothedHands.has(key)) {
            const prev = smoothedHands.get(key);
            pts = pts.map((pt, i) => ({
              x: SMOOTHING_ALPHA * pt.x + (1 - SMOOTHING_ALPHA) * prev[i].x,
              y: SMOOTHING_ALPHA * pt.y + (1 - SMOOTHING_ALPHA) * prev[i].y,
              z: pt.z
            }));
          }
          smoothedHands.set(key, pts);

          const gesture = results.gestures && results.gestures[index] && results.gestures[index][0]
            ? results.gestures[index][0].categoryName
            : "None";

          detectedHands.push({ points: pts, gesture });
        });
      }
    } catch (inferErr) {
      // Catch and skip dropped frames to prevent killing animation loop
      console.warn("Inference frame skipped:", inferErr);
    }
  }

  handsCountEl.textContent = detectedHands.length;
  const character = CHARACTERS[activeCharKey];

  // Thanos Snap Logic & Disappearance
  if (character.name === "THANOS") {
    thanosBanner.classList.remove("hidden");
    const snapped = updateSnapDetector(detectedHands, nowSec);

    if (snapped) {
      transitionFrom = isVanished ? 1.0 : 0.0;
      transitionTo = isVanished ? 0.0 : 1.0;
      transitionStart = nowSec;
      isVanished = !isVanished;
    }

    if (transitionStart !== null) {
      const frac = Math.min(1.0, (nowSec - transitionStart) / transitionDuration);
      const eased = frac * frac * (3.0 - 2.0 * frac);
      const progress = transitionFrom + (transitionTo - transitionFrom) * eased;
      applyDisappearance(ctx, width, height, progress, nowSec);
      if (frac >= 1.0) transitionStart = null;
    } else if (isVanished) {
      applyDisappearance(ctx, width, height, 1.0, nowSec);
    }

    thanosStatusText.textContent = isVanished
      ? "YOU HAVE DISINTEGRATED — PINCH & FLICK TO RETURN"
      : (backgroundPlate ? "INFINITY GAUNTLET READY [BG CAPTURED]" : "INFINITY GAUNTLET READY — PINCH & FLICK TO SNAP");
  } else {
    thanosBanner.classList.add("hidden");
    isVanished = false;
    transitionStart = null;
  }

  // Draw Skeletons if enabled
  if (showSkeleton) {
    detectedHands.forEach(hand => drawSkeleton(ctx, hand.points, width, height));
  }

  // Render Magic VFX
  renderer.renderAll(ctx, character, detectedHands, nowSec, dt, width, height);

  requestAnimationFrame(renderLoop);
}

// ----------------------------------------------------------------------------
// Disappearance Effect (Thanos Snap)
// ----------------------------------------------------------------------------
function applyDisappearance(ctx, width, height, progress, elapsed) {
  if (progress <= 0.01) return;

  ctx.save();
  if (backgroundPlate) {
    ctx.globalAlpha = Math.min(1.0, progress * 1.15);
    ctx.drawImage(backgroundPlate, 0, 0, width, height);
  } else {
    ctx.fillStyle = `rgba(18, 10, 24, ${progress * 0.88})`;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.restore();

  const count = Math.floor(18 * progress);
  for (let i = 0; i < count; i++) {
    const rx = Math.random() * width;
    const ry = Math.random() * height;
    const vx = (Math.random() - 0.5) * 40;
    const vy = -30 - Math.random() * 60;
    const colors = ["#cd9bff", "#ffea35", "#ff88ff", "#ffffff"];
    const color = colors[Math.floor(Math.random() * colors.length)];
    renderer.particles.emit(rx, ry, vx, vy, 0.6 + Math.random() * 0.4, color, 2.5);
  }
}

// ----------------------------------------------------------------------------
// Event Listeners & UI Controls
// ----------------------------------------------------------------------------
function setActiveCharacter(key) {
  if (!CHARACTERS[key]) return;
  activeCharKey = key;
  const char = CHARACTERS[key];
  activeCharNameEl.textContent = char.name;
  activeCharNameEl.style.color = char.accent;
  document.documentElement.style.setProperty("--accent-color", char.accent);

  charButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.char === key);
  });

  if (char.name !== "THANOS") {
    isVanished = false;
    transitionStart = null;
  }
}

charButtons.forEach(btn => {
  btn.addEventListener("click", () => setActiveCharacter(btn.dataset.char));
});

btnBg.addEventListener("click", () => {
  const bgCanvas = document.createElement("canvas");
  bgCanvas.width = canvas.width;
  bgCanvas.height = canvas.height;
  const bgCtx = bgCanvas.getContext("2d");
  if (mirror) {
    bgCtx.translate(canvas.width, 0);
    bgCtx.scale(-1, 1);
  }
  bgCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  backgroundPlate = bgCanvas;
  btnBg.style.borderColor = "#91ff9b";
  btnBg.style.color = "#91ff9b";
  setTimeout(() => {
    btnBg.style.borderColor = "";
    btnBg.style.color = "";
  }, 1200);
});

btnSkeleton.addEventListener("click", () => {
  showSkeleton = !showSkeleton;
  btnSkeleton.classList.toggle("active", showSkeleton);
});

btnHelp.addEventListener("click", () => helpModal.classList.remove("hidden"));
btnCloseHelp.addEventListener("click", () => helpModal.classList.add("hidden"));
helpModal.addEventListener("click", e => {
  if (e.target === helpModal) helpModal.classList.add("hidden");
});

btnRetry.addEventListener("click", () => {
  errorOverlay.classList.add("hidden");
  loadingOverlay.classList.remove("hidden");
  initVision();
});

window.addEventListener("keydown", e => {
  const key = e.key.toUpperCase();
  if (CHARACTERS[key]) {
    setActiveCharacter(key);
  } else if (key === "B") {
    btnBg.click();
  } else if (key === "S") {
    btnSkeleton.click();
  } else if (key === "M") {
    mirror = !mirror;
  } else if (key === "H") {
    helpModal.classList.toggle("hidden");
  } else if (key === "ESCAPE") {
    helpModal.classList.add("hidden");
  }
});

// Release camera tracks cleanly on tab close or refresh
window.addEventListener("beforeunload", () => {
  if (video.srcObject) {
    try {
      video.srcObject.getTracks().forEach(t => t.stop());
    } catch (_) {}
  }
});

// Resume playback if browser suspended stream when tab was switched
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && video.paused && running) {
    video.play().catch(() => {});
  }
});

window.addEventListener("DOMContentLoaded", initVision);
