const video = document.getElementById('cameraVideo');
const canvas = document.getElementById('captureCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const processed = document.getElementById('processedFrame');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const flipButton = document.getElementById('flipButton');
const emptyState = document.getElementById('emptyState');
const toast = document.getElementById('toast');

let stream = null;
let running = false;
let selectedCharacter = '1';
let mirror = true;
let sending = false;
let lastFrameAt = 0;

function notify(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => toast.classList.remove('show'), 3000);
}
function setStatus(active, label) {
  document.getElementById('stageStatus').textContent = label;
  document.getElementById('connectionLabel').textContent = active ? 'LIVE SESSION' : 'SYSTEM READY';
  document.getElementById('stateDot').classList.toggle('active', active);
}
function chooseCharacter(key) {
  selectedCharacter = key;
  document.querySelectorAll('.character-card').forEach(card => {
    card.classList.toggle('active', card.dataset.key === key);
  });
}
document.getElementById('characterList').addEventListener('click', event => {
  const card = event.target.closest('.character-card');
  if (card) chooseCharacter(card.dataset.key);
});
document.addEventListener('keydown', event => {
  if (['1','2','3','4','5'].includes(event.key) && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
    chooseCharacter(event.key);
  }
});

async function startCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS.');
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = 640;
    canvas.height = 360;
    running = true;
    emptyState.style.display = 'none';
    processed.style.display = 'block';
    startButton.disabled = true;
    stopButton.disabled = false;
    flipButton.disabled = false;
    setStatus(true, 'SUMMONING EFFECTS');
    document.getElementById('resolutionLabel').textContent = `${video.videoWidth} × ${video.videoHeight}`;
    notify('Camera connected. Show your palm to begin!');
    requestAnimationFrame(processLoop);
  } catch (error) {
    notify(error.message || 'Could not access the camera. Check browser permissions.');
    setStatus(false, 'CAMERA UNAVAILABLE');
  }
}

async function processLoop(timestamp) {
  if (!running) return;
  if (!sending && timestamp - lastFrameAt > 65 && video.readyState >= 2) {
    sending = true;
    lastFrameAt = timestamp;
    try {
      ctx.save();
      if (mirror) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.78));
      if (!blob) throw new Error('Could not capture camera frame.');
      const body = new FormData();
      body.append('frame', blob, 'camera.jpg');
      body.append('character', selectedCharacter);

      const response = await fetch('/api/process', { method: 'POST', body });
      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.error || 'The AI processor returned an error.');
      }
      const imageBlob = await response.blob();
      const imageUrl = URL.createObjectURL(imageBlob);
      const oldUrl = processed.dataset.objectUrl;
      processed.src = imageUrl;
      processed.dataset.objectUrl = imageUrl;
      if (oldUrl) URL.revokeObjectURL(oldUrl);

      document.getElementById('handsCount').textContent = response.headers.get('X-Hand-Count') ?? '—';
      document.getElementById('fpsCount').textContent = `${response.headers.get('X-FPS') ?? '—'} FPS`;
      setStatus(true, 'MAGIC ACTIVE');
    } catch (error) {
      console.error(error);
      notify(error.message || 'Frame processing failed.');
      stopCamera(false);
      return;
    } finally {
      sending = false;
    }
  }
  if (running) requestAnimationFrame(processLoop);
}

function stopCamera(showMessage = true) {
  running = false;
  sending = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  processed.removeAttribute('src');
  if (processed.dataset.objectUrl) URL.revokeObjectURL(processed.dataset.objectUrl);
  processed.dataset.objectUrl = '';
  processed.style.display = 'none';
  emptyState.style.display = 'flex';
  startButton.disabled = false;
  stopButton.disabled = true;
  flipButton.disabled = true;
  document.getElementById('handsCount').textContent = '—';
  document.getElementById('fpsCount').textContent = '— FPS';
  setStatus(false, 'CAMERA STANDBY');
  if (showMessage) notify('Session ended. Your camera is off.');
}
startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', () => stopCamera());
flipButton.addEventListener('click', () => {
  mirror = !mirror;
  notify(mirror ? 'Mirror view enabled' : 'Mirror view disabled');
});
window.addEventListener('beforeunload', () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});
fetch('/api/status').then(r => r.json()).then(data => {
  if (data.status === 'ready') setStatus(false, 'CAMERA STANDBY');
}).catch(() => setStatus(false, 'BACKEND OFFLINE'));
