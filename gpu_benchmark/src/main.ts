import "./style.css";
import TrackingWorker from "./worker?worker";

type WorkerResult = {
  type: "result";
  sessionId: number;
  frameId: number;
  capturedAt: number;
  resultAt: number;
  inferenceMs: number;
  handCount: number;
  handedness: Array<{ label: string; score: number }>;
  landmarks: Float32Array;
};

const video = document.querySelector<HTMLVideoElement>("#video")!;
const overlay = document.querySelector<HTMLCanvasElement>("#overlay")!;
const context = overlay.getContext("2d")!;
const status = document.querySelector<HTMLPreElement>("#status")!;
const widthSelect = document.querySelector<HTMLSelectElement>("#width")!;
const gpuConfirmed = document.querySelector<HTMLInputElement>("#gpu-confirmed")!;
const cameraButton = document.querySelector<HTMLButtonElement>("#camera")!;
const fileInput = document.querySelector<HTMLInputElement>("#file")!;
const exportButton = document.querySelector<HTMLButtonElement>("#export")!;
// MediaPipe's WASM loader publishes ModuleFactory through importScripts().
// A classic bundled worker preserves that global; a module worker scopes the
// loader variable and fails initialization with "ModuleFactory not set".
const worker = new TrackingWorker();

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

let stream: MediaStream | undefined;
let workerReady = false;
let running = false;
let inFlight = false;
let lastVideoTime = -1;
let frameId = 0;
let submittedFrames = 0;
let skippedBusyFrames = 0;
let detectedFrames = 0;
let lastDisplayAt = performance.now();
let displayFrames = 0;
const inferenceSamples: number[] = [];
const ageSamples: number[] = [];
const displayFrameSamples: number[] = [];
let latestResult: WorkerResult | undefined;
let mediaLabel = "none";
let sessionId = 0;
let lastSubmittedTimestampMs = -1;

function percentile(values: number[], ratio: number): number {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const index = (ordered.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.min(lower + 1, ordered.length - 1);
  const fraction = index - lower;
  return ordered[lower] * (1 - fraction) + ordered[upper] * fraction;
}

function boundedPush(values: number[], value: number): void {
  values.push(value);
  if (values.length > 1800) values.shift();
}

function webGlRenderer(): { renderer: string; gpuActive: boolean } {
  const testCanvas = document.createElement("canvas");
  const gl = testCanvas.getContext("webgl2");
  if (!gl) return { renderer: "WebGL2 unavailable", gpuActive: false };
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = info
    ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL))
    : String(gl.getParameter(gl.RENDERER));
  return {
    renderer,
    gpuActive: !/swiftshader|llvmpipe|software/i.test(renderer)
  };
}

const gpu = webGlRenderer();

function resizeOverlay(): void {
  overlay.width = video.videoWidth || 1280;
  overlay.height = video.videoHeight || 720;
}

function drawResult(result?: WorkerResult): void {
  context.clearRect(0, 0, overlay.width, overlay.height);
  if (!result) return;
  const values = result.landmarks;
  for (let hand = 0; hand < result.handCount; hand += 1) {
    const base = hand * 21 * 3;
    context.strokeStyle = hand === 0 ? "#52f7ff" : "#ff5cda";
    context.fillStyle = context.strokeStyle;
    context.lineWidth = 3;
    for (const [from, to] of connections) {
      context.beginPath();
      context.moveTo(
        values[base + from * 3] * overlay.width,
        values[base + from * 3 + 1] * overlay.height
      );
      context.lineTo(
        values[base + to * 3] * overlay.width,
        values[base + to * 3 + 1] * overlay.height
      );
      context.stroke();
    }
    for (let point = 0; point < 21; point += 1) {
      context.beginPath();
      context.arc(
        values[base + point * 3] * overlay.width,
        values[base + point * 3 + 1] * overlay.height,
        4,
        0,
        Math.PI * 2
      );
      context.fill();
    }
  }
}

function metrics() {
  const meanDisplayMs = displayFrameSamples.length
    ? displayFrameSamples.reduce((sum, value) => sum + value, 0) /
      displayFrameSamples.length
    : 0;
  return {
    schema_version: 1,
    backend: "mediapipe-web-gpu-worker",
    source: mediaLabel,
    inference_width: Number(widthSelect.value),
    submitted_frames: submittedFrames,
    skipped_busy_frames: skippedBusyFrames,
    detected_frames: detectedFrames,
    detection_rate: submittedFrames ? detectedFrames / submittedFrames : 0,
    display_fps: meanDisplayMs ? 1000 / meanDisplayMs : 0,
    mean_inference_ms: inferenceSamples.length
      ? inferenceSamples.reduce((sum, value) => sum + value, 0) /
        inferenceSamples.length
      : 0,
    p50_inference_ms: percentile(inferenceSamples, 0.5),
    p95_inference_ms: percentile(inferenceSamples, 0.95),
    p50_result_age_ms: percentile(ageSamples, 0.5),
    p95_result_age_ms: percentile(ageSamples, 0.95),
    gpu_delegate_requested: true,
    hardware_webgl_active: gpu.gpuActive,
    gpu_active: gpu.gpuActive && gpuConfirmed.checked,
    webgl_renderer: gpu.renderer,
    offline_assets: true,
    average_cpu_percent: null,
    tracking_accuracy: null,
    gesture_accuracy: null,
    hand_swaps: null
  };
}

function updateStatus(): void {
  const current = metrics();
  status.textContent = [
    `source: ${current.source}`,
    `renderer: ${current.webgl_renderer}`,
    `hardware WebGL: ${current.hardware_webgl_active}`,
    `GPU delegate verified: ${current.gpu_active}`,
    `display: ${current.display_fps.toFixed(1)} FPS`,
    `inference p50/p95: ${current.p50_inference_ms.toFixed(1)} / ${current.p95_inference_ms.toFixed(1)} ms`,
    `result age p50/p95: ${current.p50_result_age_ms.toFixed(1)} / ${current.p95_result_age_ms.toFixed(1)} ms`,
    `submitted: ${current.submitted_frames}, busy skips: ${current.skipped_busy_frames}, detected: ${current.detected_frames}`
  ].join("\n");
}

async function renderLoop(now: number, activeSession: number): Promise<void> {
  if (!running || activeSession !== sessionId) return;
  displayFrames += 1;
  boundedPush(displayFrameSamples, now - lastDisplayAt);
  lastDisplayAt = now;
  drawResult(latestResult);

  if (
    workerReady &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== lastVideoTime
  ) {
    if (inFlight) {
      skippedBusyFrames += 1;
    } else {
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      const targetWidth = Number(widthSelect.value);
      const targetHeight = Math.max(
        1,
        Math.round((sourceHeight * targetWidth) / sourceWidth)
      );
      inFlight = true;
      lastVideoTime = video.currentTime;
      const capturedAt = performance.now();
      const bitmap = await createImageBitmap(video, {
        resizeWidth: targetWidth,
        resizeHeight: targetHeight,
        resizeQuality: "medium"
      });
      frameId += 1;
      submittedFrames += 1;
      lastSubmittedTimestampMs = Math.max(
        lastSubmittedTimestampMs + 1,
        Math.round(performance.now())
      );
      worker.postMessage(
        {
          type: "frame",
          sessionId: activeSession,
          frameId,
          capturedAt,
          timestampMs: lastSubmittedTimestampMs,
          bitmap
        },
        [bitmap]
      );
    }
  }
  if (displayFrames % 15 === 0) updateStatus();
  requestAnimationFrame((timestamp) => renderLoop(timestamp, activeSession));
}

function resetMetrics(): void {
  submittedFrames = 0;
  skippedBusyFrames = 0;
  detectedFrames = 0;
  displayFrames = 0;
  inferenceSamples.length = 0;
  ageSamples.length = 0;
  displayFrameSamples.length = 0;
  latestResult = undefined;
  lastVideoTime = -1;
  lastDisplayAt = performance.now();
}

async function beginPlayback(label: string): Promise<void> {
  sessionId += 1;
  mediaLabel = label;
  resetMetrics();
  resizeOverlay();
  running = true;
  exportButton.disabled = false;
  await video.play();
  const activeSession = sessionId;
  requestAnimationFrame((timestamp) => renderLoop(timestamp, activeSession));
}

cameraButton.addEventListener("click", async () => {
  stream?.getTracks().forEach((track) => track.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, frameRate: 30 },
    audio: false
  });
  video.srcObject = stream;
  video.onloadedmetadata = () => beginPlayback("camera");
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  video.onloadedmetadata = () => beginPlayback(file.name);
});

const benchmarkParams = new URLSearchParams(window.location.search);
const requestedWidth = benchmarkParams.get("width");
if (
  requestedWidth
  && Array.from(widthSelect.options).some(
    (option) => option.value === requestedWidth
  )
) {
  widthSelect.value = requestedWidth;
}
const fixtureUrl = benchmarkParams.get("fixture");
if (fixtureUrl) {
  video.srcObject = null;
  video.loop = true;
  video.onloadedmetadata = () => beginPlayback(`fixture:${fixtureUrl}`);
  video.src = fixtureUrl;
  video.load();
}

exportButton.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(metrics(), null, 2)], {
    type: "application/json"
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `web-gpu-${Number(widthSelect.value)}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

worker.onmessage = (event) => {
  if (event.data.type === "ready") {
    workerReady = true;
    updateStatus();
    return;
  }
  if (event.data.type === "error") {
    inFlight = false;
    status.textContent = `Worker error: ${event.data.message}`;
    return;
  }
  const result = event.data as WorkerResult;
  if (result.sessionId !== sessionId) {
    inFlight = false;
    return;
  }
  latestResult = result;
  inFlight = false;
  boundedPush(inferenceSamples, result.inferenceMs);
  boundedPush(ageSamples, performance.now() - result.capturedAt);
  if (result.handCount > 0) detectedFrames += 1;
};

worker.onerror = (event) => {
  inFlight = false;
  status.textContent = `Worker crashed: ${event.message}`;
};

worker.onmessageerror = () => {
  inFlight = false;
  status.textContent = "Worker returned an unreadable message.";
};

worker.postMessage({
  type: "init",
  wasmRoot: new URL("wasm", document.baseURI).href.replace(/\/$/, ""),
  modelPath: new URL(
    "models/hand_landmarker.task",
    document.baseURI
  ).href
});
