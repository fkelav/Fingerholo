# Web GPU tracking benchmark

This prototype measures the GPU path before any full application rewrite. It
uses MediaPipe Tasks Vision in a Web Worker, allows only one frame in flight,
drops stale submissions, accepts a webcam or recorded clip, and exports JSON
metrics.

```powershell
cd gpu_benchmark
npm install
npm run prepare-assets
npm run dev
```

Use the same recorded clip and inference widths as the Python diagnostic. The
model and WASM assets are copied into `public/`, so the benchmark works without
network access after preparation.

For repeatable automation, copy a recorded clip into `public/` and pass its
local URL:

```text
http://127.0.0.1:5173/?fixture=/fixture.mp4&width=480
```

If an OpenCV MP4 does not play in the browser, run
`python tools\prepare_web_fixture.py VIDEO` from the repository root and use
`/?fixture=/fixture.webm`.

The exported report detects obvious software WebGL renderers. Confirm the
MediaPipe GPU backend in browser diagnostics, then tick the verification box;
`gpu_active` remains false until both checks pass. The benchmark intentionally
does not request a CPU delegate.
