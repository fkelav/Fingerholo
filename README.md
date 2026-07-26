# Finger Hologram

A local real-time OpenCV/MediaPipe effect that tracks up to two hands and
perspective-warps image or looping-video overlays onto selected fingertips and
palms.

The default `auto` tracker uses MediaPipe Tasks Hand Landmarker in non-blocking
live-stream mode when the bundled model and Tasks API are available. It keeps
only one inference frame in flight. In `auto` mode, a three-second warmup keeps
Tasks on machines that sustain it and switches slow or heavily dropping
systems to the lightweight legacy MediaPipe Hands backend. Explicit `tasks`
mode never falls back. Camera capture and video recording also use bounded
background pipelines, so a slow detector or encoder cannot create an
ever-growing latency queue.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

The verified model is stored at `models/hand_landmarker.task`. Restore it if
needed with:

```powershell
python tools\download_models.py
```

Use `python main.py --help` for all command-line options. Copy
`config.example.json` to make persistent settings:

```powershell
Copy-Item config.example.json config.json
python main.py --config config.json
```

Command-line values override JSON values. Tracking-specific settings include
`tracking_backend` (`auto`, `tasks`, or `legacy`), `processing_width`,
`detection_sensitivity`, `tracking_grace_seconds`,
`max_tracking_result_age_seconds`, and `hand_model_path`. Recording uses the
bounded `recording_queue_size`. Use `{timestamp}` in `output_filename` and
`performance_output_filename` for collision-resistant names.

The preview help shows the active backend, inference time, result age, and
dropped detector submissions. On exit, the console shows a compact,
human-readable performance summary and saves the complete gate-ready JSON report
under `artifacts/performance_{timestamp}.json` by default. The saved report
retains the active backend, Python/platform metadata, average display FPS, p95
result age, CPU use, dropped frames, and every per-stage timing. Recovered
geometry can remain visible briefly, but stale landmarks never enter gesture
recognition and therefore cannot toggle effects.

The packaged Windows executable keeps its console open after shutdown so the
summary remains readable. Press Enter when you are ready to close the window.

## Build a Windows executable

Install the pinned packaging dependency, then create the executable:

```powershell
python -m pip install -r requirements-build.txt
New-Item -ItemType Directory -Force build | Out-Null
python -m PyInstaller --noconfirm --clean --onedir `
  --contents-directory . `
  --name FingerHologram `
  --add-data "$PWD\assets;assets" `
  --add-data "$PWD\models;models" `
  --add-data "$PWD\custom\README.md;custom" `
  --collect-data mediapipe `
  --collect-binaries mediapipe `
  --specpath build `
  --workpath build\pyinstaller `
  --distpath dist `
  main.py
```

The packaged app is `dist\FingerHologram\FingerHologram.exe`. Its folder
includes the required model and bundled overlays. Put personal media in its
`custom` folder. The generated `build` and `dist` folders are intentionally
ignored by Git.

## Gestures

- Touch two distinct fingertips from each hand together in one four-tip cluster to instantly open a panel attached to those four selected fingers.
- Tap thumb and index on both hands, or touch one opposing fingertip pair, to close and clear the selected panel.
- Hold two fully closed fists briefly to toggle the two hand panels on or off. Curl all four fingers and wrap each thumb across the front of the fist.
- The hand effect is one solid media panel per hand: its upper edge has a corner on every fingertip and its lower edge closes across the palm.
- The fist toggle is release-gated, so holding the pose cannot immediately toggle the hand panels again.

The main panel and hand panels are independent and can be visible together. If tracking is lost, their enabled state is remembered and rendering resumes when the required hands return.

## In-app controls

- `H`: toggle complete controls and live status help
- `1`–`4`: choose a bundled overlay and leave forced split mode
- Any custom media hotkey: choose the matching file from `custom` (for example, `5.mp4` uses `5` and `a.png` uses `A`)
- `[` / `]`: decrease/increase opacity
- `-` / `+`: decrease/increase detection sensitivity
- `C`: toggle calibration and hand-distance guidance
- `S`: toggle smoothing; `R`: reset tracking recovery
- `D`: fingertip labels; `G`: glow; `B`: border; `X`: force split effect
- `I`: toggle camera inversion beneath the hologram
- `Space`: start/stop recording with an on-screen timer
- `P`: save a timestamped screenshot
- `V`: include/exclude the camera background in recordings
- `U`: hide or show every HUD overlay
- `Q` or `Escape`: quit

Letter controls accept lowercase and uppercase input. Recording and screenshot failures are reported inside the preview window.

## Custom images and videos

Put an image or video in the `custom` folder and give it a one-character filename. The filename (without its extension) becomes the hotkey: `5.mp4` is selected with `5`, and `a.png` is selected with `A`. Supported images are BMP, JPEG, PNG, TIFF, and WebP; supported videos are AVI, M4V, MKV, MOV, MP4, and WebM.

Custom files named `1` through `4` replace the matching bundled overlay. The app's control keys (`Q`, `H`, `C`, `S`, `R`, `G`, `B`, `D`, `U`, `X`, `V`, and `P`, plus the punctuation controls) are reserved and cannot be used for custom media.

Personal files under `custom` are ignored by Git; only `custom\README.md` is
part of the repository.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Performance diagnostics

Create per-frame CSV and percentile JSON reports from the same recorded clip:

```powershell
python tools\tracking_diagnostic.py input.mp4 `
  --gesture-metrics `
  --json-output artifacts\tracking.json `
  --csv-output artifacts\tracking.csv

python tools\benchmark_tracking.py input.mp4 `
  --widths 320 480 640 `
  --json-output artifacts\tracking_widths.json `
  --csv-output artifacts\tracking_widths.csv

python tools\tracking_diagnostic.py input.mp4 `
  --backend tasks --realtime --stride 1 `
  --json-output artifacts\tasks_live.json
```

The diagnostic reports preprocessing, inference, tracking, and gesture
timings. Use identical time ranges and crops when comparing widths or
backends. Use `--realtime` with the asynchronous Tasks backend so recorded
frames arrive at camera cadence instead of flooding its one-frame queue.

Normal webcam runs also save a report automatically. Override its destination
when collecting named machine profiles:

```powershell
python main.py --performance-output `
  artifacts\office_integrated_gpu_{timestamp}.json
```

## Web GPU benchmark and rewrite gate

`gpu_benchmark` is a separate TypeScript/Vite experiment, not the production
app. It runs MediaPipe Tasks Vision with the GPU delegate in a Web Worker,
allows one frame in flight, accepts webcam or recorded-video input, and exports
result-age/inference metrics.

```powershell
Set-Location gpu_benchmark
npm install
npm run prepare-assets
npm run dev
```

After `prepare-assets`, its model and WASM files are local and the benchmark
runs offline. Test 480-pixel and 720p inputs on the target machine and confirm
that the report says `gpu_active: true`.

OpenCV recordings use a codec that some browsers cannot decode. Prepare the
same hand-tracking clip as a local VP8 fixture when needed:

```powershell
python tools\prepare_web_fixture.py `
  output\finger_hologram_20260723_094013.mp4
```

Then open `http://127.0.0.1:5173/?fixture=/fixture.webm`.

Record CPU use and labeled tracking/gesture accuracy in the exported reports,
then apply the objective rewrite gate:

```powershell
python tools\evaluate_rewrite_gate.py `
  artifacts\python_metrics.json `
  artifacts\web_metrics.json
```

A GPU path is approved for a hardware tier only when every gate passes: at
least 30% lower p95 result age, 30 FPS display, at least 25% lower CPU use,
accuracy within two percentage points, offline assets, and confirmed GPU
acceleration.

Do not base the product decision on one laptop. Collect paired Python/Web
reports for at least a CPU-only or older integrated-GPU machine, a current
integrated-GPU machine, and a discrete-GPU machine. The current Python runtime
remains the compatibility fallback; a Web GPU sidecar or later Tauri edition
can be enabled for hardware tiers that pass. Python MediaPipe Tasks uses CPU on
Windows, so a stronger GPU helps only the Web benchmark/future WebGL path,
while a stronger CPU also improves the Python application.

## License

[MIT](LICENSE).
