# Hand tracking model

`hand_landmarker.task` is MediaPipe's Hand Landmarker float16 model, version 1.
The runtime and Web GPU benchmark keep local copies so detection works offline.

To restore or verify the model:

```powershell
python tools\download_models.py
```

The downloader accepts only this SHA-256:

```text
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

Source: <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
