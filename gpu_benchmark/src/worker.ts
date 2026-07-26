/// <reference lib="webworker" />

import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult
} from "@mediapipe/tasks-vision";

type InitMessage = {
  type: "init";
  wasmRoot: string;
  modelPath: string;
};

type FrameMessage = {
  type: "frame";
  sessionId: number;
  frameId: number;
  capturedAt: number;
  timestampMs: number;
  bitmap: ImageBitmap;
};

let landmarker: HandLandmarker | undefined;

function flattenLandmarks(result: HandLandmarkerResult): Float32Array {
  const output = new Float32Array(result.landmarks.length * 21 * 3);
  let offset = 0;
  for (const hand of result.landmarks) {
    for (const point of hand) {
      output[offset++] = point.x;
      output[offset++] = point.y;
      output[offset++] = point.z;
    }
  }
  return output;
}

self.onmessage = async (event: MessageEvent<InitMessage | FrameMessage>) => {
  const message = event.data;
  if (message.type === "init") {
    try {
      const vision = await FilesetResolver.forVisionTasks(message.wasmRoot);
      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: message.modelPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "error", message: String(error) });
    }
    return;
  }

  if (!landmarker) {
    message.bitmap.close();
    self.postMessage({ type: "error", message: "Worker is not initialized." });
    return;
  }

  const startedAt = performance.now();
  try {
    const result = landmarker.detectForVideo(message.bitmap, message.timestampMs);
    const finishedAt = performance.now();
    const landmarks = flattenLandmarks(result);
    const handedness = result.handedness.map((categories) => {
      const category = categories[0];
      return {
        label: category?.categoryName ?? "",
        score: category?.score ?? 0
      };
    });
    self.postMessage(
      {
        type: "result",
        sessionId: message.sessionId,
        frameId: message.frameId,
        capturedAt: message.capturedAt,
        resultAt: finishedAt,
        inferenceMs: finishedAt - startedAt,
        handCount: result.landmarks.length,
        handedness,
        landmarks
      },
      [landmarks.buffer]
    );
  } catch (error) {
    self.postMessage({ type: "error", message: String(error) });
  } finally {
    message.bitmap.close();
  }
};

export {};
