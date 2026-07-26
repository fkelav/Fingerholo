import { cp, mkdir, stat } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasmSource = path.join(
  root,
  "node_modules",
  "@mediapipe",
  "tasks-vision",
  "wasm"
);
const wasmDestination = path.join(root, "public", "wasm");
const modelDestination = path.join(
  root,
  "public",
  "models",
  "hand_landmarker.task"
);

await mkdir(path.dirname(modelDestination), { recursive: true });
await cp(wasmSource, wasmDestination, { recursive: true, force: true });

try {
  await stat(modelDestination);
  console.log(`Model already present: ${modelDestination}`);
} catch {
  execFileSync(
    "python",
    [
      path.join(root, "..", "tools", "download_models.py"),
      "--output",
      modelDestination
    ],
    { stdio: "inherit" }
  );
}
console.log(`Copied MediaPipe WASM assets to ${wasmDestination}`);
