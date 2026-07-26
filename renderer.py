"""Fast video-overlay renderer for the finger hologram effect."""

from pathlib import Path

import cv2
import numpy as np

from geometry import is_simple_convex_quadrilateral


OVERLAY_SPECS = {
    "1": {
        "label": "DARK VHS PANEL",
        "mode": "square_panel",
        "file": "dark_vhs_panel.mp4",
    },
    "2": {
        "label": "COLOR DATAMOSH STRIP",
        "mode": "wide_strip",
        "file": "color_datamosh_strip.mp4",
    },
    "3": {
        "label": "STATIC GLITCH STRIP",
        "mode": "vertical_strip",
        "file": "static_glitch_strip.mp4",
    },
    "4": {
        "label": "DARK VHS OVERLAY",
        "mode": "wide_strip",
        "file": "vhs_overlay.mp4",
    },
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
# These keys already control the application. Numeric keys 1-4 are intentionally
# not reserved, so a custom file can replace one of the bundled overlays.
RESERVED_CUSTOM_KEYS = set("qhcsrgbduxvpi[]-_=+")

# The source frames stay small for speed. They are perspective-warped into the
# full-resolution camera only inside the hand panel's bounding box.
MODE_SIZES = {
    "square_panel": (320, 320),
    "wide_strip": (480, 180),
    "vertical_strip": (180, 480),
}
VIDEO_UPDATE_INTERVAL = 3
MIN_INVERTED_CAMERA_WEIGHT = 0.35
FINGER_NUMBER_BY_TIP_ID = {4: 1, 8: 2, 12: 3, 16: 4, 20: 5}


class OverlayRenderer:
    """Render built-in or user media into fingertip-driven panels."""

    def __init__(self, assets_folder, opacity=0.68, custom_folder=None):
        self.assets_folder = Path(assets_folder)
        self.custom_folder = Path(custom_folder) if custom_folder else None
        self.overlay_specs = {
            key: {**spec, "path": self.assets_folder / spec["file"], "custom": False}
            for key, spec in OVERLAY_SPECS.items()
        }
        self._discover_custom_media()
        self.selected_key = "1"
        self.opacity = float(np.clip(opacity, 0.10, 1.0))
        self.inversion_enabled = False
        self._captures = {}
        self._frame_cache = {}
        self._render_counter = 0
        self._open_videos()

    @property
    def label(self):
        return self.overlay_specs[self.selected_key]["label"]

    @property
    def mode(self):
        return self.overlay_specs[self.selected_key]["mode"]

    @property
    def custom_keys(self):
        return tuple(
            key for key, spec in self.overlay_specs.items() if spec.get("custom")
        )

    def has_key(self, key):
        return key in self.overlay_specs

    def select(self, key):
        if key in self.overlay_specs:
            self.selected_key = key

    def adjust_opacity(self, amount):
        self.opacity = float(np.clip(self.opacity + amount, 0.10, 1.0))
        print(f"Overlay opacity: {self.opacity:.0%}")
        return self.opacity

    def toggle_inversion(self):
        self.inversion_enabled = not self.inversion_enabled
        return self.inversion_enabled

    @property
    def selected_asset_available(self):
        return (
            self.selected_key in self._captures
            or self.selected_key in self._frame_cache
        )

    def _discover_custom_media(self):
        if self.custom_folder is None or not self.custom_folder.is_dir():
            return

        for path in sorted(self.custom_folder.iterdir(), key=lambda item: item.name.lower()):
            extension = path.suffix.lower()
            if not path.is_file() or extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                continue
            key = path.stem.lower()
            if len(key) != 1 or not key.isprintable() or key.isspace():
                print(f"Warning: custom media filename must be one hotkey character: {path.name}")
                continue
            if key in RESERVED_CUSTOM_KEYS:
                print(f"Warning: custom media key '{key}' is reserved: {path.name}")
                continue
            if key in self.overlay_specs and self.overlay_specs[key].get("custom"):
                print(f"Warning: duplicate custom media key '{key}' ignored: {path.name}")
                continue
            media_type = "image" if extension in IMAGE_EXTENSIONS else "video"
            self.overlay_specs[key] = {
                "label": f"CUSTOM {path.name}",
                "mode": "wide_strip",
                "path": path,
                "custom": True,
                "media_type": media_type,
            }

    def _open_videos(self):
        for key, spec in self.overlay_specs.items():
            path = spec["path"]
            if spec.get("media_type") == "image":
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Warning: could not open image for key {key}: {path}")
                    continue
                spec["mode"] = self._mode_for_dimensions(image.shape[1], image.shape[0])
                self._frame_cache[key] = self._prepare_media(image, spec["mode"])
                continue

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                print(f"Warning: could not open video for key {key}: {path}")
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if spec.get("custom"):
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                spec["mode"] = self._mode_for_dimensions(width, height)
            self._captures[key] = capture

    def render(
        self,
        camera_frame,
        quadrilateral,
        elapsed_seconds,
        split_level=0,
        advance=True,
        overlay=None,
    ):
        """Blend a video—not the camera image—inside the four points."""
        if overlay is None:
            overlay = self.overlay_for_frame(
                elapsed_seconds, split_level=split_level, advance=advance
            )

        if overlay is None:
            self._draw_missing_asset(camera_frame, quadrilateral)
            return False

        self._warp_and_blend_roi(
            camera_frame,
            overlay,
            quadrilateral,
            self.opacity,
            invert_camera=self.inversion_enabled,
        )
        self._draw_subtle_edge(camera_frame, quadrilateral)
        return True

    def overlay_for_frame(self, elapsed_seconds, split_level=0, advance=True):
        """Prepare one media frame that can be shared by every active effect."""
        if advance:
            self._render_counter += 1
        if split_level > 0:
            return self._make_split_overlay(split_level, elapsed_seconds)
        return self._video_overlay(self.selected_key)

    def render_hands(
        self,
        camera_frame,
        hands,
        elapsed_seconds,
        split_level=0,
        advance=True,
        overlay=None,
        glow_enabled=False,
        border_enabled=False,
    ):
        """Draw one solid fingertip-corner media panel on every visible hand."""
        hands = tuple(hands)
        if not hands:
            return False
        if overlay is None:
            overlay = self.overlay_for_frame(
                elapsed_seconds, split_level=split_level, advance=advance
            )
        if overlay is None:
            return False

        rendered = False
        for hand in hands:
            if hasattr(hand, "landmarks"):
                landmarks = hand.landmarks
                palm_scale = hand.palm_scale
            else:
                landmarks, palm_scale = hand
            rendered = self._render_fingertip_panel(
                camera_frame,
                overlay,
                landmarks,
                palm_scale,
                glow_enabled=glow_enabled,
                border_enabled=border_enabled,
            ) or rendered
        return rendered

    def _video_overlay(self, key):
        """Decode video at 10 FPS while tracking geometry stays at 30 FPS."""
        cached = self._frame_cache.get(key)
        key_phase = ord(key) % VIDEO_UPDATE_INTERVAL
        should_decode = (
            cached is None
            or self._render_counter % VIDEO_UPDATE_INTERVAL == key_phase
        )

        if should_decode:
            capture = self._captures.get(key)
            if capture is None:
                return cached
            frame = self._read_video_frame(capture)
            if frame is not None:
                mode = self.overlay_specs[key]["mode"]
                self._frame_cache[key] = self._prepare_media(frame, mode)

        return self._frame_cache.get(key)

    def _make_split_overlay(self, split_level, elapsed_seconds):
        """Create the dark-top/color-bottom panel from two video streams."""
        dark = self._video_overlay("1")
        color = self._video_overlay("2")
        if dark is None or color is None:
            return dark if dark is not None else color

        target_width, target_height = MODE_SIZES["wide_strip"]
        dark = cv2.resize(dark, (target_width, target_height))
        color = cv2.resize(color, (target_width, target_height))
        output = dark.copy()

        split_y = int(target_height * 0.40)
        phase = int(elapsed_seconds * 15)
        horizontal_shift = int(np.sin(phase * 0.5) * target_width * 0.012)
        color = np.roll(color, horizontal_shift, axis=1)
        output[split_y:] = color[split_y:]

        seam = 2
        output[split_y - seam : split_y + seam, :, :3] = 240

        if split_level >= 2:
            bottom = int(target_height * 0.84)
            output[bottom:, :, :3] = cv2.addWeighted(
                output[bottom:, :, :3],
                0.45,
                np.full_like(output[bottom:, :, :3], 235),
                0.55,
                0,
            )
        return output

    @staticmethod
    def _mode_for_dimensions(width, height):
        if width <= 0 or height <= 0:
            return "wide_strip"
        ratio = width / height
        if ratio < 0.8:
            return "vertical_strip"
        if ratio < 1.3:
            return "square_panel"
        return "wide_strip"

    @classmethod
    def _prepare_media(cls, image, mode):
        image = cls._prepare_for_mode(image, mode)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        if image.shape[2] == 4:
            return image
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        return rgba

    @staticmethod
    def _read_video_frame(capture):
        ok, frame = capture.read()
        if ok:
            return frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
        return frame if ok else None

    @staticmethod
    def _prepare_for_mode(image, mode):
        target_width, target_height = MODE_SIZES[mode]

        # Rotate the source for the tall-strip mode before cropping. This keeps
        # the original video content recognizable instead of squashing it.
        if mode == "vertical_strip" and image.shape[1] > image.shape[0]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        source_height, source_width = image.shape[:2]
        target_ratio = target_width / target_height
        source_ratio = source_width / source_height

        if source_ratio > target_ratio:
            crop_width = max(1, round(source_height * target_ratio))
            left = (source_width - crop_width) // 2
            image = image[:, left : left + crop_width]
        else:
            crop_height = max(1, round(source_width / target_ratio))
            top = (source_height - crop_height) // 2
            image = image[top : top + crop_height, :]

        return cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _warp_and_blend_roi(
        frame, overlay, quadrilateral, opacity, invert_camera=False
    ):
        """Perspective-warp only the panel area, then blend it transparently."""
        points = np.asarray(quadrilateral, dtype=np.float32)
        if not is_simple_convex_quadrilateral(points, minimum_area=4.0):
            return
        frame_height, frame_width = frame.shape[:2]
        margin = 4
        left = max(0, int(np.floor(points[:, 0].min())) - margin)
        top = max(0, int(np.floor(points[:, 1].min())) - margin)
        right = min(frame_width, int(np.ceil(points[:, 0].max())) + margin + 1)
        bottom = min(frame_height, int(np.ceil(points[:, 1].max())) + margin + 1)
        roi_width = right - left
        roi_height = bottom - top
        if roi_width < 2 or roi_height < 2:
            return

        overlay_height, overlay_width = overlay.shape[:2]
        source_points = np.array(
            [
                [0, 0],
                [overlay_width - 1, 0],
                [overlay_width - 1, overlay_height - 1],
                [0, overlay_height - 1],
            ],
            dtype=np.float32,
        )
        local_points = points - np.array([left, top], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source_points, local_points)
        warped = cv2.warpPerspective(
            overlay,
            matrix,
            (roi_width, roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        roi = frame[top:bottom, left:right]
        media_alpha = warped[:, :, 3].astype(np.float32) / 255.0
        OverlayRenderer._blend_over_camera(
            roi,
            warped[:, :, :3],
            media_alpha,
            opacity,
            invert_camera=invert_camera,
        )

    @staticmethod
    def _blend_over_camera(
        roi, media, region_alpha, opacity, invert_camera=False
    ):
        """Composite media over normal or inverted camera pixels."""
        source = roi.astype(np.float32)
        if invert_camera:
            background = 255.0 - source
            media_weight = opacity * (1.0 - MIN_INVERTED_CAMERA_WEIGHT)
        else:
            background = source
            media_weight = opacity
        effect = (
            media.astype(np.float32) * media_weight
            + background * (1.0 - media_weight)
        )
        alpha = np.asarray(region_alpha, dtype=np.float32)
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        roi[:] = np.clip(
            effect * alpha + source * (1.0 - alpha), 0, 255
        ).astype(np.uint8)

    @staticmethod
    def _draw_subtle_edge(frame, quadrilateral):
        points = np.asarray(quadrilateral, dtype=np.int32)
        cv2.polylines(frame, [points], True, (220, 220, 220), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_missing_asset(frame, quadrilateral):
        center = np.mean(quadrilateral, axis=0).astype(int)
        cv2.putText(
            frame,
            "MEDIA MISSING",
            (center[0] - 90, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def fingertip_panel_polygon(landmarks):
        """Return a single panel whose upper corners are the five fingertips."""
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            return None
        # Thumb -> pinky traces the visible finger edge.  The three palm-side
        # points close it into one contiguous, square-like panel instead of a
        # glove silhouette with five separate finger strips.
        return points[[4, 8, 12, 16, 20, 17, 0, 2]].copy()

    @classmethod
    def build_hand_mask(cls, frame_shape, landmarks, palm_scale=None):
        """Return a feathered mask for the solid fingertip-corner panel."""
        height, width = frame_shape[:2]
        polygon = cls.fingertip_panel_polygon(landmarks)
        if polygon is None:
            return np.zeros((height, width), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255, cv2.LINE_AA)
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=0.8, sigmaY=0.8)

    def _render_fingertip_panel(
        self,
        frame,
        overlay,
        landmarks,
        palm_scale,
        glow_enabled=False,
        border_enabled=False,
    ):
        points = np.asarray(landmarks, dtype=np.float32)
        polygon = self.fingertip_panel_polygon(points)
        if polygon is None:
            return False
        frame_height, frame_width = frame.shape[:2]
        margin = 4
        left = max(0, int(np.floor(polygon[:, 0].min())) - margin)
        top = max(0, int(np.floor(polygon[:, 1].min())) - margin)
        right = min(frame_width, int(np.ceil(polygon[:, 0].max())) + margin + 1)
        bottom = min(frame_height, int(np.ceil(polygon[:, 1].max())) + margin + 1)
        if right - left < 2 or bottom - top < 2:
            return False

        origin = points[[0, 5, 9, 13, 17]].mean(axis=0)
        axis_y = points[9] - points[0]
        norm = float(np.linalg.norm(axis_y))
        if norm <= 1e-6:
            axis_y = np.array([0.0, -1.0], dtype=np.float32)
        else:
            axis_y /= norm
        axis_x = np.array([axis_y[1], -axis_y[0]], dtype=np.float32)
        relative = polygon - origin
        projected_x = relative @ axis_x
        projected_y = relative @ axis_y
        min_x, max_x = projected_x.min(), projected_x.max()
        min_y, max_y = projected_y.min(), projected_y.max()
        target = np.array(
            [
                # axis_y points from the wrist toward the fingertips, so the
                # source image's top edge belongs at max_y. The old min_y-first
                # order put every custom image upside down on hand panels.
                origin + axis_x * min_x + axis_y * max_y,
                origin + axis_x * max_x + axis_y * max_y,
                origin + axis_x * max_x + axis_y * min_y,
                origin + axis_x * min_x + axis_y * min_y,
            ],
            dtype=np.float32,
        ) - np.array([left, top], dtype=np.float32)
        overlay_height, overlay_width = overlay.shape[:2]
        source = np.array(
            [[0, 0], [overlay_width - 1, 0], [overlay_width - 1, overlay_height - 1], [0, overlay_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(
            overlay,
            matrix,
            (right - left, bottom - top),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        roi = frame[top:bottom, left:right]
        local_polygon = np.rint(
            polygon - np.array([left, top], dtype=np.float32)
        ).astype(np.int32)
        mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon], 255, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.8, sigmaY=0.8)
        hand_alpha = mask.astype(np.float32) / 255.0
        media_alpha = warped[:, :, 3].astype(np.float32) / 255.0
        region_alpha = hand_alpha * media_alpha
        self._blend_over_camera(
            roi,
            warped[:, :, :3],
            region_alpha,
            self.opacity,
            invert_camera=getattr(self, "inversion_enabled", False),
        )

        binary = np.where(mask > 80, 255, 0).astype(np.uint8)
        if glow_enabled:
            glow = cv2.GaussianBlur(binary, (0, 0), sigmaX=8, sigmaY=8)
            color = np.zeros_like(roi)
            color[:, :, 0] = glow
            color[:, :, 2] = glow
            cv2.addWeighted(roi, 1.0, color, 0.22, 0, dst=roi)
        if border_enabled:
            cv2.polylines(
                roi, [local_polygon], True, (245, 245, 245), 1, cv2.LINE_AA
            )
        return True

    def close(self):
        for capture in self._captures.values():
            capture.release()


def draw_neon_border(frame, quadrilateral, glow_enabled=True, border_enabled=True):
    """Optional debug border; disabled by default in main.py."""
    if not glow_enabled and not border_enabled:
        return

    points = np.asarray(quadrilateral, dtype=np.int32)
    if glow_enabled:
        x, y, width, height = cv2.boundingRect(points)
        frame_height, frame_width = frame.shape[:2]
        padding = 24
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(frame_width, x + width + padding)
        bottom = min(frame_height, y + height + padding)
        layer = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)
        local_points = points - np.array([left, top])
        cv2.polylines(layer, [local_points], True, (255, 255, 255), 7, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(layer, (0, 0), sigmaX=9, sigmaY=9)
        roi = frame[top:bottom, left:right]
        cv2.addWeighted(roi, 1.0, blurred, 0.65, 0, dst=roi)

    if border_enabled:
        cv2.polylines(frame, [points], True, (245, 245, 245), 1, cv2.LINE_AA)


def draw_fingertips(frame, quadrilateral):
    """Optional tracking markers, hidden by default."""
    for point in np.asarray(quadrilateral, dtype=np.int32):
        cv2.circle(frame, tuple(point), 7, (255, 80, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_finger_labels(
    frame, tracked_hands, drawing_active=False, selected_finger_ids=None
):
    """Label all fingertips; blue numbers feed the current panel geometry.

    Finger numbering is 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky.
    The current renderer only uses fingers 1 and 2 from both hands, and only
    when a valid two-hand quadrilateral is being rendered.
    """
    frame_height, frame_width = frame.shape[:2]
    active_ids = set()
    if drawing_active and selected_finger_ids:
        active_ids = {
            tip_id for pair in selected_finger_ids for tip_id in pair
        }
    active_numbers = {
        FINGER_NUMBER_BY_TIP_ID[tip_id]
        for tip_id in active_ids
        if tip_id in FINGER_NUMBER_BY_TIP_ID
    }

    for fingertips in tracked_hands:
        for finger_number, point in enumerate(
            np.asarray(fingertips, dtype=np.int32), start=1
        ):
            x, y = (int(point[0]), int(point[1]))
            label_x = int(np.clip(x, 14, frame_width - 15))
            label_y = int(np.clip(y - 20, 14, frame_height - 15))
            active = finger_number in active_numbers
            number_color = (255, 115, 20) if active else (245, 245, 245)
            border_color = number_color if active else (125, 125, 125)

            cv2.line(
                frame,
                (x, y),
                (label_x, label_y + 10),
                border_color,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (x, y), 3, border_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (label_x, label_y), 12, (18, 18, 18), -1, cv2.LINE_AA)
            cv2.circle(frame, (label_x, label_y), 12, border_color, 2, cv2.LINE_AA)

            text = str(finger_number)
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2
            )
            text_origin = (
                label_x - text_size[0] // 2,
                label_y + text_size[1] // 2,
            )
            cv2.putText(
                frame,
                text,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                number_color,
                2,
                cv2.LINE_AA,
            )
