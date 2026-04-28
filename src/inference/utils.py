"""
Utility functions for image and video processing in the parking detection pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import CLASS_COLORS, OUTPUT_DIR


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes on an image given a list of detection dicts.

    Each detection should have keys: x1, y1, x2, y2, class_name, confidence.
    Returns a copy of the image with annotations.
    """
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cls = det["class_name"]
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    return annotated


def save_annotated_image(image: np.ndarray, filename: str, output_dir: Path | None = None) -> Path:
    """
    Save an annotated image to the output directory.

    Returns the full path of the saved file.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), image)
    return out_path


def extract_video_frames(
    video_path: str | Path,
    interval_seconds: float = 2.0,
) -> list[np.ndarray]:
    """
    Extract frames from a video at the given interval (in seconds).

    Returns a list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * interval_seconds))

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


def image_to_bytes(image: np.ndarray, fmt: str = ".jpg") -> bytes:
    """Encode an image (BGR numpy array) to bytes in the given format."""
    success, buf = cv2.imencode(fmt, image)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()
