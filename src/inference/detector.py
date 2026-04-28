"""
ParkingDetector — Runs YOLO-based inference on parking lot images/videos.

Uses the pre-trained best1.pt model to detect free and occupied parking spaces.
Class 0 = Free (Empty), Class 1 = Occupied.
"""

import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_NAMES, CLASS_COLORS


class ParkingDetector:
    """Wrapper around the YOLO model for parking space detection."""

    def __init__(self, model_path: str | Path | None = None, conf: float | None = None):
        self.model_path = str(model_path or MODEL_PATH)
        self.conf = conf or CONFIDENCE_THRESHOLD
        print(f"[INFO] Loading model from {self.model_path} ...")
        self.model = YOLO(self.model_path)
        print("[INFO] Model loaded successfully.")

    # ── Single Image ────────────────────────────────────────────
    def detect_image(self, image_input: str | Path | np.ndarray) -> dict[str, Any]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_input : str, Path, or np.ndarray
            Path to image file or a raw BGR numpy array.

        Returns
        -------
        dict with keys:
            run_id           – unique UUID for this run
            free             – count of free spaces
            occupied         – count of occupied spaces
            total            – free + occupied
            detections       – list of dicts {x1, y1, x2, y2, class_name, confidence}
            annotated_image  – BGR numpy array with drawn boxes
            processing_ms    – inference time in milliseconds
            input_filename   – original filename (or "array" for raw input)
        """
        start = time.perf_counter()

        # Load image if path is given
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            input_filename = Path(image_input).name
        else:
            img = image_input.copy()
            input_filename = "array"

        if img is None:
            raise ValueError(f"Could not read image: {image_input}")

        # Run YOLO inference
        results = self.model.predict(source=img, conf=self.conf, save=False, verbose=False)

        # Parse results
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        free = 0
        occupied = 0
        detections: list[dict] = []

        for box, cls, conf_val in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(int(cls), "unknown")
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            if class_name == "free":
                free += 1
            else:
                occupied += 1

            # Draw bounding box on image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "class_name": class_name,
                "confidence": float(conf_val),
            })

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "run_id": str(uuid.uuid4()),
            "free": free,
            "occupied": occupied,
            "total": free + occupied,
            "detections": detections,
            "annotated_image": img,
            "processing_ms": round(elapsed_ms, 2),
            "input_filename": input_filename,
        }

    # ── Video ───────────────────────────────────────────────────
    def detect_video(
        self,
        video_path: str | Path,
        sample_interval: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Process a video file, sampling every *sample_interval* frames.

        Returns a list of result dicts (same shape as detect_image output,
        but *annotated_image* is excluded to save memory).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        results_list: list[dict] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                res = self.detect_image(frame)
                res["input_filename"] = f"{Path(video_path).name}_frame{frame_idx}"
                # Drop heavy annotated image for batch; caller can re-generate
                res.pop("annotated_image", None)
                results_list.append(res)
            frame_idx += 1

        cap.release()
        return results_list

    # ── Batch ───────────────────────────────────────────────────
    def detect_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        """Run inference on a list of image paths. Returns list of result dicts."""
        results: list[dict] = []
        for p in image_paths:
            res = self.detect_image(p)
            results.append(res)
        return results
