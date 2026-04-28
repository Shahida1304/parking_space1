"""
CLI script to run parking space detection inference.

Usage:
    python scripts/run_inference.py --image path/to/image.jpg
    python scripts/run_inference.py --video path/to/video.mp4
    python scripts/run_inference.py --batch path/to/folder/

Results are printed to stdout and stored in Snowflake.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.detector import ParkingDetector
from src.inference.utils import save_annotated_image
from src.data.snowflake_client import SnowflakeClient


def store_result(client: SnowflakeClient, result: dict, input_type: str):
    """Store inference result in Snowflake."""
    result["input_type"] = input_type
    run_id = client.insert_inference_result(result)
    client.insert_detections(run_id, result.get("detections", []))

    # Save & upload annotated image if available
    annotated = result.get("annotated_image")
    if annotated is not None:
        filename = f"{run_id}.jpg"
        saved_path = save_annotated_image(annotated, filename)
        try:
            client.upload_to_stage(str(saved_path))
        except Exception as e:
            print(f"  [WARN] Could not upload to stage: {e}")

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Parking Space Detection — Inference CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--video", type=str, help="Path to a video file")
    group.add_argument("--batch", type=str, help="Path to a folder of images")
    parser.add_argument("--no-snowflake", action="store_true", help="Skip Snowflake storage")

    args = parser.parse_args()

    detector = ParkingDetector()
    client = None

    if not args.no_snowflake:
        try:
            client = SnowflakeClient()
            client.connect()
        except Exception as e:
            print(f"[WARN] Could not connect to Snowflake: {e}")
            print("       Results will NOT be stored. Use --no-snowflake to suppress.")
            client = None

    # ── Image ───────────────────────────────────────────────
    if args.image:
        print(f"[INFO] Processing image: {args.image}")
        result = detector.detect_image(args.image)
        print(f"  Free={result['free']}  Occupied={result['occupied']}  "
              f"Total={result['total']}  Time={result['processing_ms']}ms")

        if client:
            run_id = store_result(client, result, "image")
            print(f"  → Stored in Snowflake (run_id={run_id})")
        else:
            # Save locally even without Snowflake
            filename = f"{result['run_id']}.jpg"
            path = save_annotated_image(result["annotated_image"], filename)
            print(f"  → Saved annotated image: {path}")

    # ── Video ───────────────────────────────────────────────
    elif args.video:
        print(f"[INFO] Processing video: {args.video}")
        results = detector.detect_video(args.video, sample_interval=30)
        print(f"  Processed {len(results)} frames")

        for i, result in enumerate(results):
            print(f"  Frame {i}: Free={result['free']}  Occupied={result['occupied']}")
            if client:
                store_result(client, result, "video")

    # ── Batch ───────────────────────────────────────────────
    elif args.batch:
        folder = Path(args.batch)
        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        print(f"[INFO] Processing batch: {len(image_files)} images in {folder}")

        results = detector.detect_batch([str(p) for p in image_files])
        for result in results:
            print(f"  {result['input_filename']}: Free={result['free']}  "
                  f"Occupied={result['occupied']}  Total={result['total']}")
            if client:
                store_result(client, result, "batch")

    if client:
        client.close()

    print("[DONE]")


if __name__ == "__main__":
    main()
