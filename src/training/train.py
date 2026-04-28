"""
Training script for the Parking Space Detection model.

This file is kept for REFERENCE ONLY — the model has already been trained
and the weights are saved as models/best1.pt.

The training was performed on the PKLot (PUCPR) dataset using YOLOv8.
Dataset structure:
    PKLot_PUCPR/
    └── PUCPR/
        ├── Cloudy/
        ├── Rainy/
        └── Sunny/

Each weather folder contains date-subfolders with .jpg images and
matching .xml annotation files.

Steps performed in the original notebook (Parking_Space_21.ipynb):
    1. Mount Google Drive & load PKLot_PUCPR dataset
    2. Parse XML annotations → convert to YOLO format (x_center, y_center, w, h)
    3. Split into train / val / test sets
    4. Train YOLOv8 model:
        model = YOLO("yolov8n.pt")
        model.train(
            data="parking_data.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            project="parking_runs",
            name="yolo_parking",
        )
    5. Export best weights → best1.pt
    6. Validate on test set

Total images: ~4,474
Classes: 2 (empty / occupied)
Final metrics:
    Precision : 0.997
    Recall    : 0.992
    mAP@0.5   : 0.995
    mAP@0.5:0.95 : 0.924

To retrain (if needed):
    1. Prepare dataset in YOLO format
    2. Create a parking_data.yaml file
    3. Run: python src/training/train.py
"""

from ultralytics import YOLO


def train_model(
    base_model: str = "yolov8n.pt",
    data_yaml: str = "parking_data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "parking_runs",
    name: str = "yolo_parking",
):
    """
    Train a YOLO model on a parking dataset.

    This function is provided for reference. The model has already been
    trained and the weights are stored in models/best1.pt.
    """
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
    )
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PARKING SPACE DETECTION — TRAINING SCRIPT (REFERENCE)")
    print("=" * 60)
    print()
    print("The model has already been trained. Weights: models/best1.pt")
    print("To retrain, update the data_yaml path and run this script.")
    print()
    # Uncomment below to actually train:
    # train_model()
