"""
Model evaluation script for the Parking Space Detection project.

This file is kept for REFERENCE ONLY.
It contains the evaluation metrics and visualization code extracted
from the original notebook (results_of_model.ipynb).

Known model comparison results:

    Model    | Precision | Recall | mAP@50 | mAP@50-95 | Speed (ms/img)
    ---------|-----------|--------|--------|-----------|---------------
    YOLO     |   0.997   | 0.992  | 0.995  |   0.924   |       4
    RT-DETR  |   0.996   | 0.997  | 0.995  |   0.939   |      37

The YOLO model was selected for production use (best1.pt) due to its
superior speed while maintaining comparable accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison():
    """Plot bar chart comparing YOLO and RT-DETR metrics."""
    models = ["YOLO", "RT-DETR"]
    precision = [0.997, 0.996]
    recall = [0.992, 0.997]
    map50 = [0.995, 0.995]
    map5095 = [0.924, 0.939]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 1.5 * width, precision, width, label="Precision", color="#2ecc71")
    ax.bar(x - 0.5 * width, recall, width, label="Recall", color="#3498db")
    ax.bar(x + 0.5 * width, map50, width, label="mAP@50", color="#e74c3c")
    ax.bar(x + 1.5 * width, map5095, width, label="mAP@50-95", color="#f39c12")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0.9, 1.0)

    plt.tight_layout()
    return fig


def plot_speed_comparison():
    """Plot speed comparison between models."""
    models = ["YOLO", "RT-DETR"]
    speed = [4, 37]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, speed, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Speed (ms/image)")
    ax.set_title("Inference Speed Comparison")

    for bar, val in zip(bars, speed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val} ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("PARKING SPACE DETECTION — EVALUATION SCRIPT (REFERENCE)")
    print("=" * 60)
    plot_model_comparison()
    plot_speed_comparison()
    plt.show()
