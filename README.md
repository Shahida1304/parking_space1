# 🅿️ Smart Parking Space Detection

A production-ready parking space detection system using **YOLOv8**, **Snowflake**, and **Streamlit**.

Detects free and occupied parking spaces in images/videos using a pre-trained YOLO model (`best1.pt`), stores results in **Snowflake** (tables, stages), and provides an interactive **Streamlit** dashboard for visualization and real-time inference.

---

## 📁 Project Structure

```
parking_space/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Snowflake credentials template
├── config/
│   └── settings.py                 # Configuration loader
├── models/
│   └── best1.pt                    # Pre-trained YOLO model (~188MB)
├── src/
│   ├── inference/
│   │   ├── detector.py             # ParkingDetector class (YOLO inference)
│   │   └── utils.py                # Image/video utilities
│   ├── training/
│   │   ├── train.py                # Training script (reference only)
│   │   └── evaluate.py             # Evaluation script (reference only)
│   ├── data/
│   │   └── snowflake_client.py     # Snowflake connector & queries
│   └── streamlit/
│       └── app.py                  # Streamlit dashboard
├── scripts/
│   ├── setup_snowflake.py          # Initialize Snowflake DB/tables/stages
│   └── run_inference.py            # CLI inference tool
├── notebooks/
│   ├── Parking_Space_21.ipynb      # Original training notebook
│   └── results_of_model.ipynb      # Model results notebook
├── data/
│   ├── sample_images/              # Sample parking lot images
│   └── slots.json                  # Parking slot configuration
├── outputs/                        # Annotated inference outputs
└── legacy/                         # Original Flask app (reference)
    ├── app.py
    ├── templates/index.html
    └── static/style.css
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Snowflake

Copy the environment template and fill in your Snowflake credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Snowflake account details:
```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=PARKING_DETECTION
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_ROLE=SYSADMIN
```

### 3. Set Up Snowflake Database

```bash
python scripts/setup_snowflake.py
```

This creates:
- **Database:** `PARKING_DETECTION`
- **Tables:** `INFERENCE_RUNS`, `DETECTIONS`, `MODEL_METRICS`
- **Stage:** `@PARKING_STAGE` (for annotated image storage)
- Seeds the `MODEL_METRICS` table with training results

### 4. Launch the Streamlit Dashboard

```bash
streamlit run src/streamlit/app.py
```

### 5. Or Run Inference via CLI

```bash
# Single image
python scripts/run_inference.py --image data/sample_images/example.jpg

# Video
python scripts/run_inference.py --video path/to/video.mp4

# Batch (all images in a folder)
python scripts/run_inference.py --batch data/sample_images/

# Without Snowflake
python scripts/run_inference.py --image data/sample_images/example.jpg --no-snowflake
```

---

## 📊 Dashboard Features

### Tab 1: Model Results Dashboard
- **Model Comparison Charts** — Precision, Recall, mAP for YOLO vs RT-DETR
- **Speed Comparison** — Inference time per image
- **Historical Data** — Time series of past inference results from Snowflake
- **Recent Runs Table** — Log of all inference runs stored in Snowflake

### Tab 2: Run Inference
- **Single Image Upload** — Upload → Detect → View annotated result → Store in Snowflake
- **Video Upload** — Upload → Process frames → View occupancy chart → Store in Snowflake
- **Batch Processing** — Upload multiple images → Process all → View summary → Store in Snowflake

---

## ❄️ Snowflake Schema

| Table | Description |
|-------|-------------|
| `INFERENCE_RUNS` | Each inference run (timestamp, counts, processing time) |
| `DETECTIONS` | Individual bounding boxes per run |
| `MODEL_METRICS` | Training metrics (precision, recall, mAP, speed) |
| `@PARKING_STAGE` | Internal stage for annotated images |

---

## 🧠 Model Details

| Metric | YOLO (best1.pt) | RT-DETR |
|--------|-----------------|---------|
| Precision | 0.997 | 0.996 |
| Recall | 0.992 | 0.997 |
| mAP@50 | 0.995 | 0.995 |
| mAP@50-95 | 0.924 | 0.939 |
| Speed (ms/img) | **4** | 37 |

The YOLO model was selected for production use due to its **9x faster** inference speed while maintaining comparable accuracy.

---

## 🛠️ Tech Stack

- **ML Framework:** Ultralytics YOLOv8
- **Data Warehouse:** Snowflake
- **Dashboard:** Streamlit + Plotly
- **Image Processing:** OpenCV + Pillow
- **Language:** Python 3.10+
