"""
Streamlit Dashboard for Smart Parking Space Detection.

Features:
    Tab 1 — Model Results Dashboard
        • Model performance metrics (precision, recall, mAP)
        • Speed comparison
        • Historical inference statistics from Snowflake
        • Recent inference run log

    Tab 2 — Run Inference
        • Upload single image → detect → display → store in Snowflake
        • Upload video → process frames → display → store in Snowflake
        • Batch processing → upload multiple images → process → store

Usage:
    streamlit run src/streamlit/app.py
"""

import sys
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import cv2

# ── Add project root to path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TRAINING_METRICS, MODEL_PATH, OUTPUT_DIR
from src.inference.detector import ParkingDetector
from src.inference.utils import save_annotated_image, image_to_bytes

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .metric-value { font-size: 36px; font-weight: bold; }
    .metric-label { font-size: 14px; color: #a0aec0; }
    .free-color { color: #48bb78; }
    .occupied-color { color: #fc8181; }
    .total-color { color: #63b3ed; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Model Loader ───────────────────────────────────────
@st.cache_resource
def load_detector():
    """Load YOLO model (cached so it's only loaded once)."""
    return ParkingDetector(model_path=MODEL_PATH)


def get_snowflake_client():
    """Get a SnowflakeClient instance. Returns None if connection fails."""
    try:
        from src.data.snowflake_client import SnowflakeClient
        client = SnowflakeClient()
        client.connect()
        return client
    except Exception as e:
        st.warning(f"⚠️ Snowflake connection failed: {e}")
        return None


# ── Helper Functions ──────────────────────────────────────────
def display_metrics_cards(free: int, occupied: int, total: int, processing_ms: float):
    """Display detection results as metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value free-color">{free}</div>
            <div class="metric-label">🟢 Free Slots</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value occupied-color">{occupied}</div>
            <div class="metric-label">🔴 Occupied Slots</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value total-color">{total}</div>
            <div class="metric-label">📊 Total Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ffd700;">{processing_ms:.0f}ms</div>
            <div class="metric-label">⚡ Processing Time</div>
        </div>
        """, unsafe_allow_html=True)


def store_result_in_snowflake(client, result: dict, input_type: str):
    """Store inference result in Snowflake."""
    if client is None:
        return None
    try:
        result["input_type"] = input_type
        run_id = client.insert_inference_result(result)
        client.insert_detections(run_id, result.get("detections", []))

        # Upload annotated image to stage
        annotated = result.get("annotated_image")
        if annotated is not None:
            filename = f"{run_id}.jpg"
            saved_path = save_annotated_image(annotated, filename)
            try:
                client.upload_to_stage(str(saved_path))
            except Exception:
                pass  # Stage upload is optional
        return run_id
    except Exception as e:
        st.error(f"Failed to store in Snowflake: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#                        SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/parking.png", width=80)
    st.title("🅿️ Smart Parking")
    st.markdown("---")
    st.markdown("**Model:** `best1.pt` (YOLOv8)")
    st.markdown("**Dataset:** PKLot PUCPR")
    st.markdown("**Classes:** Free, Occupied")
    st.markdown("---")
    st.markdown("##### Snowflake Status")
    sf_client = get_snowflake_client()
    if sf_client:
        st.success("✅ Connected to Snowflake")
    else:
        st.error("❌ Not connected")
        st.caption("Configure `.env` file to enable Snowflake integration")

# ══════════════════════════════════════════════════════════════
#                     MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.title("🅿️ Smart Parking Space Detection Dashboard")

tab1, tab2 = st.tabs(["📊 Model Results Dashboard", "🔍 Run Inference"])

# ──────────────────────────────────────────────────────────────
#   TAB 1: MODEL RESULTS DASHBOARD
# ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Model Performance Metrics")

    # ── Model Comparison from Training ────────────────────────
    st.subheader("📈 Model Comparison (Training Results)")

    models = list(TRAINING_METRICS.keys())
    precision = [TRAINING_METRICS[m]["precision"] for m in models]
    recall = [TRAINING_METRICS[m]["recall"] for m in models]
    map50 = [TRAINING_METRICS[m]["map50"] for m in models]
    map5095 = [TRAINING_METRICS[m]["map50_95"] for m in models]
    speed = [TRAINING_METRICS[m]["speed_ms"] for m in models]

    col1, col2 = st.columns(2)

    with col1:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(name="Precision", x=models, y=precision,
                                   marker_color="#2ecc71"))
        fig_perf.add_trace(go.Bar(name="Recall", x=models, y=recall,
                                   marker_color="#3498db"))
        fig_perf.add_trace(go.Bar(name="mAP@50", x=models, y=map50,
                                   marker_color="#e74c3c"))
        fig_perf.add_trace(go.Bar(name="mAP@50-95", x=models, y=map5095,
                                   marker_color="#f39c12"))
        fig_perf.update_layout(
            title="Detection Accuracy Metrics",
            barmode="group",
            yaxis=dict(range=[0.9, 1.0]),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col2:
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Bar(
            x=models, y=speed,
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{s} ms" for s in speed],
            textposition="outside",
        ))
        fig_speed.update_layout(
            title="Inference Speed Comparison",
            yaxis_title="ms / image",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    # ── Metrics Table ─────────────────────────────────────────
    st.subheader("📋 Detailed Metrics Table")
    metrics_df = pd.DataFrame({
        "Model": models,
        "Precision": precision,
        "Recall": recall,
        "mAP@50": map50,
        "mAP@50-95": map5095,
        "Speed (ms/img)": speed,
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Snowflake Historical Data ─────────────────────────────
    st.markdown("---")
    st.subheader("📜 Historical Inference Data (from Snowflake)")

    if sf_client:
        try:
            # Aggregated stats
            stats = sf_client.get_detection_stats()
            if stats and stats.get("TOTAL_RUNS", 0) > 0:
                st.markdown("##### Aggregated Statistics")
                scol1, scol2, scol3, scol4 = st.columns(4)
                scol1.metric("Total Runs", int(stats.get("TOTAL_RUNS", 0)))
                scol2.metric("Avg Free/Run", f"{stats.get('AVG_FREE', 0):.1f}")
                scol3.metric("Avg Occupied/Run", f"{stats.get('AVG_OCCUPIED', 0):.1f}")
                scol4.metric("Avg Processing", f"{stats.get('AVG_PROCESSING_MS', 0):.0f}ms")

            # Recent runs table
            history = sf_client.get_inference_history(limit=50)
            if history:
                st.markdown("##### Recent Inference Runs")
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

                # Time series chart
                if "TIMESTAMP" in hist_df.columns and len(hist_df) > 1:
                    hist_df["TIMESTAMP"] = pd.to_datetime(hist_df["TIMESTAMP"])
                    fig_ts = px.line(
                        hist_df, x="TIMESTAMP",
                        y=["FREE_SLOTS", "OCCUPIED_SLOTS"],
                        title="Parking Occupancy Over Time",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("No inference runs found in Snowflake yet. Run some inferences first!")
        except Exception as e:
            st.error(f"Error querying Snowflake: {e}")
    else:
        st.info("Connect to Snowflake to view historical data. Configure `.env` file.")

    # ── Snowflake Model Metrics ───────────────────────────────
    if sf_client:
        try:
            sf_metrics = sf_client.get_model_metrics()
            if sf_metrics:
                st.markdown("##### Model Metrics (Stored in Snowflake)")
                sf_df = pd.DataFrame(sf_metrics)
                st.dataframe(sf_df, use_container_width=True, hide_index=True)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
#   TAB 2: RUN INFERENCE
# ──────────────────────────────────────────────────────────────
with tab2:
    st.header("Run Parking Space Detection")

    # Sub-tabs for different input types
    inf_tab1, inf_tab2, inf_tab3 = st.tabs(["📷 Single Image", "🎥 Video", "📁 Batch Processing"])

    detector = load_detector()

    # ── Single Image ──────────────────────────────────────────
    with inf_tab1:
        st.subheader("Upload a Parking Lot Image")
        uploaded_image = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "bmp"],
            key="single_image",
        )

        if uploaded_image is not None:
            # Convert to numpy array
            file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if st.button("🔍 Detect Parking Spaces", key="detect_single"):
                with st.spinner("Running detection..."):
                    result = detector.detect_image(img)
                    result["input_filename"] = uploaded_image.name

                # Display results
                display_metrics_cards(
                    result["free"], result["occupied"],
                    result["total"], result["processing_ms"],
                )

                st.markdown("---")

                # Show annotated image
                annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

                # Legend
                st.markdown("🟢 **Free** &nbsp;&nbsp;&nbsp; 🔴 **Occupied**")

                # Store in Snowflake
                run_id = store_result_in_snowflake(sf_client, result, "image")
                if run_id:
                    st.success(f"✅ Results stored in Snowflake (Run ID: `{run_id}`)")

                # Download annotated image
                img_bytes = image_to_bytes(result["annotated_image"])
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name=f"detected_{uploaded_image.name}",
                    mime="image/jpeg",
                )

    # ── Video ─────────────────────────────────────────────────
    with inf_tab2:
        st.subheader("Upload a Parking Lot Video")
        uploaded_video = st.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv"],
            key="video_upload",
        )

        frame_interval = st.slider(
            "Process every N-th frame", min_value=1, max_value=120, value=30,
            help="Higher value = faster processing but fewer samples",
        )

        if uploaded_video is not None:
            if st.button("🔍 Process Video", key="detect_video"):
                # Save video to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_video.name).suffix
                ) as tmp:
                    tmp.write(uploaded_video.read())
                    tmp_path = tmp.name

                with st.spinner("Processing video frames..."):
                    results = detector.detect_video(tmp_path, sample_interval=frame_interval)

                st.success(f"Processed {len(results)} frames")

                if results:
                    # Summary
                    summary_df = pd.DataFrame([{
                        "Frame": r["input_filename"],
                        "Free": r["free"],
                        "Occupied": r["occupied"],
                        "Total": r["total"],
                        "Time (ms)": r["processing_ms"],
                    } for r in results])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    # Chart
                    fig = px.line(
                        summary_df, x="Frame",
                        y=["Free", "Occupied"],
                        title="Parking Occupancy Across Video Frames",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Store in Snowflake
                    if sf_client:
                        stored = 0
                        for r in results:
                            rid = store_result_in_snowflake(sf_client, r, "video")
                            if rid:
                                stored += 1
                        st.success(f"✅ {stored} frame results stored in Snowflake")

                # Cleanup
                os.unlink(tmp_path)

    # ── Batch Processing ──────────────────────────────────────
    with inf_tab3:
        st.subheader("Upload Multiple Images")
        uploaded_files = st.file_uploader(
            "Choose images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, key="batch_upload",
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images selected")

            if st.button("🔍 Process All Images", key="detect_batch"):
                all_results = []
                progress = st.progress(0)

                for i, uf in enumerate(uploaded_files):
                    file_bytes = np.frombuffer(uf.read(), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    result = detector.detect_image(img)
                    result["input_filename"] = uf.name
                    all_results.append(result)
                    progress.progress((i + 1) / len(uploaded_files))

                st.success(f"✅ Processed {len(all_results)} images")

                # Summary table
                summary_df = pd.DataFrame([{
                    "File": r["input_filename"],
                    "Free": r["free"],
                    "Occupied": r["occupied"],
                    "Total": r["total"],
                    "Time (ms)": r["processing_ms"],
                } for r in all_results])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Totals
                total_free = sum(r["free"] for r in all_results)
                total_occ = sum(r["occupied"] for r in all_results)
                display_metrics_cards(
                    total_free, total_occ, total_free + total_occ,
                    sum(r["processing_ms"] for r in all_results),
                )

                # Bar chart
                fig = px.bar(
                    summary_df, x="File", y=["Free", "Occupied"],
                    title="Batch Detection Results",
                    template="plotly_dark",
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Store in Snowflake
                if sf_client:
                    stored = 0
                    for r in all_results:
                        rid = store_result_in_snowflake(sf_client, r, "batch")
                        if rid:
                            stored += 1
                    st.success(f"✅ {stored} results stored in Snowflake")

                # Show first annotated image as preview
                if all_results:
                    first_rgb = cv2.cvtColor(all_results[0]["annotated_image"], cv2.COLOR_BGR2RGB)
                    st.image(first_rgb, caption=f"Preview: {all_results[0]['input_filename']}",
                             use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#718096;'>"
    "Smart Parking Detection | Powered by YOLOv8 + Snowflake + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
