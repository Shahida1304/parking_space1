"""
SnowflakeClient — Manages all Snowflake interactions for the Parking Detection project.

Provides methods to:
    - Create database, schema, tables, and stages
    - Insert inference results and individual detections
    - Insert / query model metrics
    - Upload annotated images to Snowflake internal stage
    - Query historical inference data for the dashboard
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import snowflake.connector

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_ROLE,
)


class SnowflakeClient:
    """Client for reading/writing parking detection data in Snowflake."""

    def __init__(self):
        self.conn = None

    # ── Connection ──────────────────────────────────────────────
    def connect(self):
        """Establish a connection to Snowflake."""
        if self.conn is not None:
            return self.conn

        self.conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
        )
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _cursor(self):
        conn = self.connect()
        return conn.cursor()

    def _execute(self, sql: str, params: tuple | None = None):
        cur = self._cursor()
        try:
            cur.execute(sql, params)
            return cur
        except Exception:
            cur.close()
            raise

    # ── Setup ───────────────────────────────────────────────────
    def setup_database(self):
        """Create the database, schema, tables, and stages if they don't exist."""
        cur = self._cursor()
        try:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE}")
            cur.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
            self._create_tables(cur)
            self._create_stages(cur)
            print("[INFO] Snowflake setup complete.")
        finally:
            cur.close()

    def _create_tables(self, cur):
        """Create all required tables."""
        # Inference Runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS INFERENCE_RUNS (
                RUN_ID        VARCHAR(36) PRIMARY KEY,
                TIMESTAMP     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                INPUT_TYPE    VARCHAR(10),
                INPUT_FILENAME VARCHAR(500),
                TOTAL_SLOTS   INTEGER,
                FREE_SLOTS    INTEGER,
                OCCUPIED_SLOTS INTEGER,
                MODEL_NAME    VARCHAR(50) DEFAULT 'best1.pt',
                PROCESSING_TIME_MS FLOAT,
                CREATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)

        # Individual Detections
        cur.execute("""
            CREATE TABLE IF NOT EXISTS DETECTIONS (
                DETECTION_ID  VARCHAR(36) PRIMARY KEY,
                RUN_ID        VARCHAR(36),
                X1 FLOAT, Y1 FLOAT, X2 FLOAT, Y2 FLOAT,
                CLASS_NAME    VARCHAR(20),
                CONFIDENCE    FLOAT,
                CREATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)

        # Model Metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS MODEL_METRICS (
                METRIC_ID       VARCHAR(36) PRIMARY KEY,
                MODEL_NAME      VARCHAR(50),
                PRECISION_SCORE FLOAT,
                RECALL_SCORE    FLOAT,
                MAP50           FLOAT,
                MAP50_95        FLOAT,
                SPEED_MS        FLOAT,
                EVALUATION_DATE TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)

        print("[INFO] Tables created / verified.")

    def _create_stages(self, cur):
        """Create internal stage for annotated image storage."""
        cur.execute("CREATE STAGE IF NOT EXISTS PARKING_STAGE")
        print("[INFO] Stage PARKING_STAGE created / verified.")

    # ── Use Database ────────────────────────────────────────────
    def _use_db(self, cur):
        cur.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        cur.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")

    # ── Insert Operations ───────────────────────────────────────
    def insert_inference_result(self, result: dict[str, Any]) -> str:
        """
        Insert an inference run record.

        Parameters
        ----------
        result : dict with keys run_id, free, occupied, total, processing_ms, input_filename

        Returns the run_id.
        """
        cur = self._cursor()
        try:
            self._use_db(cur)
            run_id = result.get("run_id", str(uuid.uuid4()))
            cur.execute(
                """
                INSERT INTO INFERENCE_RUNS
                    (RUN_ID, INPUT_TYPE, INPUT_FILENAME, TOTAL_SLOTS,
                     FREE_SLOTS, OCCUPIED_SLOTS, MODEL_NAME, PROCESSING_TIME_MS)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    result.get("input_type", "image"),
                    result.get("input_filename", ""),
                    result.get("total", 0),
                    result.get("free", 0),
                    result.get("occupied", 0),
                    result.get("model_name", "best1.pt"),
                    result.get("processing_ms", 0.0),
                ),
            )
            return run_id
        finally:
            cur.close()

    def insert_detections(self, run_id: str, detections: list[dict]):
        """Insert individual detection records for a given run."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            for det in detections:
                cur.execute(
                    """
                    INSERT INTO DETECTIONS
                        (DETECTION_ID, RUN_ID, X1, Y1, X2, Y2, CLASS_NAME, CONFIDENCE)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        run_id,
                        det["x1"],
                        det["y1"],
                        det["x2"],
                        det["y2"],
                        det["class_name"],
                        det["confidence"],
                    ),
                )
        finally:
            cur.close()

    def insert_model_metrics(self, model_name: str, metrics: dict):
        """Insert model training/evaluation metrics."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            cur.execute(
                """
                INSERT INTO MODEL_METRICS
                    (METRIC_ID, MODEL_NAME, PRECISION_SCORE, RECALL_SCORE,
                     MAP50, MAP50_95, SPEED_MS)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    model_name,
                    metrics.get("precision", 0.0),
                    metrics.get("recall", 0.0),
                    metrics.get("map50", 0.0),
                    metrics.get("map50_95", 0.0),
                    metrics.get("speed_ms", 0.0),
                ),
            )
        finally:
            cur.close()

    def upload_to_stage(self, local_path: str | Path, stage_name: str = "PARKING_STAGE"):
        """Upload a file to a Snowflake internal stage."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            cur.execute(f"PUT file://{local_path} @{stage_name} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
        finally:
            cur.close()

    # ── Query Operations ────────────────────────────────────────
    def get_inference_history(self, limit: int = 100) -> list[dict]:
        """Retrieve recent inference runs."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            cur.execute(
                """
                SELECT RUN_ID, TIMESTAMP, INPUT_TYPE, INPUT_FILENAME,
                       TOTAL_SLOTS, FREE_SLOTS, OCCUPIED_SLOTS,
                       MODEL_NAME, PROCESSING_TIME_MS
                FROM INFERENCE_RUNS
                ORDER BY TIMESTAMP DESC
                LIMIT %s
                """,
                (limit,),
            )
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cur.close()

    def get_model_metrics(self) -> list[dict]:
        """Retrieve all model metrics."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            cur.execute(
                """
                SELECT METRIC_ID, MODEL_NAME, PRECISION_SCORE, RECALL_SCORE,
                       MAP50, MAP50_95, SPEED_MS, EVALUATION_DATE
                FROM MODEL_METRICS
                ORDER BY EVALUATION_DATE DESC
                """
            )
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cur.close()

    def get_detection_stats(self) -> dict:
        """Get aggregated detection statistics."""
        cur = self._cursor()
        try:
            self._use_db(cur)
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_runs,
                    SUM(TOTAL_SLOTS) AS total_slots_detected,
                    SUM(FREE_SLOTS) AS total_free,
                    SUM(OCCUPIED_SLOTS) AS total_occupied,
                    AVG(FREE_SLOTS) AS avg_free,
                    AVG(OCCUPIED_SLOTS) AS avg_occupied,
                    AVG(PROCESSING_TIME_MS) AS avg_processing_ms
                FROM INFERENCE_RUNS
                """
            )
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row:
                return dict(zip(columns, row))
            return {}
        finally:
            cur.close()
