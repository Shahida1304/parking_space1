"""
Snowflake setup script.

Creates the PARKING_DETECTION database, schema, tables, stages,
and seeds the MODEL_METRICS table with known training results.

Usage:
    python scripts/setup_snowflake.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import TRAINING_METRICS
from src.data.snowflake_client import SnowflakeClient


def main():
    print("=" * 60)
    print(" Parking Detection — Snowflake Setup")
    print("=" * 60)
    print()

    client = SnowflakeClient()

    # 1. Create database, schema, tables, stages
    print("[STEP 1] Creating database, schema, tables, and stages ...")
    client.setup_database()
    print()

    # 2. Seed MODEL_METRICS with known training results
    print("[STEP 2] Seeding MODEL_METRICS table with training results ...")
    for model_name, metrics in TRAINING_METRICS.items():
        print(f"  -> Inserting metrics for {model_name}")
        client.insert_model_metrics(model_name, metrics)
    print()

    # 3. Verify
    print("[STEP 3] Verifying setup ...")
    metrics = client.get_model_metrics()
    print(f"  -> Found {len(metrics)} model metric records:")
    for m in metrics:
        print(f"     {m['MODEL_NAME']}: precision={m['PRECISION_SCORE']}, "
              f"recall={m['RECALL_SCORE']}, mAP50={m['MAP50']}")
    print()

    client.close()
    print("[DONE] Snowflake setup complete!")
    print()
    print("Next steps:")
    print("  1. Run the Streamlit app:  streamlit run src/streamlit/app.py")
    print("  2. Or run inference CLI:   python scripts/run_inference.py --image <path>")


if __name__ == "__main__":
    main()
