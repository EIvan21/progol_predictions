import os

# --- DYNAMIC GLOBAL STATE ---
# We read from environment variables set by run_pipeline.py
IS_LOCAL_TEST = os.getenv('IS_LOCAL_TEST', 'False').lower() == 'true'
WEIGHT_STRATEGY = int(os.getenv('WEIGHT_STRATEGY', 0))

# --- DIRECTORIES ---
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
MODEL_DIR = "models/"
REPORT_DIR = "reports/"

def get_data_limit(total_count):
    if IS_LOCAL_TEST:
        # 10% for local testing
        return int(total_count * 0.1)
    return total_count
