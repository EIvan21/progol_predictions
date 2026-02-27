import os

# Read from environment variables (set by run_pipeline.py)
IS_LOCAL_TEST = os.getenv('IS_LOCAL_TEST', 'False').lower() == 'true'
WEIGHT_STRATEGY = int(os.getenv('WEIGHT_STRATEGY', 3)) # Default to Contextual
MODEL_TYPE = os.getenv('MODEL_TYPE', 'Ensemble')

# Directories
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
MODEL_DIR = "models/"
REPORT_DIR = "reports/"

def get_data_limit(total_count):
    if IS_LOCAL_TEST:
        return max(100, int(total_count * 0.1)) # Minimum 100 matches for local
    return total_count
