import subprocess
import os
import sys

# Ensure config is imported correctly
sys.path.append(os.getcwd())
import config

def get_user_input():
    print("\n--- 🔧 PROGOL PIPELINE CONFIGURATION ---")
    
    # 1. Execution Mode
    print("Select Execution Mode:")
    print("  [1] LOCAL TEST (10% of data, faster)")
    print("  [2] PRODUCTION (100% of data, accurate)")
    mode_choice = input("Choice (1 or 2): ")
    is_local = (mode_choice == '1')
    
    # 2. Weighting Strategy
    print("\nSelect Feature Engineering Strategy:")
    print("  [0] FLAT (Matches count equally)")
    print("  [1] TEMPORAL (More weight by Match DATE)")
    print("  [2] ORDINAL (More weight by Match SEQUENCE)")
    print("  [3] CONTEXTUAL (Home/Away specific, Clean Sheets, H2H)")
    strategy_choice = input("Choice (0, 1, 2, or 3): ")
    strategy = int(strategy_choice)
    
    # Update config and set environment variables for sub-processes
    os.environ['IS_LOCAL_TEST'] = "True" if is_local else "False"
    os.environ['WEIGHT_STRATEGY'] = str(strategy)
    
    mode_str = "LOCAL TEST (10%)" if is_local else "PRODUCTION (100%)"
    strategy_names = {0: "FLAT", 1: "TEMPORAL", 2: "ORDINAL", 3: "CONTEXTUAL"}
    print(f"\n🚀 STARTING {mode_str} with {strategy_names[strategy]} strategy...")

def run_step(script_name, description):
    print(f"\n--- STEP: {description} ---")
    
    # Use sys.executable to ensure we use the SAME python (the venv one)
    # for all sub-processes. This prevents ModuleNotFoundErrors.
    python_exe = sys.executable
    
    try:
        result = subprocess.run(
            [python_exe, script_name], 
            check=True, 
            env=os.environ.copy()
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        return False

def main():
    if not os.path.exists(".env"):
        print("Error: .env file missing.")
        return

    get_user_input()
    
    # Execute each script using the venv python
    if not run_step("fetch_data.py", "Fetching Data"): return
    if not run_step("preprocess.py", "Preprocessing Data"): return
    if not run_step("train_model.py", "Hyperparameter Tuning & Training"): return
    if not run_step("generate_report.py", "Generating Methodology Report"): pass
    if not run_step("get_progol_ids.py", "Web Scraping Progol Slate"): return
    if not run_step("predict_progol.py", "Final Progol Predictions"): return

    print("\n✅ PIPELINE COMPLETE! ✅")

if __name__ == "__main__":
    main()
