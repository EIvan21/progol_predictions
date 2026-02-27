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
    print("  [1] LOCAL TEST (10% of data)")
    print("  [2] PRODUCTION (100% of data)")
    mode_choice = input("Choice (1 or 2): ")
    is_local = (mode_choice == '1')
    
    # 2. Model Architecture
    print("\nSelect Model Architecture:")
    print("  [1] XGBoost (Fast & High Performance)")
    print("  [2] Random Forest (Stable & Robust)")
    print("  [3] CatBoost (Best for Categorical Data)")
    print("  [4] Neural Network (Deep Learning - Complex Interactions)")
    model_choice = input("Choice (1, 2, 3, or 4): ")
    model_map = {'1': 'XGBoost', '2': 'RandomForest', '3': 'CatBoost', '4': 'NeuralNetwork'}
    model_type = model_map.get(model_choice, 'XGBoost')

    # 3. Feature Engineering Strategy
    print("\nSelect Feature Engineering Strategy:")
    print("  [0] FLAT (Equal Weighting)")
    print("  [1] TEMPORAL (By Date)")
    print("  [2] ORDINAL (By Sequence)")
    print("  [3] CONTEXTUAL (Home/Away Splits, Clean Sheets)")
    strategy_choice = input("Choice (0, 1, 2, or 3): ")
    strategy = int(strategy_choice)
    
    # Set environment variables for sub-processes
    os.environ['IS_LOCAL_TEST'] = "True" if is_local else "False"
    os.environ['WEIGHT_STRATEGY'] = str(strategy)
    os.environ['MODEL_TYPE'] = model_type
    
    mode_str = "LOCAL TEST (10%)" if is_local else "PRODUCTION (100%)"
    print(f"\n🚀 STARTING {mode_str} using {model_type} with strategy {strategy}...")

def run_step(script_name, description):
    print(f"\n--- STEP: {description} ---")
    python_exe = sys.executable
    try:
        subprocess.run([python_exe, script_name], check=True, env=os.environ.copy())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        return False

def main():
    if not os.path.exists(".env"):
        print("Error: .env file missing.")
        return

    get_user_input()
    
    if not run_step("fetch_data.py", "Fetching Data"): return
    if not run_step("preprocess.py", "Preprocessing Data"): return
    if not run_step("train_model.py", "Training Model"): return
    if not run_step("generate_report.py", "Generating Report"): pass
    if not run_step("get_progol_ids.py", "Web Scraping Progol Slate"): return
    if not run_step("predict_progol.py", "Final Progol Predictions"): return

    print("\n✅ PIPELINE COMPLETE! ✅")

if __name__ == "__main__":
    main()
