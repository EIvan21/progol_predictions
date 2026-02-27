import subprocess
import os
import sys

def run_step(command, description):
    print(f"\n--- STEP: {description} ---")
    try:
        # Run with current python environment
        result = subprocess.run([sys.executable] + command.split(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        return False

def main():
    print("🚀 STARTING PROGOL AUTONOMOUS PIPELINE 🚀")
    
    if not os.path.exists(".env"):
        print("Error: .env file missing. Run setup.sh first.")
        return

    # 1. Fetch Massive Data
    if not run_step("fetch_data.py", "Fetching Massive Dataset (22 Leagues, 6 Years)"):
        return
        
    # 2. Preprocessing
    if not run_step("preprocess.py", "Processing Match Data & Rolling Averages"):
        return
        
    # 3. Training & Tuning (Tuning added for accuracy)
    if not run_step("train_model.py", "Hyperparameter Tuning & Training Multi-Class XGBoost"):
        return
        
    # 4. Generate LaTeX Methodology Report
    if not run_step("generate_report.py", "Generating LaTeX Methodology PDF Report"):
        print("Note: PDF generation skipped or failed, but pipeline continues.")
        
    # 5. Scrape & Resolve Match IDs
    if not run_step("get_progol_ids.py", "Automated Web Scraping for current Progol slate"):
        return
        
    # 6. Final Prediction
    if not run_step("predict_progol.py", "Generating Friday's Progol Predictions"):
        print("\nPipeline finished, but check the logs for errors.")
        return

    print("\n✅ PIPELINE COMPLETE! ✅")
    print("Check 'reports/methodology_report.pdf' for details on model performance.")

if __name__ == "__main__":
    main()
