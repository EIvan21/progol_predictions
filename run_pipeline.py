import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_user_input():
    print("\n--- PROGOL PIPELINE CONFIGURATION ---")

    print("Select Execution Mode:")
    print("  [1] LOCAL TEST (10% of data)")
    print("  [2] PRODUCTION (100% of data)")
    is_local = input("Choice (1 or 2): ") == '1'

    print("\nSelect Model Architecture:")
    print("  [1] XGBoost  [2] RandomForest  [3] CatBoost  [4] NeuralNet  [5] Ensemble")
    model_map = {'1': 'XGBoost', '2': 'RandomForest', '3': 'CatBoost', '4': 'NeuralNetwork', '5': 'Ensemble'}
    model_type = model_map.get(input("Choice (1-5): "), 'Ensemble')

    print("\nRun Optuna tuning before training? (extends runtime)")
    do_tune = input("(y/N): ").strip().lower() == 'y'

    print("\nRun walk-forward validation after training? (sanity-checks robustness)")
    do_wf = input("(Y/n): ").strip().lower() != 'n'

    print("\nRun economic backtest after training? (Kelly + ROI)")
    do_bt = input("(Y/n): ").strip().lower() != 'n'

    os.environ['IS_LOCAL_TEST'] = "True" if is_local else "False"
    os.environ['MODEL_TYPE'] = model_type

    return {'tune': do_tune, 'walk_forward': do_wf, 'backtest': do_bt}


def run_step(module: str, description: str, args: list = None) -> bool:
    print(f"\n--- {description} ---")
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        return False


def main():
    if not os.path.exists(".env"):
        print("Error: .env file missing. Copy .env.example to .env and add your API key.")
        return

    flags = get_user_input()

    steps = [
        ("src.progol.ingest.fetch_data", "Fetching data", None, True),
        ("src.progol.modeling.preprocess", "Preprocessing", None, True),
    ]
    if flags['tune']:
        steps.append(("src.progol.modeling.tune", "Optuna tuning", ["--trials", "30", "--timeout", "1200"], False))
    steps.append(("src.progol.modeling.train", "Training model", None, True))
    if flags['walk_forward']:
        steps.append(("src.progol.modeling.walk_forward", "Walk-forward validation", ["--folds", "6"], False))
    if flags['backtest']:
        steps.append(("src.progol.modeling.backtest", "Economic backtest", ["--kelly", "0.25", "--min-edge", "0.04"], False))
    steps += [
        ("src.progol.reporting.eda", "Generating EDA report", None, False),
        ("src.progol.reporting.generate_report", "Generating training summary", None, False),
        ("src.progol.ingest.get_progol_ids", "Scraping Progol slate", None, True),
        ("src.progol.modeling.predict", "Final predictions", None, True),
        ("src.progol.reporting.score_report", "Generating exact-score maps", None, False),
    ]

    for module, desc, args, required in steps:
        ok = run_step(module, desc, args)
        if not ok and required:
            return

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
