import json
import os
import datetime
import config
from pylatex import Document, Section, Subsection, Table, Tabular, MultiColumn, Package
from pylatex.utils import italic, bold, NoEscape

def get_strategy_description(strategy):
    descriptions = {
        0: (
            "FLAT (Equal Weighting)",
            "In this strategy, all 5 matches in the rolling window are treated with equal importance. "
            "A match result from one month ago has the same mathematical impact as a match result from yesterday. "
            "This provides a stable, low-variance baseline but may miss sudden changes in team form."
        ),
        1: (
            "TEMPORAL (Recency Bias by Date)",
            "This strategy utilizes an Exponential Moving Average (EMA) based on the specific calendar dates of matches. "
            "Matches that occurred more recently in time are given exponentially more weight. "
            "This assumes that the passage of time is the primary factor in team decay and form evolution."
        ),
        2: (
            "ORDINAL (Recency Bias by Sequence)",
            "This strategy applies an EMA based on the sequence of matches (shifts), regardless of the actual date. "
            "The very last match played is given the highest weight, followed by the second to last, etc. "
            "This is ideal for sports where teams play a fixed number of games (like Progol) and 'rhythm' is more important than calendar time."
        )
    }
    return descriptions.get(strategy, ("Unknown", "No description available."))

def generate_methodology_report():
    metrics_path = 'models/metrics.json'
    if not os.path.exists(metrics_path):
        print("Error: metrics.json not found. Run training first.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # DYNAMIC FILENAME
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "LOCAL" if config.IS_LOCAL_TEST else "PROD"
    strat_name = ["FLAT", "TEMPORAL", "ORDINAL"][config.WEIGHT_STRATEGY]
    
    report_name = f"experiment_{mode_str}_{strat_name}_{timestamp}"
    output_path = os.path.join(config.REPORT_DIR, report_name)

    doc = Document(default_filepath=output_path)
    doc.packages.append(Package('geometry', options=['margin=1in']))
    doc.packages.append(Package('hyperref'))
    doc.packages.append(Package('booktabs'))

    # Title
    title = f"Progol Experiment: {mode_str} - {strat_name}"
    doc.preamble.append(NoEscape(r'\title{' + title + '}'))
    doc.preamble.append(NoEscape(r'\author{Autonomous Progol System}'))
    doc.preamble.append(NoEscape(r'\date{' + datetime.date.today().strftime("%B %d, %Y") + '}'))
    doc.append(NoEscape(r'\maketitle'))

    # 1. Experiment Overview
    with doc.create(Section('Experiment Overview')):
        doc.append(f"This report documents the results of the '{strat_name}' experiment run in '{mode_str}' mode. ")
        doc.append(f"The model was trained on a total of {int(metrics['classification_report']['macro avg']['support'])} matches.")

    # 2. Methodology Section (DYNAMIC)
    strategy_title, strategy_text = get_strategy_description(config.WEIGHT_STRATEGY)
    with doc.create(Section('Methodology: Weighting Strategy')):
        with doc.create(Subsection(strategy_title)):
            doc.append(strategy_text)

    # 3. Regularization & Hyperparameters
    with doc.create(Section('Hyperparameter Optimization')):
        doc.append("To prevent overfitting, we performed a Grid Search with Stratified K-Fold Cross-Validation. "
                   "The following optimal parameters were identified for this specific experiment:")
        with doc.create(Subsection('Optimal Parameters')):
            for k, v in metrics['best_params'].items():
                doc.append(italic(f"{k}: {v}"))
                doc.append(NoEscape(r"\\"))

    # 4. Performance Metrics
    with doc.create(Section('Performance Analysis')):
        doc.append(f"Accuracy: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
        
        with doc.create(Subsection('Classification Report')):
            with doc.create(Tabular('l|cccc')) as table:
                table.add_hline()
                table.add_row(("Class", "Precision", "Recall", "F1-Score", "Support"))
                table.add_hline()
                for label, stats in metrics['classification_report'].items():
                    if label in ['accuracy', 'macro avg', 'weighted avg']: continue
                    table.add_row((label, f"{stats['precision']:.2f}", f"{stats['recall']:.2f}", f"{stats['f1-score']:.2f}", int(stats['support'])))
                table.add_hline()

    # 5. Feature Importance
    with doc.create(Section('Feature Importance')):
        with doc.create(Tabular('l|r')) as table:
            table.add_hline()
            table.add_row(("Feature", "Importance Score"))
            table.add_hline()
            sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                table.add_row((feature.replace('_', ' '), f"{importance:.4f}"))
            table.add_hline()

    # Save LaTeX and attempt PDF
    try:
        doc.generate_tex()
        print(f"LaTeX source saved to {output_path}.tex")
        # PDF generation will still warn if pdflatex isn't installed, but the .tex is safe!
        doc.generate_pdf(clean_tex=True)
        print(f"Report generated successfully at {output_path}.pdf")
    except Exception as e:
        print(f"Warning: PDF generation failed, but {output_path}.tex is saved.")

if __name__ == "__main__":
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    generate_methodology_report()
