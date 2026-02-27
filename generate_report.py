import json
import os
from pylatex import Document, Section, Subsection, Table, Tabular, MultiColumn, Package
from pylatex.utils import italic, bold, NoEscape
import datetime

def generate_methodology_report(metrics_path='models/metrics.json', output_path='reports/methodology_report'):
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run training first.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create document
    doc = Document(default_filepath=output_path)
    doc.packages.append(Package('geometry', options=['margin=1in']))
    doc.packages.append(Package('hyperref'))
    doc.packages.append(Package('booktabs'))

    # Title
    doc.preamble.append(NoEscape(r'\title{Progol Prediction Engine: Methodology and Performance Report}'))
    doc.preamble.append(NoEscape(r'\author{Autonomous Progol System}'))
    doc.preamble.append(NoEscape(r'\date{' + datetime.date.today().strftime("%B %d, %Y") + '}'))
    doc.append(NoEscape(r'\maketitle'))

    # 1. Introduction
    with doc.create(Section('Introduction')):
        doc.append("This report outlines the methodology used to develop the Progol Prediction Engine. "
                   "The system is designed to predict football match outcomes (Home, Draw, Away) "
                   "using a Multi-Class XGBoost model. Our primary goal is to ensure high predictive power "
                   "while rigorously preventing overfitting.")

    # 2. Preventing Overfitting
    with doc.create(Section('Methodology: Avoiding Overfitting')):
        doc.append("To guarantee that the model generalizes well to unseen data, the following strategies were implemented:")
        with doc.create(Subsection('Stratified Train-Test Split')):
            doc.append("The dataset was split into training (80%) and testing (20%) sets using stratification. "
                       "This ensures that the proportion of Home, Draw, and Away results remains consistent across both sets, "
                       "preventing the model from learning biases from imbalanced subsets.")
        
        with doc.create(Subsection('K-Fold Cross-Validation')):
            doc.append("During hyperparameter tuning, we utilized 5-Fold Stratified Cross-Validation. "
                       "This means the training data was split 5 times, and the model was validated on different folds "
                       "in each iteration. This reduces the variance of our performance estimates.")

        with doc.create(Subsection('Regularization Techniques')):
            doc.append("XGBoost's built-in regularization parameters were used:")
            doc.append(NoEscape(r"\begin{itemize}"))
            doc.append(NoEscape(r"\item \textbf{Gamma}: Minimum loss reduction required to make a further partition."))
            doc.append(NoEscape(r"\item \textbf{Subsample}: Randomly sampling training data to prevent the trees from becoming too complex."))
            doc.append(NoEscape(r"\item \textbf{Colsample\_bytree}: Subsampling features for each tree construction."))
            doc.append(NoEscape(r"\end{itemize}"))

    # 3. Hyperparameter Adjustment
    with doc.create(Section('Hyperparameter Optimization')):
        doc.append("We performed a Grid Search over the following parameter space:")
        with doc.create(Tabular('l|l')) as table:
            table.add_hline()
            table.add_row(("Hyperparameter", "Values Tested"))
            table.add_hline()
            table.add_row(("max\_depth", "[3, 5, 7]"))
            table.add_row(("learning\_rate", "[0.01, 0.05, 0.1]"))
            table.add_row(("subsample", "[0.7, 0.8, 0.9]"))
            table.add_row(("colsample\_bytree", "[0.7, 0.8, 0.9]"))
            table.add_hline()

        doc.append("\n\nThe best parameters identified through Grid Search were:")
        with doc.create(Subsection('Optimal Parameters')):
            for k, v in metrics['best_params'].items():
                doc.append(italic(f"{k}: {v}"))
                doc.append(NoEscape(r"\\"))

    # 4. Performance Metrics
    with doc.create(Section('Performance Analysis')):
        doc.append(f"The model achieved an overall Accuracy of {metrics['accuracy']:.4f} and "
                   f"a Macro F1-Score of {metrics['f1_macro']:.4f}.")
        
        with doc.create(Subsection('Classification Report')):
            with doc.create(Tabular('l|cccc')) as table:
                table.add_hline()
                table.add_row(("Class", "Precision", "Recall", "F1-Score", "Support"))
                table.add_hline()
                for label, stats in metrics['classification_report'].items():
                    if label in ['accuracy', 'macro avg', 'weighted avg']: continue
                    table.add_row((label, f"{stats['precision']:.2f}", f"{stats['recall']:.2f}", f"{stats['f1-score']:.2f}", int(stats['support'])))
                table.add_hline()

        with doc.create(Subsection('Feature Importance')):
            doc.append("The relative importance of features in determining the outcome is as follows:")
            with doc.create(Tabular('l|r')) as table:
                table.add_hline()
                table.add_row(("Feature", "Importance Score"))
                table.add_hline()
                sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    table.add_row((feature.replace('_', ' '), f"{importance:.4f}"))
                table.add_hline()

    # 5. Conclusion
    with doc.create(Section('Conclusion')):
        doc.append("The methodology ensures a robust model by balancing complexity and generalization. "
                   "The use of stratified splits, weighted sampling, and cross-validated grid search provides "
                   "a reliable foundation for predicting the Progol slate while avoiding overly optimistic conclusions.")

    try:
        # Always save the .tex file first
        doc.generate_tex()
        print(f"LaTeX source saved to {output_path}.tex")
        
        # Try to generate PDF
        doc.generate_pdf(clean_tex=True)
        print(f"Report generated successfully at {output_path}.pdf")
    except Exception as e:
        print(f"Warning: PDF generation failed because {e}")
        print(f"You can upload {output_path}.tex to Overleaf.com to get your PDF.")

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    generate_methodology_report()
