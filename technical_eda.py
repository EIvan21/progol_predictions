import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Configuration
DATA_PATH = 'data/processed/final_train_data.csv'
REPORT_DIR = 'reports/technical_eda/'
os.makedirs(REPORT_DIR, exist_ok=True)

def run_technical_eda():
    print("🔬 INITIALIZING RIGOROUS FEATURE AUDIT...")
    df = pd.read_csv(DATA_PATH)
    
    # Define current IPI features
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features].fillna(0)
    y = df['target']

    # 1. Multicollinearity Analysis (Correlation Heatmap)
    print("📊 Generating Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, annot=False, cmap='RdBu', center=0)
    plt.title("IPI Feature Correlation Matrix")
    plt.savefig(f"{REPORT_DIR}correlation_heatmap.png")
    plt.close()

    # 2. VIF (Variance Inflation Factor)
    print("🧮 Calculating VIF Scores...")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    vif_data.to_csv(f"{REPORT_DIR}vif_analysis.csv", index=False)

    # 3. Feature-to-Target Relationship (Mutual Information)
    print("🎯 Calculating Mutual Information Scores...")
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_results = pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)
    mi_results.to_csv(f"{REPORT_DIR}mutual_information.csv")

    # 4. Outlier & Distribution Analysis
    print("📉 Detecting Noise and Outliers...")
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=X, orient="h", palette="Set2")
    plt.title("Feature Distribution and Outlier Detection")
    plt.savefig(f"{REPORT_DIR}distribution_boxplots.png")
    plt.close()

    print(f"
✅ AUDIT COMPLETE. Results saved in {REPORT_DIR}")
    print(f"Top 3 High-Signal Features (MI):
{mi_results.head(3)}")
    print(f"
Top 3 Potential Redundant Features (VIF > 5):
{vif_data[vif_data['VIF'] > 5].head(3)}")

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        run_technical_eda()
    else:
        print("Data not found. Run preprocessing first.")
