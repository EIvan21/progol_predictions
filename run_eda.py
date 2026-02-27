import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from fpdf import FPDF

# Setup Styles
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 300

# Setup Directories
os.makedirs("reports/eda_plots", exist_ok=True)
DB_PATH = "data/progol.db"

def get_data_from_db():
    if not os.path.exists(DB_PATH): return None
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'FT'", conn)
    conn.close()
    return df

def generate_strategic_viz(df):
    print("🚀 GENERATING SENIOR STRATEGIC VISUALIZATIONS...")
    
    # 1. Efficiency Heatmap by League
    plt.figure(figsize=(12, 8))
    league_map = {262: "Liga MX", 39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1", 253: "MLS"}
    df['league_name'] = df['league_id'].map(league_map).fillna(df['league_id'].astype(str))
    
    # Calculate Efficiency Metrics
    df['off_eff'] = df['goals_home'] / (df['home_shots'] + 1)
    df['press_idx'] = (df['home_possession'] * df['home_corners']) / 100
    
    league_stats = df.groupby('league_name')[['off_eff', 'press_idx']].mean().sort_values('off_eff', ascending=False)
    sns.heatmap(league_stats, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Strategic Efficiency Heatmap by League", fontsize=14, fontweight='bold')
    plt.savefig("reports/eda_plots/efficiency_heatmap.png", bbox_inches='tight')
    plt.close()

    # 2. Field Tilt Analysis (Possession vs Corners)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(min(2000, len(df))), x='home_possession', y='home_corners', hue='league_name', alpha=0.5)
    plt.title("Field Tilt: Possession vs Corners Correlation", fontsize=14, fontweight='bold')
    plt.savefig("reports/eda_plots/field_tilt.png", bbox_inches='tight')
    plt.close()

    # 3. Outcome Pie Chart (Standard)
    plt.figure(figsize=(8, 6))
    res = df.apply(lambda r: 'Home' if r['goals_home'] > r['goals_away'] else ('Draw' if r['goals_home'] == r['goals_away'] else 'Away'), axis=1)
    res.value_counts().plot.pie(autopct='%1.1f%%', colors=['#4e79a7', '#f28e2b', '#e15759'], wedgeprops={'width': 0.5})
    plt.title("Global Match Outcome Distribution", fontsize=14, fontweight='bold')
    plt.savefig("reports/eda_plots/outcome_dist.png", bbox_inches='tight')
    plt.close()

    return df

class EDA_Report(FPDF):
    def header(self):
        self.set_fill_color(33, 47, 61)
        self.rect(0, 0, 210, 35, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 22)
        self.cell(0, 25, 'IPI STRATEGIC AUDIT REPORT', 0, 1, 'C')
        self.ln(10)

def generate_pdf(df):
    print("📄 COMPILING STRATEGIC PDF...")
    pdf = EDA_Report()
    pdf.add_page()
    pdf.set_text_color(33, 47, 61)

    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Executive Summary", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, f"Audit performed on {len(df):,} professional matches. Focused on Strategy 7 (Efficiency Interactions).")
    pdf.ln(5)

    # Visualization 1: Heatmap
    pdf.image("reports/eda_plots/efficiency_heatmap.png", x=15, y=None, w=180)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, "Insight: Certain leagues favor clinical efficiency (Off_Eff) over sustained pressure (Press_Idx).", 0, 1, 'C')
    pdf.ln(15)

    # Visualization 2: Field Tilt
    pdf.add_page()
    pdf.image("reports/eda_plots/field_tilt.png", x=15, y=None, w=180)
    pdf.ln(5)
    pdf.cell(0, 5, "Insight: Possession without corners indicates 'safe' play with lower scoring probability.", 0, 1, 'C')
    pdf.ln(15)

    # Visualization 3: Pie
    pdf.image("reports/eda_plots/outcome_dist.png", x=55, y=None, w=100)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf.output(f"reports/strategic_eda_{timestamp}.pdf")
    print(f"✅ STRATEGIC REPORT GENERATED: reports/strategic_eda_{timestamp}.pdf")

if __name__ == "__main__":
    df = get_data_from_db()
    if df is not None:
        generate_strategic_viz(df)
        generate_pdf(df)
