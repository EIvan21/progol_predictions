# 📊 Progol System: Data Architecture & Feature Philosophy

This document explains where our data lives, how we chose our variables, and why we are focusing on specific "High-Signal" dimensions.

---

## 1. Data Storage: Where is it?
Currently, your data is stored in a **Local SQLite Database** (`data/progol.db`).

### Why SQLite? (The "Open Mode" Strategy)
- **Zero Latency:** Since the database is a file on the same disk as the CPU, fetching 70,000 rows takes milliseconds.
- **Portability:** The entire history can be backed up just by copying one file.
- **Is it enough?** For a single-user project running on a VM, **Yes, it is optimal.** 

### Can we move it to a "Distance" (GCP Cloud)?
If you want to access the data from multiple computers or a mobile app, we can move it to:
1. **Google BigQuery:** Best for massive analysis (Millions of rows).
2. **Google Cloud SQL (PostgreSQL):** Best for real-time applications.
**Recommendation:** Unless you plan to have multiple people using the system at once, keep it in SQLite to save money and complexity.

---

## 2. Active Variables: What are we using and why?

We use a **Differential Engine**. Instead of looking at a team's isolated stats, we look at the **GAP** between the Home and Away team.

### The Signal Core
| Variable | Why? | Example Value |
| :--- | :--- | :--- |
| **Elo Rating** | Measures absolute team power. A team that beats a strong opponent gains more points than one that beats a weak one. | Home: 1650, Away: 1420 (Gap: +230) |
| **Rolling Form** | Measures momentum. How many points (3, 1, 0) did they get in the last 5 matches? | Home: 2.2 pts/avg, Away: 1.0 pts/avg |
| **Goal Differential** | Measures dominance. Does the team win by 1 goal or 4? | +1.5 goals gap |
| **Shot Volume** | Measures "Offensive Intent." Some teams get lucky with 1 shot; teams with 15 shots are more likely to win long-term. | +5.2 shots gap |
| **Target Encoding** | Measures "Latent Bias." Referees and Venues have hidden patterns (e.g., a Ref that never calls penalties for Away teams). | Ref_Enc: 0.48 (48% Home Win Rate) |

---

## 3. Technical Example: Data Call
To get these variables, we perform a complex **recursive query** on our database. 

**Conceptual SQL Example:**
```sql
-- Get the average shots for Team A in their last 5 matches BEFORE today
SELECT AVG(home_shots) 
FROM matches 
WHERE (home_id = 123) 
AND date < '2024-02-27' 
ORDER BY date DESC LIMIT 5;
```
This is then subtracted from Team B's average to create the **`roll_sh_diff`** feature.

---

## 4. Why aren't we using "More" variables?
You asked a senior question: *Why not add 100 more columns?*

In sports data, there is a phenomenon called **The Curse of Dimensionality**. 
1. **Overfitting:** If we add "Number of Yellow Cards" or "Wind Speed," the model might find a "fake" pattern (e.g., "Team A always wins when it's 22°C"). This won't work next week.
2. **Data Sparsity:** API-Sports has detailed data for the Premier League, but **zero** data for the "Mexican Expansion League" second division. If we use a variable that only exists for 50% of the matches, it ruins the training for the other 50%.
3. **Signal-to-Noise Ratio:** Goals, Shots, and Elo contain 90% of the "truth." Adding 50 minor variables only adds 10% more truth but **500% more noise.**

---

## 🚀 Conclusion
We are using a **"Lean & Mean"** feature set that maximizes signal while maintaining zero leakage. The current architecture is designed for **Generalization**—ensuring it works just as well for Liga MX as it does for the Eredivisie.
