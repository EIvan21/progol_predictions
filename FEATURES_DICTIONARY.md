# 📊 IPI Features Dictionary (Integrated Progol Index)

This document describes the active dimensions used by the Progol Prediction Engine during training and inference. The system utilizes a **Differential Architecture**, where most features represent the gap between the Home and Away teams.

---

## 1. Contextual & Identity Features

| Feature Name | Data Type | Description | Values / Examples | Nature |
| :--- | :--- | :--- | :--- | :--- |
| `league_id` | Integer | Official API-Football unique identifier for the league. | `262` (Liga MX), `39` (EPL) | Categorical |
| `venue_encoded` | Float | Mean Target Encoded value of the stadium. Represents historical Home Win Rate at this venue. | `0.42` to `0.65` | Numeric (Latent) |
| `ref_encoded` | Float | Mean Target Encoded value of the referee. Represents historical bias or game flow impact. | `0.30` to `0.55` | Numeric (Latent) |

---

## 2. Differential Performance Gaps (Home - Away)
*Note: A positive value indicates Home dominance; a negative value indicates Away dominance.*

| Feature Name | Data Type | Description | Min/Max (Approx) |
| :--- | :--- | :--- | :--- |
| `roll_gf_diff` | Float | Difference in Average Goals Scored (last 5 games). | `-3.0` to `+3.0` |
| `roll_ga_diff` | Float | Difference in Average Goals Conceded (last 5 games). | `-3.0` to `+3.0` |
| `roll_sh_diff` | Float | Difference in Average Shots on Goal (last 5 games). | `-10.0` to `+10.0` |
| `roll_po_diff` | Float | Difference in Average Possession percentage. | `-25.0` to `+25.0` |
| `roll_co_diff` | Float | Difference in Average Corner Kicks awarded. | `-8.0` to `+8.0` |

---

## 3. High-Signal Interaction Differentials
*These are derived "Efficiency" metrics calculated as: (Home Metric - Away Metric).*

| Feature Name | Data Type | Description | Logic |
| :--- | :--- | :--- | :--- |
| `off_efficiency_diff` | Float | Gap in "Clinical Finishing". | `(Avg Goals / Avg Shots)` |
| `pressure_index_diff` | Float | Gap in "Field Tilt" or sustained pressure. | `(Possession * Corners) / 100` |
| `def_resilience_diff` | Float | Gap in "Defensive Survival" ability. | `(Avg Shots Allowed / Avg Goals Against)` |

---

## 4. Derived Strategic Features

| Feature Name | Data Type | Description | Logic |
| :--- | :--- | :--- | :--- |
| `power_score_diff` | Float | Gap in current winning momentum (last 10 games). | `Win_Rate_Home - Win_Rate_Away` |
| `cs_rate_diff` | Float | Gap in defensive stability (Clean Sheet probability). | `CS_Rate_Home - CS_Rate_Away` |

---

## 🚀 Data Summary (Production Scale)
- **Total Features:** 11 active signal columns.
- **Normalization:** All numeric features are processed via `StandardScaler` (Mean=0, Std=1) before entering the Stacking Ensemble.
- **Missing Values:** Handled via `Safe Defaults` (Rolling averages default to 0 if no history exists; Power Score defaults to 0.33).
