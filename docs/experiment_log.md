# Experiment log
One row per run. Never delete rows; add a new one.

Common to all runs: data = DHS API pull, 111 state-survey rows (2013, 2018, 2024); features = the 8 predictors in README section 5; random seed 42; no vaccination-derived predictor (checked in code). Fill in Date and Owner from your own records.

| # | Date | Owner | Model / method | Target | Validation | RMSE (pts) | Notes / leakage check |
|---|---|---|---|---|---|---|---|
| 1 | [date] | [owner] | Baseline: training-set mean | DTP3 | Leave-one-state-out | 25.28 | Whole held-out states kept out of training |
| 2 | [date] | [owner] | Ridge regression (standardised inputs) | DTP3 | Leave-one-state-out | 11.01 | Whole held-out states kept out of training |
| 3 | [date] | [owner] | Random forest | DTP3 | Leave-one-state-out | 11.38 | Whole held-out states kept out of training |
| 4 | [date] | [owner] | Gradient boosting | DTP3 | Leave-one-state-out | 11.34 | Whole held-out states kept out of training |
| 5 | [date] | [owner] | Baseline: training-set mean | DTP3 | Train 2013-18, test 2024 | 22.51 | Forward test uses only earlier surveys |
| 6 | [date] | [owner] | Ridge regression (standardised inputs) | DTP3 | Train 2013-18, test 2024 | 12.86 | Forward test uses only earlier surveys |
| 7 | [date] | [owner] | Random forest | DTP3 | Train 2013-18, test 2024 | 13.39 | Forward test uses only earlier surveys |
| 8 | [date] | [owner] | Gradient boosting | DTP3 | Train 2013-18, test 2024 | 14.77 | Forward test uses only earlier surveys |
| 9 | [date] | [owner] | Baseline: training-set mean | Zero-dose proxy | Leave-one-state-out | 25.60 | Whole held-out states kept out of training |
| 10 | [date] | [owner] | Ridge regression (standardised inputs) | Zero-dose proxy | Leave-one-state-out | 9.58 | Whole held-out states kept out of training |
| 11 | [date] | [owner] | Random forest | Zero-dose proxy | Leave-one-state-out | 9.84 | Whole held-out states kept out of training |
| 12 | [date] | [owner] | Gradient boosting | Zero-dose proxy | Leave-one-state-out | 10.11 | Whole held-out states kept out of training |
| 13 | [date] | [owner] | Baseline: training-set mean | Zero-dose proxy | Train 2013-18, test 2024 | 22.86 | Forward test uses only earlier surveys |
| 14 | [date] | [owner] | Ridge regression (standardised inputs) | Zero-dose proxy | Train 2013-18, test 2024 | 10.89 | Forward test uses only earlier surveys |
| 15 | [date] | [owner] | Random forest | Zero-dose proxy | Train 2013-18, test 2024 | 10.96 | Forward test uses only earlier surveys |
| 16 | [date] | [owner] | Gradient boosting | Zero-dose proxy | Train 2013-18, test 2024 | 10.22 | Forward test uses only earlier surveys |
| 17 | [date] | [owner] | Random forest, permutation importance | DTP3 and zero-dose proxy | Test error after shuffling each predictor | +2.3 (DTP3), +3.9 (zero-dose) for women's secondary education | Strongest predictor; every other predictor matters less |
| 18 | [date] | [owner] | Ridge with women's secondary education only (exploratory) | DTP3 | Leave-one-state-out | 12.0 (vs 11.0 with all eight) | Post hoc: chosen after seeing the main results |

Rules: fix the random seed; split by state, not by row; exclude any variable derived from vaccination status.
