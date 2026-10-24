# Childhood Immunisation Coverage and Inequalities in Nigeria
*eHA Academy Data Science Capstone, Public Health Analytics track*

## 1. Problem (5W)
- **Who:** immunisation programme managers at the National Primary Health Care Development Agency and State Primary Health Care Boards
- **What:** Too many children miss routine vaccines, and the gap is uneven across places and groups. In the 2023-24 DHS, DTP3 coverage was 53.4%, the zero-dose proxy (100 minus DTP1) was 37.1%, and 39.4% of children were fully vaccinated.
- **Where / When:** Nigeria; DHS rounds from 1990 to 2024, with the 2018 and 2023-24 rounds used for group comparisons.
- **Why:** Programme managers need to know where and among whom coverage is lowest so outreach can be targeted.
- **SMART metrics:** measured from the DHS (children aged 12-23 months), nationally and by state. The baselines are the 2023-24 values; the targets are proposed for the next DHS round and should be agreed by the team.

| Metric | Specific and measurable | Baseline (2023-24) | Proposed target (next DHS round) |
|---|---|---|---|
| Coverage | National DTP3 coverage | 53.4% | [team to set] |
| Zero-dose | Children with no DTP1 (100 minus DTP1) | 37.1% | Below 30% |
| Low-coverage states | States with DTP3 below 50% | 12 of 37 | None |
| Wealth gap | Richest minus poorest fifth, DTP3 | 49.6 points | Below 40 points |
| Model | Unseen-state DTP3 error vs training mean | 11.0 vs 25.3 points (56% lower) | Achieved (at least 50% lower) |

All are time-bound to the next DHS round and relevant to targeting outreach.

**Questions:** (1) How has coverage changed over time? (2) Where and among which groups is it lowest? (3) What indicators are associated with state coverage, and can a model predict it? (4) What should programme managers do with this?

## 2. Data
| Source | What it gives | Used for |
|---|---|---|
| WHO/UNICEF WUENIC 2025 revision, survey data | National survey coverage for all countries; 22 Nigerian surveys (1,140 rows) | Trends, survey comparison, one-line African context |
| DHS Program API (`src/dhs_api_pull.py`) | State and group values for the 2013, 2018 and 2024 DHS | State gaps; wealth, residence and education gaps; modelling |

- 12 cleaning issues are documented in the workbook sheet `Cleaning_Log` (for example, 122 rows with a blank survey name, and zone rows removed from the state table).
- The DHS API national values for 2018 and 2024 equal the WUENIC-based values exactly (difference 0.0).
- **Labels:** DHS surveys are named by fieldwork year (NDHS 2018 and NDHS 2023-24). WUENIC labels the same surveys by the children's birth year (2017 and 2022).

## 3. Repository structure
```
data/raw/        downloaded source files (not edited)
data/processed/  cleaned tables and Excel workbooks
src/             scripts (data pull, modelling)
notebooks/       EDA and modelling notebooks
reports/         technical report, slide deck
dashboard/       Power BI file (.pbix)
docs/            experiment log, module-to-method table
```

## 4. How to reproduce
```
py -m pip install -r requirements.txt
py src/dhs_api_pull.py      # pulls the DHS indicator tables
py src/dhs_state_model.py   # fits the models; logs every run to experiment_log.csv
```
The Excel workbooks are in `data/processed/` and the Power BI dashboard is in `dashboard/`. The random seed is 42.

## 5. Methods
See `docs/module_to_method.md` and `docs/experiment_log.md`.

- **Data:** 111 state-survey rows (36 states and the FCT across three surveys). Targets: DTP3 coverage and the zero-dose proxy.
- **Predictors (8, none derived from vaccination):** antenatal care 4+ visits; antenatal care from a skilled provider; births in a health facility; births with a skilled attendant; women with no education; women with secondary or higher education; total fertility rate; share of the population in the lowest wealth quintile.
- **Models:** training-set mean (baseline), ridge regression, random forest, gradient boosting.
- **Validation:** leave-one-state-out, and forward in time (train on 2013 and 2018, test on 2024). Metrics: RMSE, MAE and R-squared.

## 6. Key results
RMSE in percentage points (lower is better).

| Target | Validation | Training mean | Ridge | Random forest | Gradient boosting |
|---|---|---|---|---|---|
| DTP3 | Leave-one-state-out | 25.28 | **11.01** | 11.38 | 11.34 |
| DTP3 | Train 2013-18, test 2024 | 22.51 | **12.86** | 13.39 | 14.77 |
| Zero-dose proxy | Leave-one-state-out | 25.60 | **9.58** | 9.84 | 10.11 |
| Zero-dose proxy | Train 2013-18, test 2024 | 22.86 | 10.89 | 10.96 | **10.22** |

- Ridge cuts DTP3 error for an unseen state from 25.3 to 11.0 points (R-squared 0.80), 56% lower than the baseline.
- Advanced models did not beat the simple model; ridge was best in 3 of 4 comparisons. We chose ridge for interpretability.
- Women's secondary education is the strongest predictor (shuffling it raises random-forest test error by 2.3 points for DTP3 and 3.9 for zero-dose).
- Ebonyi, Niger and Bauchi are poorly predicted for both targets.

## 7. Limitations
- State-level results are ecological: they describe states, not children, and show association, not cause.
- Small sample: 111 rows and 37 states; state samples are 49 to 271 children, and the API gives no confidence intervals for states.
- Survey types disagree, and the WUENIC survey rows are flagged CRUDE (not official WHO/UNICEF estimates). The African comparison mixes survey years.
- The single-variable check (women's secondary education alone) was chosen after seeing the main results (post hoc).
- Only three survey rounds, so findings may not generalise.

## 8. Team
| Member | Role | GitHub |
|---|---|---|
| Moses O. Falodun | Team lead: data cleaning, Power BI dashboard and charts | @MoFalodun1 |
| Muflihat Abdulmajid | Technical report | @Muflih14 |
| Aisha Hilal Galadanchi | Problem statement, review and compilation | aishahilalgaladanchi-crypto |

One commit per member per active day.
