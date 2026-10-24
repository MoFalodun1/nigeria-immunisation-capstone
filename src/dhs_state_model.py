"""
Stage 2 of the capstone: state-level predictors from the DHS Program API,
then a baseline vs advanced model of state vaccination coverage.

    py -m pip install requests pandas numpy openpyxl scikit-learn
    py dhs_state_model.py                      # pulls data, builds panel, runs models
    py dhs_state_model.py --api-key YOUR_KEY   # optional DHS API key
    py dhs_state_model.py --panel-csv dhs_output/state_panel.csv   # skip the download, re-run models only

WHAT IT DOES
1. Reads the live DHS indicator catalogue for Nigeria and finds a set of
   predictors (antenatal care, facility delivery, women's education, fertility,
   household wealth). It tries known indicator IDs first, then searches the
   catalogue labels. It PRINTS exactly which indicator it picked: read that
   list before trusting any result.
2. Pulls state values for the three surveys with state data
   (NG2013DHS, NG2018DHS, NG2024DHS) and builds a state-by-survey table
   (37 states x 3 surveys = up to 111 rows).
3. Targets: DTP3 coverage, and the zero-dose proxy (100 - DTP1).
4. Models: training-mean baseline, ridge regression (interpretable baseline),
   random forest and gradient boosting (advanced).
5. Validation (no leakage between train and test):
   a) leave-one-state-out: every row of a state is held out together
   b) forward-in-time: train on 2013 and 2018, test on 2024
6. Writes dhs_output/dhs_state_model_results.xlsx and experiment_log.csv.

LIMITS (say these in the report)
- Aggregated data: every result is ECOLOGICAL (about states, not children).
- About 111 rows, so models are small and results are noisy.
- No vaccination-derived variable is used as a predictor (leakage guard in code).
- Associations are not causes.
"""
import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.dhsprogram.com/rest/dhs"
COUNTRY = "NG"
SURVEYS = ["NG2013DHS", "NG2018DHS", "NG2024DHS"]
PER_PAGE = 1000
SEED = 42
MIN_COVERAGE = 0.80      # a predictor must exist for at least 80% of state-survey rows
TARGET_IDS = {"DTP1": "CH_VACC_C_DP1", "DTP3": "CH_VACC_C_DP3"}

# key, description, known indicator IDs (tried first), keyword sets (fallback label search)
PREDICTORS = [
    ("anc_4plus", "Antenatal care: 4+ visits",
     ["RH_ANCN_W_N4P"], [("antenatal", "4+"), ("antenatal", "four or more")]),
    ("anc_skilled", "Antenatal care from a skilled provider",
     ["RH_ANCP_W_SKP"], [("antenatal care", "skilled")]),
    ("facility_birth", "Births in a health facility",
     ["RH_DELP_C_DHF"], [("delivered in a health facility",), ("delivery", "health facility")]),
    ("skilled_birth", "Births assisted by a skilled provider",
     ["RH_DELA_C_SKP"], [("assistance", "skilled")]),
    ("women_no_edu", "Women with no education",
     ["ED_EDAT_W_NED"], [("women", "no education")]),
    ("women_sec_plus", "Women with secondary or higher education",
     ["ED_EDAT_W_SEH"], [("women", "secondary or higher")]),
    ("tfr", "Total fertility rate",
     ["FE_FRTR_W_TFR"], [("total fertility rate",)]),
    ("wealth_lowest", "Population in the lowest wealth quintile",
     ["HC_WIXQ_P_LOW"], [("lowest wealth quintile",)]),
]

# Anything about vaccination must never be a predictor (leakage guard)
LEAK = re.compile(r"vaccin|dpt|penta|measles|bcg|polio|immuni|card", re.I)

ZONES = {}
for _z, _m in {
    "North Central": ["FCT Abuja", "Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
}.items():
    for _s in _m:
        ZONES[_s] = _z


# ---------------------------------------------------------------- API helpers
def get_json(endpoint, params, tries=4):
    url = f"{BASE}/{endpoint}"
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    sys.exit(f"Could not reach {url} after {tries} tries: {last}\n"
             "Check your internet connection, or whether your network blocks api.dhsprogram.com.")


def get_all_pages(endpoint, params):
    rows, page = [], 1
    while True:
        data = get_json(endpoint, dict(params, perpage=PER_PAGE, page=page, f="json"))
        rows.extend(data.get("Data", []) or [])
        if page >= int(data.get("TotalPages", 1) or 1):
            return rows
        page += 1
        time.sleep(0.3)


def fetch_catalogue(api_key):
    params = {"countryIds": COUNTRY}
    if api_key:
        params["apiKey"] = api_key
    cat = pd.DataFrame(get_all_pages("indicators", params))
    if cat.empty or "IndicatorId" not in cat:
        sys.exit("The indicator catalogue came back empty. Send me the console output.")
    for c in ("Label", "Definition"):
        if c not in cat:
            cat[c] = ""
    cat["_label"] = cat["Label"].astype(str).str.lower()
    cat["_text"] = (cat["Label"].astype(str) + " " + cat["Definition"].astype(str)).str.lower()
    return cat


def resolve_predictors(cat):
    ids = set(cat["IndicatorId"])
    chosen, report = {}, []
    for key, desc, known, kwsets in PREDICTORS:
        hit, how = None, ""
        for c in known:
            if c in ids:
                hit, how = c, "known ID"
                break
        if hit is None:
            for field, where in (("_label", "label"), ("_text", "label+definition")):
                for kws in kwsets:
                    m = cat[cat[field].apply(lambda t: all(k in t for k in kws))]
                    m = m[~m["Label"].astype(str).str.contains(LEAK)]
                    if len(m):
                        hit, how = m.iloc[0]["IndicatorId"], f"{where} search {kws} ({len(m)} candidates, took the first)"
                        break
                if hit is not None:
                    break
        if hit is not None:
            label = cat.loc[cat["IndicatorId"] == hit, "Label"].iloc[0]
            if LEAK.search(str(label)):
                report.append((key, desc, "REJECTED (vaccination-related label)", hit, label))
                continue
            chosen[key] = hit
            report.append((key, desc, how, hit, label))
        else:
            report.append((key, desc, "NOT FOUND", "", ""))
    return chosen, pd.DataFrame(report, columns=["Key", "Wanted", "How found", "IndicatorId", "Catalogue label"])


# ---------------------------------------------------------------- data
def pull_state_panel(pred_ids, api_key):
    all_ids = list(TARGET_IDS.values()) + list(pred_ids.values())
    params = {"countryIds": COUNTRY, "surveyIds": ",".join(SURVEYS),
              "indicatorIds": ",".join(all_ids), "breakdown": "subnational"}
    if api_key:
        params["apiKey"] = api_key
    raw = pd.DataFrame(get_all_pages("data", params))
    if raw.empty:
        sys.exit("The API returned no state data. Send me the console output.")
    return raw


def build_panel(raw, pred_ids):
    id_to_key = {v: k for k, v in TARGET_IDS.items()}
    id_to_key.update({v: k for k, v in pred_ids.items()})
    g = lambda c: raw[c] if c in raw else pd.Series([None] * len(raw), index=raw.index)
    df = pd.DataFrame({
        "SurveyId": g("SurveyId").astype(str),
        "IndicatorId": g("IndicatorId"),
        "Label": g("CharacteristicLabel").astype(str),
        "Value": pd.to_numeric(g("Value"), errors="coerce"),
        "N": pd.to_numeric(g("DenominatorUnweighted"), errors="coerce"),
        "Pref": pd.to_numeric(g("IsPreferred"), errors="coerce"),
    })
    df = df[df["Label"].str.startswith("..")].copy()          # states only (zones have no leading "..")
    df["State"] = df["Label"].str.lstrip(". ").str.strip()
    df["Year"] = df["SurveyId"].str.extract(r"(\d{4})")[0].astype(float)
    df["Key"] = df["IndicatorId"].map(id_to_key)
    if df["Pref"].notna().any():
        df = df[(df["Pref"] == 1) | df["Pref"].isna()]
    df = df.dropna(subset=["Key", "Year", "Value"])
    wide = df.pivot_table(index=["State", "Year"], columns="Key", values="Value", aggfunc="first")
    n3 = df[df["Key"] == "DTP3"].groupby(["State", "Year"])["N"].first().rename("N_DTP3")
    panel = wide.join(n3).reset_index()
    panel["Year"] = panel["Year"].astype(int)
    panel["Zone"] = panel["State"].map(ZONES)
    if panel["Zone"].isna().any():
        print("  WARNING: unmapped state names:", sorted(panel.loc[panel["Zone"].isna(), "State"].unique()))
    for need in ("DTP1", "DTP3"):
        if need not in panel:
            sys.exit(f"{need} is missing from the pulled data. Send me the console output.")
    panel["ZeroDoseProxy"] = 100 - panel["DTP1"]
    return panel


# ---------------------------------------------------------------- models
def run_models(panel, preds, out_dir, pred_report):
    try:
        from sklearn.base import clone
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\nscikit-learn is not installed, so the models were skipped.\n"
              "Run:  py -m pip install scikit-learn\nthen:  py dhs_state_model.py --panel-csv dhs_output/state_panel.csv")
        return None

    def make_models():
        imp = lambda: SimpleImputer(strategy="median")
        return {
            "Baseline 0: training mean": Pipeline([("i", imp()), ("m", DummyRegressor(strategy="mean"))]),
            "Baseline 1: Ridge regression": Pipeline([("i", imp()), ("s", StandardScaler()),
                                                      ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))]),
            "Advanced A: Random forest": Pipeline([("i", imp()), ("m", RandomForestRegressor(
                n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=SEED, n_jobs=-1))]),
            "Advanced B: Gradient boosting": Pipeline([("i", imp()), ("m", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED))]),
        }

    def metrics(y, p):
        return (float(np.sqrt(mean_squared_error(y, p))), float(mean_absolute_error(y, p)), float(r2_score(y, p)))

    targets = {"DTP3 coverage (%)": "DTP3", "Zero-dose proxy (100-DTP1, %)": "ZeroDoseProxy"}
    rows, preds_out, imp_rows, run_id = [], [], [], 0
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    data = panel.dropna(subset=["DTP3", "DTP1"]).reset_index(drop=True)
    X = data[preds]
    groups = data["State"].values
    train_t = data["Year"] < data["Year"].max()
    test_t = ~train_t

    for tname, tcol in targets.items():
        y = data[tcol].values
        # ---- (a) leave-one-state-out
        oof = {m: np.zeros(len(data)) for m in make_models()}
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            for name, mdl in make_models().items():
                f = clone(mdl).fit(X.iloc[tr], y[tr])
                oof[name][te] = f.predict(X.iloc[te])
        # ---- (b) forward in time
        fwd, fitted = {}, {}
        for name, mdl in make_models().items():
            f = clone(mdl).fit(X[train_t], y[train_t])
            fwd[name] = f.predict(X[test_t])
            fitted[name] = f
        for name in oof:
            for vname, yy, pp in (("Leave-one-state-out", y, oof[name]),
                                  (f"Train <= {int(data.loc[train_t,'Year'].max())}, test {int(data.loc[test_t,'Year'].max())}",
                                   y[test_t], fwd[name])):
                run_id += 1
                rmse, mae, r2 = metrics(yy, pp)
                rows.append({"Run": run_id, "Timestamp": stamp, "Target": tname, "Model": name, "Validation": vname,
                             "N_test_rows": len(yy), "RMSE_pp": round(rmse, 2), "MAE_pp": round(mae, 2),
                             "R2": round(r2, 3), "Seed": SEED, "Features": ", ".join(preds)})
        # ---- error analysis table (best non-baseline model by leave-one-state-out RMSE)
        cand = {n: metrics(y, oof[n])[0] for n in oof if not n.startswith("Baseline 0")}
        best = min(cand, key=cand.get)
        pr = data[["State", "Zone", "Year"]].copy()
        pr["Target"] = tname
        pr["Best_model"] = best
        pr["Actual"] = y
        pr["Predicted_leave_one_state_out"] = oof[best]
        pr["Residual"] = pr["Actual"] - pr["Predicted_leave_one_state_out"]
        pr["Abs_error"] = pr["Residual"].abs()
        preds_out.append(pr)
        # ---- feature importance
        ridge = fitted["Baseline 1: Ridge regression"]
        coefs = ridge.named_steps["m"].coef_
        for k, c in zip(preds, coefs):
            imp_rows.append({"Target": tname, "Method": "Ridge standardised coefficient (fit 2013+2018)",
                             "Feature": k, "Value": round(float(c), 3)})
        rf = fitted["Advanced A: Random forest"]
        pi = permutation_importance(rf, X[test_t], y[test_t], n_repeats=30, random_state=SEED,
                                    scoring="neg_root_mean_squared_error")
        for k, m_, s_ in zip(preds, pi.importances_mean, pi.importances_std):
            imp_rows.append({"Target": tname, "Method": "Random forest permutation importance on test year (RMSE rise, pp)",
                             "Feature": k, "Value": round(float(m_), 3)})

    metrics_df = pd.DataFrame(rows)
    pred_df = pd.concat(preds_out, ignore_index=True)
    zone_err = (pred_df.groupby(["Target", "Zone"])["Abs_error"].mean().round(2).reset_index()
                .rename(columns={"Abs_error": "Mean_abs_error_pp"}))
    state_err = (pred_df.groupby(["Target", "State", "Zone"])["Abs_error"].mean().round(2).reset_index()
                 .rename(columns={"Abs_error": "Mean_abs_error_pp"})
                 .sort_values(["Target", "Mean_abs_error_pp"], ascending=[True, False]))
    imp_df = pd.DataFrame(imp_rows)
    checks = pd.DataFrame([
        ("Rows used (state x survey)", len(data)),
        ("States", data["State"].nunique()),
        ("Surveys", ", ".join(map(str, sorted(data["Year"].unique())))),
        ("Predictors used", ", ".join(preds)),
        ("Leakage guard", "No predictor label matches vaccination terms (vaccin, DPT, penta, measles, BCG, polio, immuni, card)."),
        ("Validation", "Leave-one-state-out keeps all surveys of a held-out state out of training; forward-in-time trains on earlier surveys only."),
        ("Random seed", SEED),
        ("Caveat", "Ecological, aggregated data (about states, not children). Associations are not causes."),
    ], columns=["Check", "Result"])

    from openpyxl.styles import Font
    out_dir = Path(out_dir)
    xlsx = out_dir / "dhs_state_model_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        metrics_df.to_excel(xw, sheet_name="Metrics", index=False)
        pred_df.round(2).to_excel(xw, sheet_name="Predictions", index=False)
        zone_err.to_excel(xw, sheet_name="Error_by_zone", index=False)
        state_err.to_excel(xw, sheet_name="Error_by_state", index=False)
        imp_df.to_excel(xw, sheet_name="Importance", index=False)
        pred_report.to_excel(xw, sheet_name="Predictors_found", index=False)
        data.round(2).to_excel(xw, sheet_name="Panel", index=False)
        checks.to_excel(xw, sheet_name="Checks", index=False)
        for ws in xw.book.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True)
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    log = out_dir / "experiment_log.csv"
    metrics_df.to_csv(log, mode="a", index=False, header=not log.exists())
    return metrics_df, xlsx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out-dir", default="dhs_output")
    ap.add_argument("--panel-csv", default=None, help="skip the download and use this panel file")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    if a.panel_csv:
        panel = pd.read_csv(a.panel_csv)
        known = [k for k, *_ in PREDICTORS if k in panel]
        report = pd.DataFrame({"Key": known, "How found": "from panel csv"})
        print(f"Loaded {len(panel)} rows from {a.panel_csv}")
    else:
        print("1/4 Reading the Nigeria indicator catalogue ...")
        cat = fetch_catalogue(a.api_key)
        print(f"    {len(cat)} indicators listed")
        pred_ids, report = resolve_predictors(cat)
        print("\nPREDICTOR REPORT (read this: it shows which indicator was used for each predictor)")
        print(report.to_string(index=False))
        if len(pred_ids) < 3:
            sys.exit("\nFewer than 3 predictors were found. Send me this output (and indicator_catalogue_all.csv).")
        cat.drop(columns=["_text", "_label"]).to_csv(out / "indicator_catalogue_all.csv", index=False)
        print("\n2/4 Downloading state data ...")
        raw = pull_state_panel(pred_ids, a.api_key)
        raw.to_csv(out / "raw_state_predictors.csv", index=False)
        print(f"    {len(raw):,} rows")
        print("3/4 Building the state-by-survey panel ...")
        panel = build_panel(raw, pred_ids)
        panel.to_csv(out / "state_panel.csv", index=False)
        print(f"    {len(panel)} rows, {panel['State'].nunique()} states, surveys {[int(y) for y in sorted(panel['Year'].unique())]}")

    cols = [k for k, *_ in PREDICTORS if k in panel.columns]
    cov = panel[cols].notna().mean().round(2)
    print("\nPredictor coverage (share of rows with a value):")
    print(cov.to_string())
    preds = [k for k in cols if cov[k] >= MIN_COVERAGE]
    dropped = [k for k in cols if k not in preds]
    if dropped:
        print("Dropped for low coverage:", ", ".join(dropped))
    if len(preds) < 3:
        sys.exit("Fewer than 3 usable predictors. Send me the console output.")
    print("Using:", ", ".join(preds))

    print("\n4/4 Fitting models and validating (about a minute) ...")
    res = run_models(panel, preds, out, report)
    if res is None:
        return
    metrics_df, xlsx = res
    print("\nRESULTS (lower RMSE/MAE is better; R2 below 0 means worse than predicting the average)")
    show = metrics_df[["Target", "Model", "Validation", "RMSE_pp", "MAE_pp", "R2"]]
    print(show.to_string(index=False))
    print(f"\nSaved: {xlsx}\n       {out / 'experiment_log.csv'}\n       {out / 'state_panel.csv'}")
    print("Send me the RESULTS table above and the PREDICTOR REPORT.")


if __name__ == "__main__":
    main()
