"""
Pull Nigerian childhood-immunisation indicators from the DHS Program API
(national, state/region, and background-group breakdowns) into tidy tables
for Excel and Power BI.

    pip install requests pandas openpyxl
    python dhs_api_pull.py                # default: Nigeria, output in ./dhs_output
    python dhs_api_pull.py --api-key YOUR_KEY --out-dir my_folder

IMPORTANT - READ FIRST
1. Aggregated indicators only: no microdata, so no approval is needed. That also
   means you can analyse ONE grouping at a time (state, OR wealth, OR residence),
   not combinations, and any "factors" analysis is ecological, not individual-level.
2. Indicator IDs below for DTP1, DTP3 and measles were confirmed in public package
   documentation. IDs for "all basic vaccinations" and "no vaccinations" are listed
   under two spellings (VACC / VACS) because sources disagree. The script checks
   them against the live indicator catalogue and REPORTS what it found; read the
   console output and indicator_catalogue.csv before trusting the tables.
3. Response field names are handled defensively (.get). If a column you expect is
   empty, open raw_all_levels.csv and check the field names the API actually sent.

Outputs (in --out-dir):
    indicator_catalogue.csv     vaccination-related indicators the API lists for Nigeria
    raw_all_levels.csv          everything returned, untouched
    dhs_nigeria_vaccination_tables.xlsx
        National | Subnational | Background | Gaps | Checks | Indicator_Catalogue
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.dhsprogram.com/rest/dhs"
COUNTRY = "NG"
PER_PAGE = 1000

# friendly name -> candidate indicator IDs (first one found in the live catalogue is used)
CANDIDATES = {
    "DTP1 (pentavalent 1)": ["CH_VACC_C_DP1"],
    "DTP3 (pentavalent 3)": ["CH_VACC_C_DP3"],
    "Measles (MCV1)": ["CH_VACC_C_MSL"],
    "All 8 basic vaccinations": ["CH_VACS_C_BAS", "CH_VACC_C_BAS"],
    "No vaccinations (zero-dose)": ["CH_VACS_C_NON", "CH_VACC_C_NON"],
    "BCG": ["CH_VACC_C_BCG"],  # not independently confirmed; reported if missing
}
LEVELS = ["national", "subnational", "background"]


def get_json(endpoint: str, params: dict, tries: int = 4) -> dict:
    url = f"{BASE}/{endpoint}"
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network, HTTP or JSON error
            last = e
            time.sleep(2 * (k + 1))
    sys.exit(f"Could not reach {url} after {tries} tries: {last}\n"
             "Check your internet connection, or whether your network blocks api.dhsprogram.com.")


def get_all_pages(endpoint: str, params: dict) -> list:
    rows, page = [], 1
    while True:
        p = dict(params, perpage=PER_PAGE, page=page, f="json")
        data = get_json(endpoint, p)
        rows.extend(data.get("Data", []) or [])
        total_pages = int(data.get("TotalPages", 1) or 1)
        if page >= total_pages:
            return rows
        page += 1
        time.sleep(0.3)


def catalogue(api_key):
    params = {"countryIds": COUNTRY}
    if api_key:
        params["apiKey"] = api_key
    rows = get_all_pages("indicators", params)
    cat = pd.DataFrame(rows)
    if cat.empty:
        return cat
    text = (cat.get("Label", "").astype(str) + " " + cat.get("Definition", "").astype(str)).str.lower()
    keep = text.str.contains("vaccin|dpt|measles|bcg|pentavalent|polio", regex=True)
    return cat[keep].reset_index(drop=True)


def resolve_indicators(cat: pd.DataFrame):
    found, missing = {}, []
    ids = set(cat["IndicatorId"]) if not cat.empty and "IndicatorId" in cat else set()
    for name, cands in CANDIDATES.items():
        hit = next((c for c in cands if c in ids), None)
        if hit:
            found[name] = hit
        elif not ids:  # catalogue unavailable: try the first candidate directly
            found[name] = cands[0]
        else:
            missing.append((name, cands))
    return found, missing


def pull(indicator_ids, api_key):
    frames = []
    for level in LEVELS:
        params = {"countryIds": COUNTRY, "indicatorIds": ",".join(indicator_ids), "breakdown": level}
        if api_key:
            params["apiKey"] = api_key
        rows = get_all_pages("data", params)
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"  {level}: no rows returned")
            continue
        df.insert(0, "Level", level)
        frames.append(df)
        print(f"  {level}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tidy(raw: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    g = lambda c: raw[c] if c in raw else pd.Series([None] * len(raw))
    out = pd.DataFrame({
        "Level": raw["Level"],
        "Survey_ID": g("SurveyId"),
        "Survey_Year": pd.to_numeric(g("SurveyYear"), errors="coerce"),
        "Survey_Type": g("SurveyType"),
        "Indicator_ID": g("IndicatorId"),
        "Indicator": g("IndicatorId").map(id_to_name).fillna(g("Indicator")),
        "Category": g("CharacteristicCategory"),
        "Group": g("CharacteristicLabel"),
        "Value_Pct": pd.to_numeric(g("Value"), errors="coerce"),
        "CI_Low": pd.to_numeric(g("CILow"), errors="coerce"),
        "CI_High": pd.to_numeric(g("CIHigh"), errors="coerce"),
        "N_Weighted": pd.to_numeric(g("DenominatorWeighted"), errors="coerce"),
        "N_Unweighted": pd.to_numeric(g("DenominatorUnweighted"), errors="coerce"),
        "Is_Preferred": g("IsPreferred"),
        "Region_ID": g("RegionId"),
        "By_Variable": g("ByVariableLabel"),
    })
    # keep the preferred record where the API flags duplicates
    if out["Is_Preferred"].notna().any():
        pref = pd.to_numeric(out["Is_Preferred"], errors="coerce")
        out = out[(pref == 1) | pref.isna()]
    return out.reset_index(drop=True)


def gaps(bg: pd.DataFrame) -> pd.DataFrame:
    """Inequality summaries per survey, indicator and grouping."""
    rows = []
    for (sid, yr, ind, cat), g in bg.dropna(subset=["Value_Pct"]).groupby(
            ["Survey_ID", "Survey_Year", "Indicator", "Category"]):
        if g["Group"].nunique() < 2:
            continue
        hi, lo = g.loc[g.Value_Pct.idxmax()], g.loc[g.Value_Pct.idxmin()]
        lab = g["Group"].astype(str).str.lower()
        poor = g[lab.str.contains("lowest|poorest")]
        rich = g[lab.str.contains("highest|richest")]
        rows.append({
            "Survey_ID": sid, "Survey_Year": yr, "Indicator": ind, "Category": cat,
            "Highest_Group": hi.Group, "Highest_Pct": hi.Value_Pct,
            "Lowest_Group": lo.Group, "Lowest_Pct": lo.Value_Pct,
            "Absolute_Gap_pp": hi.Value_Pct - lo.Value_Pct,
            "Relative_Ratio": (hi.Value_Pct / lo.Value_Pct) if lo.Value_Pct else None,
            "Richest_minus_Poorest_pp": (rich.Value_Pct.iloc[0] - poor.Value_Pct.iloc[0])
            if len(rich) and len(poor) else None,
        })
    return pd.DataFrame(rows)


def checks(t: pd.DataFrame, found, missing) -> pd.DataFrame:
    rows = [("Indicators used", ", ".join(f"{k} = {v}" for k, v in found.items()))]
    for name, cands in missing:
        rows.append((f"Indicator NOT FOUND in catalogue: {name}", "tried " + ", ".join(cands)))
    rows.append(("Surveys returned", ", ".join(sorted(t["Survey_ID"].dropna().astype(str).unique()))))
    for lvl in LEVELS:
        s = t[t.Level == lvl]
        rows.append((f"Rows: {lvl}", len(s)))
    sub = t[t.Level == "subnational"]
    if len(sub):
        n = sub.groupby("Survey_ID")["Group"].nunique()
        rows.append(("Distinct subnational groups per survey (states/zones)", "; ".join(f"{k}: {v}" for k, v in n.items())))
    bg = t[t.Level == "background"]
    if len(bg):
        rows.append(("Background categories found", "; ".join(sorted(bg["Category"].dropna().astype(str).unique()))))
    rows.append(("Missing Value_Pct", int(t["Value_Pct"].isna().sum())))
    rows.append(("Sanity check", "Compare National DTP3, measles and all-basic values with your workbook's Coverage sheet "
                                 "(DHS rows). They should be close; small differences can come from definitions."))
    return pd.DataFrame(rows, columns=["Check", "Result"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None, help="optional DHS API key")
    ap.add_argument("--out-dir", default="dhs_output")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("1/4 Reading the indicator catalogue for Nigeria ...")
    cat = catalogue(a.api_key)
    cat.to_csv(out / "indicator_catalogue.csv", index=False)
    print(f"    {len(cat)} vaccination-related indicators listed (indicator_catalogue.csv)")
    found, missing = resolve_indicators(cat)
    for k, v in found.items():
        print(f"    using {k}: {v}")
    for name, cands in missing:
        print(f"    NOT FOUND: {name} (tried {', '.join(cands)}). Look in indicator_catalogue.csv for the right ID.")
    if not found:
        sys.exit("No usable indicators. Open indicator_catalogue.csv, pick IDs, and edit CANDIDATES.")

    print("2/4 Downloading data ...")
    raw = pull(list(found.values()), a.api_key)
    if raw.empty:
        sys.exit("The API returned no data. Check the indicator IDs and your connection.")
    raw.to_csv(out / "raw_all_levels.csv", index=False)

    print("3/4 Tidying and computing gaps ...")
    id_to_name = {v: k for k, v in found.items()}
    t = tidy(raw, id_to_name)
    national = t[t.Level == "national"]
    sub = t[t.Level == "subnational"]
    bg = t[t.Level == "background"]
    gp = gaps(bg)
    chk = checks(t, found, missing)

    print("4/4 Writing Excel file ...")
    with pd.ExcelWriter(out / "dhs_nigeria_vaccination_tables.xlsx", engine="openpyxl") as xw:
        national.to_excel(xw, sheet_name="National", index=False)
        sub.to_excel(xw, sheet_name="Subnational", index=False)
        bg.to_excel(xw, sheet_name="Background", index=False)
        gp.to_excel(xw, sheet_name="Gaps", index=False)
        chk.to_excel(xw, sheet_name="Checks", index=False)
        cat.to_excel(xw, sheet_name="Indicator_Catalogue", index=False)
    print("\nDone. Read the Checks sheet first.")
    print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
