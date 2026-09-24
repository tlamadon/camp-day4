"""Replicate Table 1, columns (1)-(2) of Blundell, Pistaferri & Saporta-Eksten (2016).

Follows AER_2012_1549_data/descriptive_stats.do, sample 1 (baseline estimation
sample), and compares the result to the numbers printed in the published table.

Columns (3)-(6) of the paper need data4estimation_nopart.dta and data4sample3.dta,
which the archive does not ship, so they cannot be reproduced from this package.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DTA = ROOT / "data/bps2016/AER_2012_1549_data/output/data4estimation.dta"
OUT = ROOT / "data/bps2016/table1_replication.csv"


# ---------------------------------------------------------------- Stata helpers
def rsum(df, cols, missing=False):
    """egen rowtotal. `missing=True` mirrors the `, missing` option: the result is
    missing only when every component is missing; otherwise missings count as 0."""
    return df[cols].sum(axis=1, min_count=1 if missing else 0)


def sround(x, unit=1.0):
    """Stata's round(): ties go away from zero, unlike Python's banker's rounding."""
    if x is None or pd.isna(x):
        return np.nan
    return float(np.sign(x) * np.floor(abs(x) / unit + 0.5) * unit)


def mean(s):
    return np.nan if s.notna().sum() == 0 else s.mean()


def p50(s):
    """summarize, detail p50. Stata averages the two central order statistics when
    N is even and takes the upper one otherwise -- i.e. the ordinary median."""
    return np.nan if s.notna().sum() == 0 else s.median()


def lag2(df, value):
    """l2.var under `tsset person year` where year advances in steps of 2."""
    key = pd.DataFrame({"person": df["person"], "year": df["year"] + 2, "v": df[value]})
    merged = df[["person", "year"]].merge(key, on=["person", "year"], how="left")
    return merged["v"].to_numpy()


# ---------------------------------------------------------------- build the vars
d = pd.read_stata(DTA, convert_categoricals=False)

# Consumption ----------------------------------------------------------------
d["rent_all"] = d["rent"].where(d["rent"].notna(), d["renteq"])
d["health_ser"] = rsum(d, ["nurse", "doctor", "prescription"], missing=True)
d["utilities"] = rsum(d, ["electric", "heating", "water", "miscutils"], missing=True)
d["transport"] = rsum(
    d, ["carins", "carrepair", "parking", "busfare", "taxifare", "othertrans"], missing=True
)
d["educ_ser"] = rsum(d, ["tuition", "otherschool"], missing=True)

CONS = [
    "totcons", "ndcons", "food", "gasoline", "services", "fout", "hinsurance",
    "health_ser", "utilities", "transport", "educ_ser", "childcare",
    "homeinsure", "rent_all",
]
# the do-file zeroes out missings for every consumption row before summarizing
for v in CONS:
    d[v] = d[v].fillna(0.0)

# Assets ---------------------------------------------------------------------
d["assets"] = rsum(
    d, ["cash", "bonds", "stocks", "busval", "penval", "house", "real_estate", "carval"],
    missing=True,
)
d["house_re"] = rsum(d, ["house", "real_estate"], missing=True)
d["assets_ot"] = rsum(
    d, ["cash", "bonds", "stocks", "busval", "penval", "carval"], missing=True
)
d["tot_debt"] = rsum(d, ["other_debt", "mortgage1", "mortgage2"], missing=True)
d["mortgage"] = rsum(d, ["mortgage1", "mortgage2"], missing=True)
# rsum without `missing`: an all-missing row becomes 0, not missing
d["networth"] = rsum(d, ["assets"]) - rsum(d, ["tot_debt"])
d["netfinworth"] = rsum(d, ["cash", "bonds", "stocks", "penval"])

# Earners --------------------------------------------------------------------
d["head_employed"] = 1.0
# `gen ba = educ>=3` in Stata: missing educ is larger than any number, so it is TRUE
d["ba"] = ((d["educ"] >= 3) | d["educ"].isna()).astype(float)
d["wba"] = ((d["weduc"] >= 3) & d["weduc"].notna()).astype(float)

wives = d["wife_employed"] == 1

# Volatility -----------------------------------------------------------------
VOL = ["log_w", "log_ww", "log_y", "log_yw", "log_c"]
for v in VOL:
    d["d" + v] = d[v] - lag2(d, v)


# ---------------------------------------------------------------- the comparison
# published Table 1, columns (1) and (2); None = cell left blank in the paper
PAPER = {
    "Consumption: Total":              ("totcons",      39257,  32920),
    "  Nondurable cons.":              ("ndcons",        8552,   7800),
    "    Food at home":                ("food",          6165,   5200),
    "    Gasoline":                    ("gasoline",      2339,   1800),
    "  Services":                      ("services",     30705,  24280),
    "    Food out":                    ("fout",          2816,   1800),
    "    Health ins.":                 ("hinsurance",    1572,    750),
    "    Health serv.":                ("health_ser",    1317,    650),
    "    Utilities":                   ("utilities",     3856,   3300),
    "    Transportation":              ("transport",     3896,   2040),
    "    Education":                   ("educ_ser",      2589,      0),
    "    Child care":                  ("childcare",      714,      0),
    "    Home ins.":                   ("homeinsure",     581,    480),
    "    Rent (or rent eq.)":          ("rent_all",     13364,   9900),
    "Assets: Total":                   ("assets",      389105, 221000),
    "  Housing and RE":                ("house_re",    232671, 160000),
    "  Financial assets":              ("assets_ot",   156670,  35500),
    "Total debt":                      ("tot_debt",    108984,  78000),
    "  Mortgage":                      ("mortgage",     99083,  70000),
    "  Other debt":                    ("other_debt",   10218,   2000),
    "Total net worth":                 ("networth",    280131, 112000),
    "Total net financial worth":       ("netfinworth",  96576,  13600),
    "Head: Participation rate":        ("head_employed", 1.00,   None),
    "Head: Earnings | work":           ("ly",           67008,  48237),
    "Head: Hours worked | work":       ("hours",         2302,   2226),
    "Head: Share with some college":   ("ba",            0.59,   None),
    "Wife: Participation rate":        ("wife_employed", 0.80,   None),
    "Wife: Earnings | work":           ("wly",          32988,  26600),
    "Wife: Hours worked | work":       ("hourw",         1688,   1864),
    "Wife: Share with some college":   ("wba",           0.60,   None),
}
WIFE_CONDITIONAL = {"wly", "hourw"}
SHARES = {"head_employed", "ba", "wife_employed", "wba"}

# Panel B. The paper's note says these are standard deviations; descriptive_stats.do
# posts r(Var). The published numbers match the SD -- see the diagnostic at the end.
PAPER_VOL = {
    "SD Δ log W1 (head wage)":     ("dlog_w",  0.498),
    "SD Δ log W2 (wife wage)":     ("dlog_ww", 0.457),
    "SD Δ log Y1 (head earnings)": ("dlog_y",  0.513),
    "SD Δ log Y2 (wife earnings)": ("dlog_yw", 0.616),
    "SD Δ log C (consumption)":    ("dlog_c",  0.321),
}

rows = []
for label, (var, pm, pmed) in PAPER.items():
    s = d.loc[wives, var] if var in WIFE_CONDITIONAL else d[var]
    unit = 0.01 if var in SHARES else 1.0
    rows.append(
        {
            "row": label,
            "paper_mean": pm,
            "repl_mean": sround(mean(s), unit),
            "paper_median": pmed,
            "repl_median": None if pmed is None else sround(p50(s), unit),
        }
    )

rows.append({"row": "Observations", "paper_mean": 10479, "repl_mean": len(d),
             "paper_median": None, "repl_median": None})

for label, (var, pv) in PAPER_VOL.items():
    rows.append(
        {
            "row": label,
            "paper_mean": pv,
            "repl_mean": sround(d[var].std(ddof=1), 0.001),
            "paper_median": None,
            "repl_median": None,
        }
    )

t = pd.DataFrame(rows)


def diff(row):
    p, r = row["paper_mean"], row["repl_mean"]
    if p in (None, 0) or pd.isna(r):
        return ""
    return f"{100 * (r - p) / abs(p):+.2f}%"


def mdiff(row):
    p, r = row["paper_median"], row["repl_median"]
    if p is None or r is None or pd.isna(r):
        return ""
    if p == 0:
        return "exact" if r == 0 else "n/a"
    return f"{100 * (r - p) / abs(p):+.2f}%"


t["mean_gap"] = t.apply(diff, axis=1)
t["median_gap"] = t.apply(mdiff, axis=1)

fmt = lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else (
    f"{x:,.2f}" if isinstance(x, float) and abs(x) < 10 else f"{x:,.0f}"
    if isinstance(x, (int, float)) and float(x).is_integer() else f"{x:,.3f}"
)

hdr = f"{'':<32}{'paper':>12}{'replicated':>14}{'gap':>10}   {'paper':>10}{'replicated':>12}{'gap':>10}"
print(f"{'':<32}{'--- mean ---':^36}   {'--- median ---':^32}")
print(hdr)
print("-" * len(hdr))
for _, r in t.iterrows():
    print(
        f"{r['row']:<32}{fmt(r['paper_mean']):>12}{fmt(r['repl_mean']):>14}"
        f"{r['mean_gap']:>10}   {fmt(r['paper_median']):>10}"
        f"{fmt(r['repl_median']):>12}{r['median_gap']:>10}"
    )

t.to_csv(OUT, index=False)
print(f"\nwritten: {OUT.relative_to(ROOT)}")

# Panel B diagnostic: descriptive_stats.do posts r(Var), but the published table
# matches the standard deviation, as the note under Table 1 states.
print("\nPanel B, both readings:")
print(f"{'':<22}{'paper':>10}{'r(Var) in .do':>15}{'sqrt = SD':>12}")
for label, (var, pv) in PAPER_VOL.items():
    v = d[var].var(ddof=1)
    print(f"{label.split('(')[0].strip():<22}{pv:>10.3f}{v:>15.3f}{np.sqrt(v):>12.3f}")
