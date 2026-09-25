"""Variance-covariance matrix of family earnings growth, BPS (2016) estimation sample.

Family earnings are head + spouse earnings, deflated: `log_toty` in data4estimation.dta
equals log(ly + wly) - log(price) to float precision.

The PSID panel behind the paper has six biennial waves (earnings years 1998, 2000,
2002, 2004, 2006, 2008), so a panel balanced over all six levels yields five
biennial growth rates and a 5x5 second-moment matrix.

Two versions are reported:
  raw       cov of Dlog family earnings (the covariance itself removes wave means)
  residual  cov of the BPS first-stage residual, following the dlog_toty regression
            in residual_measures.do -- residualized on the full sample, then
            restricted to the balanced households, which is the order BPS use.

OLS residuals depend only on the column space of the design, not its
parameterization, so collinear dummy sets are kept and resolved by least squares
rather than hand-dropping reference categories the way Stata's `xi` does.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DTA = ROOT / "data/bps2016/AER_2012_1549_data/output/data4estimation.dta"
OUTDIR = ROOT / "data/bps2016"

WAVES = [98.0, 100.0, 102.0, 104.0, 106.0, 108.0]
GROWTH = WAVES[1:]                       # Dlog dated at its end point
LABELS = [f"D{int(1900 + y)}" for y in GROWTH]   # D2000 .. D2008


# ------------------------------------------------------------------ Stata helpers
def lag2(d, col):
    """l2.col under `tsset person year`, where year advances in steps of two."""
    key = d[["person", "year", col]].rename(columns={col: "_v"})
    key = key.assign(year=key["year"] + 2)
    m = d[["person", "year"]].merge(key, on=["person", "year"], how="left")
    return pd.Series(m["_v"].to_numpy(), index=d.index)


def tabdum(s, prefix):
    """`tab x, gen(prefix)`: one indicator per observed level, missing where x is."""
    cols = {}
    for i, v in enumerate(np.sort(s.dropna().unique()), 1):
        col = (s == v).astype(float)
        col[s.isna()] = np.nan
        cols[f"{prefix}{i}"] = col
    return pd.DataFrame(cols, index=s.index)


def interact(frame, series, name):
    """Products of a 0/1 series with every column of `frame`, propagating missings."""
    out = frame.mul(series, axis=0)
    out.columns = [f"{name}X{c}" for c in frame.columns]
    return out


# ------------------------------------------------------------------ load and build
d = pd.read_stata(DTA, convert_categoricals=False).sort_values(["person", "year"])

_tot = (d["ly"].fillna(0) + d["wly"].fillna(0)).replace(0, np.nan)
d["log_toty_chk"] = np.log(_tot) - np.log(d["price"])
d["dlog_toty"] = d["log_toty"] - lag2(d, "log_toty")

# --- level covariates, exactly as residual_measures.do defines them
d["empl"] = (d["empst"] == 1).astype(float)
d["unempl"] = (d["empst"] == 2).astype(float)
d["retir"] = (d["empst"] == 3).astype(float)
d["w_empl"] = (d["wempst"] == 1).astype(float)
d["w_unempl"] = (d["wempst"] == 2).astype(float)
d["w_retir"] = (d["wempst"] == 3).astype(float)
for src, pre in (("race", ""), ("wrace", "w_")):
    # `gen other = race>=3` is true for missing race in Stata, but the tab-generated
    # race dummies are missing there, so those rows leave e(sample) regardless
    d[f"{pre}white"] = (d[src] == 1).astype(float).mask(d[src].isna())
    d[f"{pre}black"] = (d[src] == 2).astype(float).mask(d[src].isna())
    d[f"{pre}other"] = (d[src] >= 3).astype(float).mask(d[src].isna())
d["kidsout"] = (d["outkid"] == 1).astype(float).mask(d["outkid"].isna())
d["bigcity"] = d["smsa"].isin([1, 2]).astype(float)
d["extra"] = (d["tyoth"] > 0).astype(float)

DIFFED = ["kidsout", "bigcity", "kids", "fsize", "empl", "w_empl",
          "unempl", "w_unempl", "retir", "w_retir", "extra"]
d = pd.concat([d, pd.DataFrame({"d" + v: d[v] - lag2(d, v) for v in DIFFED},
                               index=d.index)], axis=1)

# --- dummy blocks
yrd = tabdum(d["year"], "yrd")
blocks = [
    yrd,
    tabdum(d["yb"], "ybd"), tabdum(d["wyb"], "w_ybd"),
    tabdum(d["educ"], "edd"), tabdum(d["weduc"], "wedd"),
    tabdum(d["fsize"], "fd"), tabdum(d["kids"], "chd"),
    tabdum(d["dfsize"], "dfd"), tabdum(d["dkids"], "dchd"),
    tabdum(d["state"], "stated"),
    d[["white", "w_white", "black", "w_black", "other", "w_other",
       "empl", "unempl", "retir", "w_empl", "w_unempl", "w_retir",
       "kidsout", "bigcity", "extra",
       "dempl", "dunempl", "dretir", "dw_empl", "dw_unempl", "dw_retir",
       "dextra", "dkidsout", "dbigcity"]],
]

# --- the x-year interactions the earnings regression carries
edd, wedd = tabdum(d["educ"], "edd"), tabdum(d["weduc"], "wedd")
for c in edd.columns:
    blocks.append(interact(yrd, edd[c], c))
for c in wedd.columns:
    blocks.append(interact(yrd, wedd[c], c))
for v in ["white", "w_white", "black", "w_black", "other", "w_other",
          "empl", "w_empl", "unempl", "w_unempl", "retir", "w_retir",
          "bigcity", "dbigcity"]:
    blocks.append(interact(yrd, d[v], v))

X = pd.concat(blocks, axis=1)

# ------------------------------------------------------------------ first stage
y = d["dlog_toty"]
insample = y.notna() & X.notna().all(axis=1)

Xs = X.loc[insample].to_numpy(dtype=float)
ys = y.loc[insample].to_numpy(dtype=float)
Xs = np.column_stack([np.ones(len(Xs)), Xs])
beta, _, rank, _ = np.linalg.lstsq(Xs, ys, rcond=None)

resid = pd.Series(np.nan, index=d.index, name="utoty")
resid.loc[insample] = ys - Xs @ beta
d = d.assign(utoty=resid)
r2 = 1 - float(np.nansum(resid ** 2) / ((len(ys) - 1) * ys.var(ddof=1)))

# ------------------------------------------------------------------ balanced panels
wide_raw = d.pivot(index="person", columns="year", values="dlog_toty")[GROWTH]
wide_res = d.pivot(index="person", columns="year", values="utoty")[GROWTH]
wide_raw.columns = wide_res.columns = LABELS

bal_raw = wide_raw.dropna()
bal_res = wide_res.dropna()


def report(name, W):
    V = W.cov()                       # subtracts each wave's own mean
    C = W.corr()
    print(f"\n{'=' * 78}\n{name}   N = {len(W)} households, {W.shape[1]} growth rates\n{'=' * 78}")
    print("\nVariance-covariance\n")
    print(V.to_string(float_format=lambda x: f"{x: .4f}"))
    print("\nCorrelation\n")
    print(C.to_string(float_format=lambda x: f"{x: .3f}"))

    k = W.shape[1]
    v = V.to_numpy()
    print("\nMean by band (lag in waves; one wave = two years)\n")
    print(f"  {'lag':>4}  {'mean':>9}  {'cells':>6}")
    bands = {}
    for lag in range(k):
        vals = np.array([v[i, i + lag] for i in range(k - lag)])
        bands[lag] = vals.mean()
        print(f"  {lag:>4}  {vals.mean():>9.4f}  {len(vals):>6}")

    # textbook permanent + transitory read: Dy_t = zeta_t + e_t - e_{t-1} (waves)
    s_e = -bands[1]
    s_z = bands[0] - 2 * s_e
    print("\nIf Dy = zeta_t + e_t - e_(t-1) in waves, so var = s_zeta + 2 s_e and")
    print("lag-1 cov = -s_e, with longer lags zero:")
    print(f"  s_e (transitory)  = {s_e: .4f}")
    print(f"  s_zeta (permanent)= {s_z: .4f}")
    print(f"  share permanent   = {s_z / (s_z + 2 * s_e): .3f}")
    print(f"  lag>=2 cov, mean  = {np.mean([bands[l] for l in range(2, k)]): .4f}  (0 under that process)")
    return V


print("Family earnings = head + spouse, deflated (log_toty)")
print(f"  max |log_toty - (log(ly+wly) - log(price))| = "
      f"{np.nanmax(np.abs(d['log_toty'] - d['log_toty_chk'])):.2e}")
print(f"\nEstimation sample: {len(d):,} obs, {d['person'].nunique():,} households, 6 waves")
print(f"Dlog_toty non-missing: {int(y.notna().sum()):,} obs")
print(f"First stage: {int(insample.sum()):,} obs, {X.shape[1]:,} regressors, "
      f"rank {rank}, R2 {r2:.3f}")
print(f"Balanced over all 5 growth rates -- raw: {len(bal_raw)} households, "
      f"residual: {len(bal_res)} households")

# how much of the level difference from Table 1 is pooling, how much is selection
dlog_y_all = (d["log_y"] - lag2(d, "log_y")).var(ddof=1)
pooled_all = d["dlog_toty"].var(ddof=1)
pooled_bal = d.loc[d["person"].isin(bal_raw.index), "dlog_toty"].var(ddof=1)
print("\nScale check, pooled over waves (variance of the biennial log change)")
print(f"  head earnings, all households     {dlog_y_all:.4f}   "
      f"(Table 1 prints SD {np.sqrt(round(dlog_y_all, 3)):.3f})")
print(f"  family earnings, all households   {pooled_all:.4f}")
print(f"  family earnings, balanced only    {pooled_bal:.4f}")

V_raw = report("RAW  Dlog family earnings", bal_raw)
V_res = report("RESIDUAL  BPS first stage (utoty)", bal_res)

V_raw.to_csv(OUTDIR / "vcov_family_earnings_growth_raw.csv")
V_res.to_csv(OUTDIR / "vcov_family_earnings_growth_residual.csv")
print(f"\nwritten: {(OUTDIR / 'vcov_family_earnings_growth_raw.csv').relative_to(ROOT)}")
print(f"written: {(OUTDIR / 'vcov_family_earnings_growth_residual.csv').relative_to(ROOT)}")

# ------------------------------------------------------------------ LaTeX output
def latex_matrix(V, path, caption, label, notes):
    """booktabs-only LaTeX for a symmetric matrix. Positives carry \\phantom{-} so the
    decimal points line up in right-aligned columns without siunitx."""
    heads = [f"$\\Delta {c[1:]}$" for c in V.columns]
    ncol = len(heads)

    def cell(x, diag):
        pad = "" if x < 0 else "\\phantom{-}"
        body = f"{pad}{x:.4f}"
        if diag:
            return f"${pad}\\mathbf{{{x:.4f}}}$"
        return f"${body}$"

    # The tabular is measured into scratch box 0 so the notes can be set to the
    # table's own width. Putting the notes in a \multicolumn inside the tabular
    # instead forces the tabular to the notes' width and stretches the last column.
    lines = [
        r"\sbox0{%",
        rf"  \begin{{tabular}}{{l{'r' * ncol}}}",
        r"    \toprule",
        "    " + " & ".join([""] + heads) + r" \\",
        r"    \midrule",
    ]
    for i, row in enumerate(V.index):
        cells = [cell(V.iat[i, j], i == j) for j in range(ncol)]
        lines.append("    " + " & ".join([f"$\\Delta {row[1:]}$"] + cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \usebox0",
        r"  \par\vspace{4pt}",
        r"  \begin{minipage}{\wd0}",
        rf"    \footnotesize {notes}",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    path.write_text(tex)
    return tex


NOTES = (
    r"\textit{Notes:} Family earnings are the sum of head and spouse annual earnings, "
    r"deflated. Entries are covariances of the biennial change in log family earnings, "
    r"dated at the end point of the change, after removing the first-stage projection of "
    r"Blundell, Pistaferri and Saporta-Eksten (2016): log earnings growth on year, cohort, "
    r"education, race, family size, number of children, employment status, state, and the "
    r"education, race, employment and metropolitan interactions with year, entered in both "
    r"levels and differences. The projection is taken on the full estimation sample "
    rf"({{FS_N}} observations, $R^2 = {{FS_R2}}$) and the covariances on the "
    rf"{{BAL_N}} households observed in all six PSID waves, 1999--2009. Diagonal entries "
    r"in bold. Source: replication package for \textit{{Consumption Inequality and Family "
    r"Labor Supply}}, \textit{{American Economic Review}} 106(2)."
)

tex = latex_matrix(
    V_res,
    OUTDIR / "vcov_family_earnings_growth_residual.tex",
    caption="Variance--covariance matrix of family earnings growth",
    label="tab:vcov-family-earnings-growth",
    notes=NOTES.replace("{FS_N}", f"{int(insample.sum()):,}")
               .replace("{FS_R2}", f"{r2:.3f}")
               .replace("{BAL_N}", f"{len(bal_res)}"),
)
print("\n" + "=" * 78 + "\nLaTeX\n" + "=" * 78 + "\n")
print(tex)
print(f"written: {(OUTDIR / 'vcov_family_earnings_growth_residual.tex').relative_to(ROOT)}")
