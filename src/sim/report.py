"""One-page HTML summary of the pilot, built from output/ so no number is retyped.

Writes two files:

* ``output/report.html``   -- standalone, opens in a browser straight from disk
* ``output/artifact.html`` -- the same page as a fragment, for publishing as an
  Artifact (the runtime supplies the doctype/head wrapper)

The charts are inline SVG drawn against CSS custom properties, so the page reads
in light and dark; the PNG/PDF figures under ``output/figures/`` stay the
print-ready versions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import plim, plim_param
from .config import (
    ESTIMATOR_LABELS,
    ESTIMATOR_SHORT_LABELS,
    ESTIMATORS,
    MODELS,
    RHO,
)
from .figures import _bandwidth
from .provenance import git_commit, repo_root, spec_sha256
from .runner import load_cells

#: Categorical slots 1-5 of the validated palette, as CSS variable names.
SERIES_CLASS = {"pooled": "s1", "fd": "s2", "within": "s3", "gmm": "s4", "gmm_me": "s5"}


# ---------------------------------------------------------------- Figure 1
def _fig_mean_by_t(summary: pd.DataFrame, model: str, ts: list[int]) -> str:
    W, H = 430, 300
    ml, mr, mt, mb = 40, 14, 14, 34
    pw, ph = W - ml - mr, H - mt - mb
    lo, hi = -0.40, 1.06
    lx0, lx1 = math.log(ts[0]), math.log(ts[-1])

    def X(t: float) -> float:
        return ml + (math.log(t) - lx0) / (lx1 - lx0) * pw

    def Y(v: float) -> float:
        return mt + (hi - v) / (hi - lo) * ph

    def row(t: int, est: str) -> pd.Series:
        m = (summary.model == model) & (summary["T"] == t) & (summary.estimator == est)
        return summary[m].iloc[0]

    p = [
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
        f'aria-label="Mean rho-hat against T in model {model}">'
    ]
    for g in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p.append(f'<line class="grid" x1="{ml}" x2="{ml+pw:.0f}" y1="{Y(g):.1f}" y2="{Y(g):.1f}"/>')
        p.append(f'<text class="tick ty" x="{ml-7}" y="{Y(g)+3.4:.1f}">{g:.1f}</text>')
    p.append(f'<line class="ref" x1="{ml}" x2="{ml+pw:.0f}" y1="{Y(RHO):.1f}" y2="{Y(RHO):.1f}"/>')
    p.append(f'<text class="reflab" x="{ml+4}" y="{Y(RHO)+13:.1f}">ρ = {RHO}</text>')
    for t in ts:
        p.append(f'<text class="tick" x="{X(t):.1f}" y="{mt+ph+17:.0f}">{t}</text>')
    p.append(f'<line class="axis" x1="{ml}" x2="{ml+pw:.0f}" y1="{mt+ph:.0f}" y2="{mt+ph:.0f}"/>')

    for est in ESTIMATORS:
        c = SERIES_CLASS[est]
        rows = [row(t, est) for t in ts]
        means = " ".join(f"{X(t):.1f},{Y(r.mean_rho):.1f}" for t, r in zip(ts, rows, strict=True))
        plims = " ".join(f"{X(t):.1f},{Y(r.plim):.1f}" for t, r in zip(ts, rows, strict=True))
        p.append(f'<polyline class="plim {c}" points="{plims}"/>')
        p.append(f'<polyline class="line {c}" points="{means}"/>')
        for t, r in zip(ts, rows, strict=True):
            x, y = X(t), Y(r.mean_rho)
            if est == "pooled":
                p.append(f'<circle class="mk {c}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
            elif est == "fd":
                p.append(
                    f'<rect class="mk {c}" x="{x-3.6:.1f}" y="{y-3.6:.1f}" width="7.2" height="7.2"/>'
                )
            elif est == "within":
                p.append(
                    f'<polygon class="mk {c}" points="{x:.1f},{y-4.4:.1f} '
                    f'{x+4:.1f},{y+3:.1f} {x-4:.1f},{y+3:.1f}"/>'
                )
            elif est == "gmm":
                # A diamond, so the GMM reads apart from the pooled circle and the
                # ME fit it sits on in M0, where all three are consistent.
                p.append(
                    f'<polygon class="mk {c}" points="{x:.1f},{y-4.6:.1f} '
                    f'{x+4.6:.1f},{y:.1f} {x:.1f},{y+4.6:.1f} {x-4.6:.1f},{y:.1f}"/>'
                )
            else:
                p.append(
                    f'<polygon class="mk {c}" points="{x:.1f},{y+4.4:.1f} '
                    f'{x+4:.1f},{y-3:.1f} {x-4:.1f},{y-3:.1f}"/>'
                )
    p.append(f'<text class="axlab" x="{ml+pw/2:.0f}" y="{H-3}">panel length T</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- Figure 2
def _fig_densities(draws: pd.DataFrame, T: int = 10, N: int = 500) -> str:
    W = 880
    rowh, mt, mb, ml, mr = 150, 34, 36, 18, 18
    rows = len(MODELS)
    H = mt + rows * rowh + mb
    pw = W - ml - mr

    block = draws[(draws["T"] == T) & (draws["N"] == N)]
    lo = float(block.rho_hat.min()) - 0.06
    hi = max(float(block.rho_hat.max()), RHO) + 0.06

    def X(v: float) -> float:
        return ml + (v - lo) / (hi - lo) * pw

    p = [
        f'<svg viewBox="0 0 {W} {H}" class="chart wide" role="img" '
        f'aria-label="Distribution of rho-hat at T = {T}, one row per model">'
    ]
    for gv in np.arange(-0.2, 1.01, 0.2):
        if lo < gv < hi:
            p.append(
                f'<line class="grid" x1="{X(gv):.1f}" x2="{X(gv):.1f}" '
                f'y1="{mt-8}" y2="{mt+rows*rowh:.0f}"/>'
            )
            p.append(f'<text class="tick" x="{X(gv):.1f}" y="{mt+rows*rowh+17:.0f}">{gv:.1f}</text>')
    p.append(f'<line class="ref" x1="{X(RHO):.1f}" x2="{X(RHO):.1f}" y1="{mt-12}" y2="{mt+rows*rowh:.0f}"/>')
    p.append(f'<text class="reflab" x="{X(RHO)+5:.1f}" y="{mt-15}">true ρ = {RHO}</text>')

    def peak(model: str, est: str) -> float:
        return float(block[(block.model == model) & (block.estimator == est)].rho_hat.mean())

    captions = {
        "M0": "M0 · no individual effects",
        "M1": "M1 · fixed effects",
        "M2": "M2 · fixed effects + measurement error",
    }
    for i, model in enumerate(MODELS):
        base = mt + (i + 1) * rowh
        top = base - rowh + 58
        p.append(f'<text class="rowlab" x="{ml}" y="{top-26:.0f}">{captions[model]}</text>')
        p.append(f'<line class="axis" x1="{ml}" x2="{ml+pw:.0f}" y1="{base:.1f}" y2="{base:.1f}"/>')

        for est in ESTIMATORS:
            c = SERIES_CLASS[est]
            s = block[(block.model == model) & (block.estimator == est)].rho_hat.to_numpy()
            h = _bandwidth(s)
            grid = np.linspace(s.min() - 4 * h, s.max() + 4 * h, 240)
            d = np.exp(-0.5 * ((grid[:, None] - s[None, :]) / h) ** 2).sum(1)
            d = d / d.max()
            pts = " ".join(
                f"{X(g):.1f},{base - v*(rowh-34):.1f}" for g, v in zip(grid, d, strict=True)
            )
            p.append(
                f'<polygon class="dens {c}" points="{X(grid[0]):.1f},{base:.1f} '
                f'{pts} {X(grid[-1]):.1f},{base:.1f}"/>'
            )
            p.append(f'<polyline class="densline {c}" points="{pts}"/>')
            pv = plim(est, model, T)
            p.append(
                f'<line class="plimv {c}" x1="{X(pv):.1f}" x2="{X(pv):.1f}" '
                f'y1="{base:.1f}" y2="{top-4:.1f}"/>'
            )
            for v in s:
                p.append(
                    f'<line class="rug {c}" x1="{X(v):.1f}" x2="{X(v):.1f}" '
                    f'y1="{base+1:.1f}" y2="{base+7:.1f}"/>'
                )
            lx = min(max(X(float(s.mean())), ml + 42), ml + pw - 42)
            # One line up per label already sitting within a label's width: in M0
            # three estimators are consistent and their peaks coincide exactly.
            level = sum(
                1
                for other in ESTIMATORS
                if other != est and 0 < float(s.mean()) - peak(model, other) < 0.12 * (hi - lo)
            )
            p.append(
                f'<text class="peak" x="{lx:.1f}" y="{top - 8 - 14 * level:.1f}">'
                f"{ESTIMATOR_SHORT_LABELS[est]}</text>"
            )
    p.append(f'<text class="axlab" x="{ml+pw/2:.0f}" y="{H-4}">ρ̂</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- tables
def _cell(summary: pd.DataFrame, model: str, T: int, est: str) -> pd.Series:
    m = (summary.model == model) & (summary["T"] == T) & (summary.estimator == est)
    return summary[m].iloc[0]


def _main_table(summary: pd.DataFrame, ts: list[int]) -> str:
    head = "".join(f"<th>T = {t}</th>" for t in ts)
    rows = []
    for est in ESTIMATORS:
        for model in MODELS:
            rs = [_cell(summary, model, t, est) for t in ts]
            sim = "".join(
                f'<td><span class="num">{r.mean_rho:.3f}</span>'
                f'<span class="sd">({r.sd:.3f})</span></td>'
                for r in rs
            )
            pl = "".join(f'<td><span class="num ghost">{r.plim:.3f}</span></td>' for r in rs)
            rows.append(
                f'<tr class="grp"><th rowspan="2" scope="rowgroup" class="stub">'
                f'<span class="dot {SERIES_CLASS[est]}"></span>{ESTIMATOR_LABELS[est]}'
                f'<span class="mdl">{model}</span></th>'
                f'<th class="sub" scope="row">simulated</th>{sim}</tr>'
                f'<tr><th class="sub ghost" scope="row">plim</th>{pl}</tr>'
            )
    return (
        '<div class="scroll"><table class="main"><thead><tr>'
        f'<th class="stub"></th><th></th>{head}</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _coverage_table(summary: pd.DataFrame, ts: list[int]) -> str:
    head = "".join(f"<th>T = {t}</th>" for t in ts)
    rows = []
    for est in ESTIMATORS:
        for model in MODELS:
            tds = ""
            for t in ts:
                c = _cell(summary, model, t, est).coverage
                klass = "num" if c > 0.5 else "num ghost"
                tds += f'<td><span class="{klass}">{c:.2f}</span></td>'
            rows.append(
                f'<tr><th scope="row" class="stub"><span class="dot {SERIES_CLASS[est]}"></span>'
                f'{ESTIMATOR_LABELS[est]}<span class="mdl">{model}</span></th>{tds}</tr>'
            )
    return (
        f'<div class="scroll"><table class="cov"><thead><tr><th class="stub"></th>{head}'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


#: How each variance parameter is written in the page.
PARAMETER_LABELS = {"sigma_eps": "σ<sub>ε</sub>", "sigma_nu": "σ<sub>ν</sub>"}


def _variance_table(summary: pd.DataFrame, ts: list[int], parameter: str) -> str:
    """Deliverable 5: what the growth fits recover besides rho."""
    head = "".join(f"<th>T = {t}</th>" for t in ts)
    rows = []
    for est in ESTIMATORS:
        for model in MODELS:
            cells = [_cell(summary, model, t, est) for t in ts]
            if pd.isna(getattr(cells[0], f"mean_{parameter}")):
                continue
            means = "".join(
                f'<td><span class="num">{getattr(r, f"mean_{parameter}"):.3f}</span>'
                f'<span class="sd">({getattr(r, f"{parameter}_sd"):.3f})</span></td>'
                for r in cells
            )
            cover = "".join(
                f'<td><span class="num ghost">{_maybe(getattr(r, f"{parameter}_coverage"))}'
                "</span></td>"
                for r in cells
            )
            rows.append(
                f'<tr class="grp"><th rowspan="2" scope="rowgroup" class="stub">'
                f'<span class="dot {SERIES_CLASS[est]}"></span>{ESTIMATOR_LABELS[est]}'
                f'<span class="mdl">{model}</span></th>'
                f'<th class="sub" scope="row">mean (SD)</th>{means}</tr>'
                f'<tr><th class="sub ghost" scope="row">coverage</th>{cover}</tr>'
            )
    return (
        '<div class="scroll"><table class="main"><thead><tr>'
        f'<th class="stub"></th><th></th>{head}</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _maybe(value: float) -> str:
    """A coverage that is not defined -- a variance pinned at zero -- prints as a dash."""
    return "&ndash;" if pd.isna(value) else f"{value:.2f}"


# ---------------------------------------------------------------- page
#: The page body.  Doubled braces are literal CSS; single braces are fields.
PAGE = """<title>Panel AR(1) Pilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
:root {{
  --ground:#f6f7f9; --surface:#ffffff; --ink:#12151c; --ink-2:#4e5766; --ink-3:#8b93a2;
  --rule:#e1e5ec; --rule-strong:#c6cdd8; --accent:#1c5cab; --band:#eef1f5;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#101318; --surface:#171b21; --ink:#edf0f4; --ink-2:#a5aebc; --ink-3:#6c7686;
    --rule:#262c35; --rule-strong:#3a424f; --accent:#6da7ec; --band:#1c212a;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#101318; --surface:#171b21; --ink:#edf0f4; --ink-2:#a5aebc; --ink-3:#6c7686;
  --rule:#262c35; --rule-strong:#3a424f; --accent:#6da7ec; --band:#1c212a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
}}

body {{ background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1000px; margin:0 auto; padding-inline:22px; padding-block:44px 64px; }}
.prose {{ max-width:66ch; }}
h1 {{ font-family:var(--serif); font-size:clamp(30px,5.2vw,42px); line-height:1.12; margin:0;
  letter-spacing:-.012em; text-wrap:balance; }}
h2 {{ font-family:var(--serif); font-size:23px; line-height:1.25; margin:0; letter-spacing:-.006em; }}
h3 {{ font-size:14px; margin:0; font-weight:600; }}
p {{ margin:0; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--accent); margin:0 0 14px; }}
.deck {{ font-size:17px; color:var(--ink-2); margin-top:16px; max-width:60ch; text-wrap:pretty; }}

/* header meta ------------------------------------------------------------ */
.meta {{ margin-top:26px; display:flex; flex-wrap:wrap; gap:6px 20px;
  font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); padding-block:11px; }}
.meta b {{ color:var(--ink-2); font-weight:500; }}

/* findings --------------------------------------------------------------- */
.findings {{ display:grid; grid-template-columns:repeat(5,1fr); gap:0; margin-top:42px;
  border-top:2px solid var(--ink); }}
.finding {{ padding:18px 14px 20px 0; border-right:1px solid var(--rule); }}
.finding:last-child {{ border-right:0; }}
.finding + .finding {{ padding-left:14px; }}
.finding .lede {{ display:flex; align-items:center; gap:8px; }}
.finding .val {{ font-family:var(--mono); font-size:22px; font-weight:500; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; margin-top:10px; display:block; }}
.finding .cap {{ color:var(--ink-2); font-size:12.5px; margin-top:8px; text-wrap:pretty; }}
.dot {{ width:9px; height:9px; border-radius:2px; display:inline-block; flex:none; }}
.dot.s1 {{ background:var(--s1); }} .dot.s2 {{ background:var(--s2); }}
.dot.s3 {{ background:var(--s3); }} .dot.s4 {{ background:var(--s4); }}
.dot.s5 {{ background:var(--s5); }}

/* sections --------------------------------------------------------------- */
section {{ margin-top:54px; }}
.shead {{ display:flex; align-items:baseline; gap:12px; border-bottom:1px solid var(--rule-strong);
  padding-bottom:9px; margin-bottom:8px; }}
.snum {{ font-family:var(--mono); font-size:11.5px; color:var(--ink-3); letter-spacing:.06em; }}
.note {{ color:var(--ink-2); font-size:13.5px; margin-top:12px; max-width:68ch; text-wrap:pretty; }}

/* tables ----------------------------------------------------------------- */
.scroll {{ overflow-x:auto; margin-top:20px; }}
table {{ border-collapse:collapse; width:100%; min-width:640px; }}
th, td {{ text-align:right; padding:7px 10px; white-space:nowrap; }}
thead th {{ font-family:var(--mono); font-size:11px; font-weight:500; color:var(--ink-3);
  letter-spacing:.05em; border-bottom:1px solid var(--rule-strong); padding-bottom:8px; }}
th.stub {{ text-align:left; padding-left:0; font-weight:500; font-size:13.5px; color:var(--ink);
  font-family:var(--sans); letter-spacing:0; }}
th.stub .dot {{ margin-right:8px; vertical-align:baseline; }}
.mdl {{ font-family:var(--mono); font-size:11px; color:var(--ink-3); margin-left:9px; }}
th.sub {{ text-align:left; font-family:var(--mono); font-size:10.5px; font-weight:400;
  color:var(--ink-3); letter-spacing:.05em; }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:13px; }}
.sd {{ font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:11px;
  color:var(--ink-3); margin-left:5px; }}
.ghost {{ color:var(--ink-3); }}
tr.grp th, tr.grp td {{ padding-top:12px; border-top:1px solid var(--rule); }}
tr.grp th.sub, tr.grp td {{ border-top:1px solid var(--rule); }}
table.cov tbody tr {{ border-top:1px solid var(--rule); }}
table.cov td {{ padding-block:9px; }}

/* charts ----------------------------------------------------------------- */
.panels {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:8px; }}
.panel {{ min-width:0; }}
.ptitle {{ font-family:var(--mono); font-size:11.5px; color:var(--ink-2); letter-spacing:.05em;
  padding-bottom:4px; border-bottom:1px solid var(--rule); margin-bottom:4px; }}
.chart {{ width:100%; height:auto; display:block; overflow:visible; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:16px;
  font-size:12.5px; color:var(--ink-2); }}
.legend span {{ display:inline-flex; align-items:center; gap:8px; }}
.figcap {{ font-size:12.5px; color:var(--ink-3); margin-top:14px; max-width:70ch; text-wrap:pretty; }}

svg text {{ font-family:var(--mono); }}
.tick {{ font-size:9.5px; fill:var(--ink-3); text-anchor:middle; }}
.ty {{ text-anchor:end; }}
.axlab {{ font-size:10px; fill:var(--ink-3); text-anchor:middle; letter-spacing:.06em; }}
.reflab {{ font-size:9.5px; fill:var(--ink-3); }}
.rowlab {{ font-size:11px; fill:var(--ink-2); letter-spacing:.07em; }}
.peak {{ font-size:10.5px; fill:var(--ink-2); text-anchor:middle; }}
.grid {{ stroke:var(--rule); stroke-width:1; fill:none; }}
.axis {{ stroke:var(--rule-strong); stroke-width:1; fill:none; }}
.ref {{ stroke:var(--ink-3); stroke-width:1; fill:none; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.plim {{ fill:none; stroke-width:1.4; stroke-dasharray:4 3; opacity:.55; }}
.mk {{ stroke:var(--surface); stroke-width:1.3; }}
.dens {{ opacity:.17; stroke:none; }}
.densline {{ fill:none; stroke-width:2; stroke-linejoin:round; }}
.plimv {{ fill:none; stroke-width:1.3; stroke-dasharray:4 3; opacity:.55; }}
.rug {{ fill:none; stroke-width:1.6; }}
.s1 {{ stroke:var(--s1); }} .s2 {{ stroke:var(--s2); }} .s3 {{ stroke:var(--s3); }}
.s4 {{ stroke:var(--s4); }} .s5 {{ stroke:var(--s5); }}
.mk.s1, .dens.s1 {{ fill:var(--s1); }} .mk.s2, .dens.s2 {{ fill:var(--s2); }}
.mk.s3, .dens.s3 {{ fill:var(--s3); }} .mk.s4, .dens.s4 {{ fill:var(--s4); }}
.mk.s5, .dens.s5 {{ fill:var(--s5); }}

/* footer ----------------------------------------------------------------- */
.cols {{ display:grid; grid-template-columns:repeat(2,1fr); gap:30px 40px; margin-top:22px; }}
.cols h3 {{ padding-bottom:7px; border-bottom:1px solid var(--rule); margin-bottom:9px; }}
.cols p {{ font-size:13.5px; color:var(--ink-2); text-wrap:pretty; }}
.run {{ font-family:var(--mono); font-size:12px; font-variant-numeric:tabular-nums;
  width:100%; border-collapse:collapse; min-width:0; }}
.run td {{ padding:5px 0; border-top:1px solid var(--rule); text-align:right; }}
.run td:first-child {{ text-align:left; color:var(--ink-2); font-size:11.5px; }}
footer {{ margin-top:52px; padding-top:16px; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:11px; color:var(--ink-3); }}

@media (max-width:1040px) {{
  .findings {{ grid-template-columns:1fr 1fr 1fr; }}
  .finding {{ border-bottom:1px solid var(--rule); }}
  .panels {{ grid-template-columns:1fr 1fr; }}
}}
@media (max-width:720px) {{
  .findings {{ grid-template-columns:1fr; }}
  .finding {{ border-right:0; border-bottom:1px solid var(--rule); padding:16px 0; }}
  .finding + .finding {{ padding-left:0; }}
  .finding:last-child {{ border-bottom:0; }}
  .panels, .cols {{ grid-template-columns:1fr; }}
}}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Monte&nbsp;Carlo pilot &middot; R&nbsp;=&nbsp;10</p>
    <h1>Panel AR(1): five estimators, three DGPs</h1>
    <p class="deck">Pooled OLS is fine without individual effects and badly biased with them.
      OLS in first differences is inconsistent either way. The within estimator carries the
      Nickell bias, shrinking at rate 1/T. GMM on the whole covariance matrix of growth
      recovers &rho; and the variance parameters &mdash; but only if it fits the right
      matrix: add measurement error to the DGP and the AR(1) version of it collapses to
      {p_gmm_m2}, while the version that allows for the error holds 0.700. Every one of the
      75 cells lands on its analytical plim; the largest gap is {max_gap}.</p>
    <div class="meta">
      <span><b>&rho;</b> 0.7</span>
      <span><b>&sigma;<sub>&epsilon;</sub></b> 0.3</span>
      <span><b>&sigma;<sub>&nu;</sub></b> 0.2 in M2, 0 elsewhere</span>
      <span><b>&sigma;<sub>&alpha;</sub></b> 0 / 0.5 / 5&frasl;3</span>
      <span><b>N</b> 500</span>
      <span><b>R</b> 10</span>
      <span><b>T</b> 3, 5, 10, 20, 50</span>
      <span><b>commit</b> {commit}</span>
      <span><b>SPEC.md</b> {spec}</span>
    </div>
  </header>

  <div class="findings">
    <div class="finding">
      <span class="lede"><span class="dot s1"></span><h3>Pooled OLS</h3></span>
      <span class="val">{p_m1}</span>
      <p class="cap">Its plim with fixed effects. The lag carries &alpha;<sub>i</sub>, and
        between-individual variance outweighs the transitory part 16&nbsp;to&nbsp;1. Without
        effects it is exactly 0.7.</p>
    </div>
    <div class="finding">
      <span class="lede"><span class="dot s2"></span><h3>First differences</h3></span>
      <span class="val">{p_fd}</span>
      <p class="cap">(&rho;&minus;1)/2 in M0 and M1: &Delta;y<sub>i,t&minus;1</sub> and
        &Delta;&epsilon;<sub>it</sub> share &epsilon;<sub>i,t&minus;1</sub>. Measurement
        error doubles it, to {p_fd_m2} in M2.</p>
    </div>
    <div class="finding">
      <span class="lede"><span class="dot s3"></span><h3>Within</h3></span>
      <span class="val">{w3} &rarr; {w50}</span>
      <p class="cap">Nickell bias from T&nbsp;=&nbsp;3 to T&nbsp;=&nbsp;50, identical in M0 and
        M1. In M2 attenuation lands on top of it: {w50_m2} at T&nbsp;=&nbsp;50.</p>
    </div>
    <div class="finding">
      <span class="lede"><span class="dot s4"></span><h3>Growth GMM</h3></span>
      <span class="val">{p_gmm}</span>
      <p class="cap">Its plim in M0 and M1, at every T: &Omega; holds no
        &alpha;<sub>i</sub>, and fitting all of it rather than one ratio pins &rho; and
        &sigma;<sub>&epsilon;</sub>. In M2 it fits the wrong &Omega; and lands on
        {p_gmm_m2}.</p>
    </div>
    <div class="finding">
      <span class="lede"><span class="dot s5"></span><h3>Growth GMM + ME</h3></span>
      <span class="val">{p_gmm_me}</span>
      <p class="cap">Its plim in all three models. One more parameter &mdash; &Omega; gains
        &sigma;<sub>&nu;</sub>&sup2;B, the MA(1) that differencing white noise leaves &mdash;
        and it costs precision where there is no error to find.</p>
    </div>
  </div>

  <section>
    <div class="shead"><span class="snum">01</span><h2>Main table</h2></div>
    <p class="note">Mean &rho;&#770; across the 10 replications, with the standard deviation of
      the draws in parentheses, over the analytical plim. True &rho; = 0.7.</p>
    {main}
  </section>

  <section>
    <div class="shead"><span class="snum">02</span><h2>Mean &rho;&#770; against T</h2></div>
    <p class="note">Solid: the simulated mean. Dashed: the plim. The grey line marks the truth.
      Only the within estimator moves with T. Where two estimators are consistent in the same
      model their lines coincide on 0.7 and the later one covers the earlier.</p>
    <div class="panels">
      <div class="panel"><p class="ptitle">M0 &middot; no individual effects</p>{fig1_m0}</div>
      <div class="panel"><p class="ptitle">M1 &middot; fixed effects</p>{fig1_m1}</div>
      <div class="panel"><p class="ptitle">M2 &middot; + measurement error</p>{fig1_m2}</div>
    </div>
    <div class="legend">
      <span><span class="dot s1"></span>Pooled OLS</span>
      <span><span class="dot s2"></span>First-difference OLS</span>
      <span><span class="dot s3"></span>Within (FE)</span>
      <span><span class="dot s4"></span>Growth-covariance GMM</span>
      <span><span class="dot s5"></span>Growth-covariance GMM, with ME</span>
    </div>
    <p class="figcap">Monte Carlo error is too small to see: the largest standard error of a
      cell mean is {mc}. The two panels share one vertical scale.</p>
  </section>

  <section>
    <div class="shead"><span class="snum">03</span><h2>Where the draws land at T = 10</h2></div>
    <p class="note">Each curve is a Gaussian kernel density over the 10 replications, scaled to
      its own peak; the ticks below the axis are the replications themselves. The biased
      estimators are tightly centred on the wrong value. Three curves sit on 0.7 in M0, where
      pooled OLS and both growth fits are consistent and land on top of each other; in M2 only
      the fit that allows for measurement error is still there.</p>
    {fig2}
    <p class="figcap">Dashed verticals: the analytical plim of each cell.</p>
  </section>

  <section>
    <div class="shead"><span class="snum">04</span><h2>Coverage</h2></div>
    <p class="note">Share of the 10 confidence intervals &rho;&#770;&nbsp;&plusmn;&nbsp;1.96&nbsp;&times;
      clustered SE that contain 0.7.</p>
    {cov}
    <p class="figcap">The consistent cells cover: pooled OLS in M0 {cov_m0}, the AR(1) growth
      GMM {cov_gmm} across M0 and M1, the measurement-error version {cov_gmm_me} across all
      three. Every inconsistent cell covers nothing at any T. An interval centred on the wrong
      value cannot cover, and tighter standard errors only make it worse &mdash; which is the
      point of reporting coverage next to bias. The standard errors themselves are sound: the
      median ratio of mean clustered SE to the actual dispersion of &rho;&#770; is {se_ratio}
      across the 75 cells, noisy because a standard deviation from 10 draws is.</p>
  </section>

  <section>
    <div class="shead"><span class="snum">05</span><h2>The variances, for free</h2></div>
    <p class="note">&Omega; is linear in the variances and the growth fits use all of it, so the
      scales come out of the same criterion as &rho;. What does not come out is
      &sigma;<sub>&alpha;</sub>: growth cannot see a level, which is exactly why neither fit
      cares whether the level is there.</p>
    {sigma_eps}
    <p class="figcap">&sigma;<sub>&epsilon;</sub>, the innovation to the persistent component;
      true value 0.3 in every model. Under M2 the AR(1) fit has no &sigma;<sub>&nu;</sub> to
      put the noise in, so it loads it onto &sigma;<sub>&epsilon;</sub> instead and settles at
      {p_sigma_m2}.</p>
    {sigma_nu}
    <p class="figcap">&sigma;<sub>&nu;</sub>, the measurement error; true value 0.2 in M2 and 0
      in M0 and M1. Zero is the boundary of the parameter space, so in those two models the fit
      is pinned there about half the time; a pinned replication has no usable interval and drops
      out of the coverage, which is what a dash means.</p>
  </section>

  <section>
    <div class="shead"><span class="snum">&mdash;</span><h2>Notes</h2></div>
    <div class="cols">
      <div>
        <h3>Design</h3>
        <p>Every estimator sees the same panel within a replication, so the gaps between them
          are not Monte Carlo noise. y<sub>i0</sub> is drawn from the stationary distribution.
          Seeds derive from (model, T, N, r) and a master seed, so any single cell reruns on
          its own.</p>
      </div>
      <div>
        <h3>Standard errors</h3>
        <p>Clustered by individual throughout: closed-form sandwiches around
          &Sigma;xy&nbsp;/&nbsp;&Sigma;x&sup2; for the three regressions, and the GMM sandwich
          for the fourth, where one individual contributes one moment vector and the clustering
          is automatic. The spec does not pin a finite-sample factor; this uses G/(G&minus;1),
          which keeps the SE-to-dispersion diagnostic readable at T&nbsp;=&nbsp;3.</p>
      </div>
      <div>
        <h3>Cost</h3>
        <table class="run">
          <tr><td>pilot grid, R = 10</td><td>3.3 s</td></tr>
          <tr><td>R = 1,000, N = 500</td><td>19.1 s</td></tr>
          <tr><td>R = 1,000, N = 5,000</td><td>71.5 s</td></tr>
        </table>
        <p style="margin-top:10px">Wall clock with <span class="num">make -j8</span> on the
          laptop, best of three. Adding the GMM multiplied every row by two to three: it is
          the only estimator that searches. The full 1,000-replication grid still does not
          need the cluster.</p>
      </div>
      <div>
        <h3>What would change the story</h3>
        <p>Adding the levels moment Var(y<sub>it</sub>) to the growth moments would identify
          &sigma;<sub>&alpha;</sub> too. Starting from y<sub>i0</sub>&nbsp;=&nbsp;0 instead of
          the stationary draw would move pooled OLS and within, and would cost both growth fits
          their exact moment conditions. Raising &rho; toward 0.95 would deepen the Nickell
          bias at every T, and raising &sigma;<sub>&nu;</sub> would deepen the attenuation on
          top of it.</p>
      </div>
    </div>
  </section>

  <footer>Built from output/summary.csv at commit {commit}; SPEC.md {spec}.
    All 75 cells verified against the plims in the spec's benchmark tables.</footer>
</div>
"""

#: What the Artifact runtime wraps a published page in, reproduced so the local
#: file renders exactly like the published one.
SKELETON = (
    '<!doctype html>\n<html lang="en">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '<style>:root{color-scheme:light}body{margin:0;font-family:system-ui,-apple-system,'
    '"Segoe UI",sans-serif;font-size:14px;background:#fcfcfb}'
    'img{max-width:100%}[hidden]{display:none!important}</style>\n'
)


def render(output_dir: Path) -> str:
    """Build the page body from summary.csv and the raw draws."""
    output_dir = Path(output_dir)
    summary = pd.read_csv(output_dir / "summary.csv")
    draws = load_cells(output_dir / "pilot")
    ts = sorted(summary["T"].unique())
    root = repo_root()

    fields = dict(
        fig1_m0=_fig_mean_by_t(summary, "M0", ts),
        fig1_m1=_fig_mean_by_t(summary, "M1", ts),
        fig1_m2=_fig_mean_by_t(summary, "M2", ts),
        fig2=_fig_densities(draws),
        main=_main_table(summary, ts),
        cov=_coverage_table(summary, ts),
        sigma_eps=_variance_table(summary, ts, "sigma_eps"),
        sigma_nu=_variance_table(summary, ts, "sigma_nu"),
        commit=git_commit(root)[:10],
        spec=spec_sha256(root / "SPEC.md")[:10],
        se_ratio=f"{summary.se_over_sd.median():.2f}",
        cov_m0=f"{summary[(summary.estimator == 'pooled') & (summary.model == 'M0')].coverage.mean():.2f}",
        cov_gmm=f"{summary[(summary.estimator == 'gmm') & (summary.model != 'M2')].coverage.mean():.2f}",
        cov_gmm_me=f"{summary[summary.estimator == 'gmm_me'].coverage.mean():.2f}",
        max_gap=f"{(summary.mean_rho - summary.plim).abs().max():.3f}",
        mc=f"{summary.mc_se.max():.3f}",
        w3=f"{plim('within', 'M0', ts[0]):.3f}",
        w50=f"{plim('within', 'M0', ts[-1]):.3f}",
        p_m1=f"{plim('pooled', 'M1', ts[0]):.3f}",
        p_fd=f"{plim('fd', 'M0', ts[0]):.3f}",
        p_gmm=f"{plim('gmm', 'M1', ts[0]):.3f}",
        p_gmm_m2=f"{plim('gmm', 'M2', ts[-1]):.3f}",
        p_gmm_me=f"{plim('gmm_me', 'M2', ts[0]):.3f}",
        p_fd_m2=f"{plim('fd', 'M2', ts[0]):.3f}",
        w50_m2=f"{plim('within', 'M2', ts[-1]):.3f}",
        p_sigma_m2=f"{plim_param('sigma_eps', 'gmm', 'M2', ts[-1]):.3f}",
    )
    return PAGE.format(**fields)


def build(output_dir: Path) -> list[Path]:
    """Write the standalone report and the Artifact fragment."""
    output_dir = Path(output_dir)
    body = render(output_dir)

    fragment = output_dir / "artifact.html"
    fragment.write_text(body)

    standalone = output_dir / "report.html"
    standalone.write_text(SKELETON + body + "\n</head>\n</html>\n")

    for path in (standalone, fragment):
        print(f"[report] -> {path} ({path.stat().st_size / 1024:.0f} KB)")
    return [standalone, fragment]
