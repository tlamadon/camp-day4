"""Collapse the raw replication draws into the summary statistics of SPEC.md.

Reads every parquet under the output folder, so the pilot and a later HPC run
share one pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import plim, plim_sigma_eps
from .config import ESTIMATORS, MODELS, RHO, SIGMA_EPS, Z95
from .runner import load_cells

GROUP = ["model", "T", "N", "estimator"]

METRIC_COLUMNS = [
    *GROUP,
    "R",
    "mean_rho",
    "plim",
    "mean_bias",
    "mc_se",
    "plim_bias",
    "sd",
    "rmse",
    "coverage",
    "mean_se",
    "se_over_sd",
    # The same five statistics for sigma_eps; NaN unless the estimator returns one.
    "mean_sigma_eps",
    "sigma_eps_plim",
    "sigma_eps_bias",
    "sigma_eps_mc_se",
    "sigma_eps_sd",
    "sigma_eps_rmse",
    "sigma_eps_coverage",
    "sigma_eps_mean_se",
    "sigma_eps_se_over_sd",
]


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _sigma_eps_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    """The same five statistics for sigma_eps, over the cells that report one."""
    block = draws[draws["sigma_eps_hat"].notna()].copy()
    if block.empty:
        return pd.DataFrame(columns=GROUP)

    block["error"] = block["sigma_eps_hat"] - SIGMA_EPS
    block["covered"] = (block["sigma_eps_se"] * Z95 >= block["error"].abs()).astype(float)

    out = (
        block.groupby(GROUP, observed=True)
        .agg(
            reps=("sigma_eps_hat", "size"),
            mean_sigma_eps=("sigma_eps_hat", "mean"),
            sigma_eps_sd=("sigma_eps_hat", lambda s: s.std(ddof=1)),
            sigma_eps_rmse=("error", _rmse),
            sigma_eps_coverage=("covered", "mean"),
            sigma_eps_mean_se=("sigma_eps_se", "mean"),
        )
        .reset_index()
    )
    out["sigma_eps_plim"] = [plim_sigma_eps(e) for e in out["estimator"]]
    out["sigma_eps_bias"] = out["mean_sigma_eps"] - SIGMA_EPS
    out["sigma_eps_mc_se"] = out["sigma_eps_sd"] / np.sqrt(out["reps"])
    out["sigma_eps_se_over_sd"] = out["sigma_eps_mean_se"] / out["sigma_eps_sd"]
    return out.drop(columns="reps")


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, T, N, estimator) cell."""
    draws = draws.copy()
    draws["error"] = draws["rho_hat"] - RHO
    draws["covered"] = (draws["se"] * Z95 >= draws["error"].abs()).astype(float)

    grouped = draws.groupby(GROUP, observed=True)
    out = grouped.agg(
        R=("rho_hat", "size"),
        mean_rho=("rho_hat", "mean"),
        sd=("rho_hat", lambda s: s.std(ddof=1)),
        rmse=("error", _rmse),
        coverage=("covered", "mean"),
        mean_se=("se", "mean"),
    ).reset_index()

    out = out.merge(_sigma_eps_metrics(draws), on=GROUP, how="left")
    out["mean_bias"] = out["mean_rho"] - RHO
    out["mc_se"] = out["sd"] / np.sqrt(out["R"])
    out["plim"] = [plim(e, m, t) for e, m, t in zip(out["estimator"], out["model"], out["T"], strict=True)]
    out["plim_bias"] = out["plim"] - RHO
    out["se_over_sd"] = out["mean_se"] / out["sd"]

    order = {name: i for i, name in enumerate(ESTIMATORS)}
    out = out.sort_values(
        ["estimator", "model", "N", "T"],
        key=lambda col: col.map(order) if col.name == "estimator" else col,
    ).reset_index(drop=True)
    return out[METRIC_COLUMNS]


def _cell_grid(summary: pd.DataFrame, formatter) -> tuple[pd.DataFrame, list[int]]:
    """Pivot a summary into rows = estimator x model (x N) and columns = T."""
    t_values = sorted(summary["T"].unique())
    n_values = sorted(summary["N"].unique())
    show_n = len(n_values) > 1

    rows = []
    for estimator in ESTIMATORS:
        for model in MODELS:
            for n in n_values:
                block = summary[
                    (summary["estimator"] == estimator)
                    & (summary["model"] == model)
                    & (summary["N"] == n)
                ]
                if block.empty:
                    continue
                row: dict[str, object] = {"Estimator": estimator, "Model": model}
                if show_n:
                    row["N"] = n
                for t in t_values:
                    cell = block[block["T"] == t]
                    row[f"T={t}"] = formatter(cell.iloc[0]) if len(cell) else ""
                rows.append(row)
    return pd.DataFrame(rows), t_values


def _fmt_main(row: pd.Series) -> str:
    return f"{row['mean_rho']:.3f} ({row['sd']:.3f}) [{row['plim']:.3f}]"


def _fmt_coverage(row: pd.Series) -> str:
    return f"{row['coverage']:.2f}"


def main_table(summary: pd.DataFrame) -> pd.DataFrame:
    return _cell_grid(summary, _fmt_main)[0]


def coverage_table(summary: pd.DataFrame) -> pd.DataFrame:
    return _cell_grid(summary, _fmt_coverage)[0]


def sigma_eps_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Deliverable 5: sigma_eps-hat and its coverage, for the cells that report it."""
    block = summary[summary["mean_sigma_eps"].notna()]
    t_values = sorted(block["T"].unique())
    statistics = [
        ("mean (SD) [plim]", lambda r: f"{r['mean_sigma_eps']:.3f} ({r['sigma_eps_sd']:.3f})"
                                       f" [{r['sigma_eps_plim']:.3f}]"),
        ("coverage", lambda r: f"{r['sigma_eps_coverage']:.2f}"),
    ]

    rows = []
    for estimator in ESTIMATORS:
        for model in MODELS:
            cells = block[(block["estimator"] == estimator) & (block["model"] == model)]
            if cells.empty:
                continue
            for name, formatter in statistics:
                row: dict[str, object] = {
                    "Estimator": estimator, "Model": model, "Statistic": name
                }
                for t in t_values:
                    cell = cells[cells["T"] == t]
                    row[f"T={t}"] = formatter(cell.iloc[0]) if len(cell) else ""
                rows.append(row)
    return pd.DataFrame(rows)


def _to_markdown(table: pd.DataFrame, title: str, caption: str) -> str:
    """Render a pivoted table as a padded GitHub markdown table (no extra deps)."""
    columns = list(table.columns)
    cells = [[str(v) for v in row] for row in table.itertuples(index=False)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in cells)) if cells else len(str(col))
        for i, col in enumerate(columns)
    ]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)) + " |"

    body = [line([str(c) for c in columns]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    body += [line(row) for row in cells]
    return f"## {title}\n\n" + "\n".join(body) + f"\n\n_{caption}_\n"


def build(output_dir: Path, draws_dir: Path | None = None) -> pd.DataFrame:
    """Write summary.csv and the two markdown deliverable tables."""
    output_dir = Path(output_dir)
    draws_dir = Path(draws_dir) if draws_dir is not None else output_dir / "pilot"

    summary = summarize(load_cells(draws_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")

    (output_dir / "table_main.md").write_text(
        _to_markdown(
            main_table(summary),
            "Main table: mean rho-hat by estimator, model and T",
            "Cells: mean rho-hat (SD across replications) [analytical plim]. True rho = 0.7.",
        )
    )
    (output_dir / "table_coverage.md").write_text(
        _to_markdown(
            coverage_table(summary),
            "Coverage table: share of 95% CIs containing rho = 0.7",
            "CI = rho-hat +/- 1.96 x clustered SE; clustering by individual.",
        )
    )
    (output_dir / "table_sigma_eps.md").write_text(
        _to_markdown(
            sigma_eps_table(summary),
            "Sigma-eps table: mean sigma-eps-hat and its coverage",
            "Only the growth-covariance GMM identifies sigma_eps; true sigma_eps = 0.3.",
        )
    )
    print(f"[summary] {len(summary)} cells -> {output_dir / 'summary.csv'}")
    return summary
