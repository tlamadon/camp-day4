"""Collapse the raw replication draws into the summary statistics of SPEC.md.

Reads every parquet under the output folder, so the pilot and a later HPC run
share one pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import plim, plim_param
from .config import ESTIMATORS, MODELS, RHO, SIGMA_EPS, Z95, sigma_nu
from .runner import load_cells

GROUP = ["model", "T", "N", "estimator"]

#: The variance parameters, with the true value each cell is scored against.
#: sigma_nu's truth is the model's, so the same estimator is scored against 0 in
#: M0 and M1 and against 0.2 in M2.
PARAMETERS: dict[str, object] = {
    "sigma_eps": lambda model: SIGMA_EPS,
    "sigma_nu": sigma_nu,
}


def _parameter_columns(parameter: str) -> list[str]:
    return [
        f"mean_{parameter}",
        f"{parameter}_plim",
        f"{parameter}_bias",
        f"{parameter}_mc_se",
        f"{parameter}_sd",
        f"{parameter}_rmse",
        f"{parameter}_coverage",
        f"{parameter}_mean_se",
        f"{parameter}_se_over_sd",
    ]


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
    # The same five statistics per variance parameter; NaN where none is reported.
    *[column for parameter in PARAMETERS for column in _parameter_columns(parameter)],
]


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _parameter_metrics(draws: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """The same five statistics for one variance parameter, where it is reported.

    A replication whose variance was pinned at zero has no usable standard error
    -- the truth is on the boundary of the parameter space there -- so it drops
    out of the coverage and the mean SE rather than being counted either way.
    """
    hat, se = f"{parameter}_hat", f"{parameter}_se"
    block = draws[draws[hat].notna()].copy()
    if block.empty:
        return pd.DataFrame(columns=GROUP)

    truth = PARAMETERS[parameter]
    block["truth"] = [truth(m) for m in block["model"]]
    block["error"] = block[hat] - block["truth"]
    block["covered"] = np.where(
        block[se].notna(), (block[se] * Z95 >= block["error"].abs()).astype(float), np.nan
    )

    out = (
        block.groupby(GROUP, observed=True)
        .agg(
            reps=(hat, "size"),
            mean=(hat, "mean"),
            sd=(hat, lambda s: s.std(ddof=1)),
            rmse=("error", _rmse),
            coverage=("covered", "mean"),
            mean_se=(se, "mean"),
            truth=("truth", "first"),
        )
        .reset_index()
    )
    out["plim"] = [
        plim_param(parameter, e, m, t)
        for e, m, t in zip(out["estimator"], out["model"], out["T"], strict=True)
    ]
    out["bias"] = out["mean"] - out["truth"]
    out["mc_se"] = out["sd"] / np.sqrt(out["reps"])
    out["se_over_sd"] = out["mean_se"] / out["sd"]

    names = {c: f"{parameter}_{c}" for c in ("plim", "bias", "mc_se", "sd", "rmse",
                                             "coverage", "mean_se", "se_over_sd")}
    names["mean"] = f"mean_{parameter}"
    return out.drop(columns=["reps", "truth"]).rename(columns=names)


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

    for parameter in PARAMETERS:
        out = out.merge(_parameter_metrics(draws, parameter), on=GROUP, how="left")
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


def _blank(value: float, digits: int = 3) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def variance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Deliverable 5: the variance parameters and their coverage, where reported."""
    t_values = sorted(summary["T"].unique())
    rows = []
    for parameter in PARAMETERS:
        block = summary[summary[f"mean_{parameter}"].notna()]
        statistics = [
            ("mean (SD) [plim]", lambda r, p=parameter: f"{r[f'mean_{p}']:.3f}"
                                 f" ({r[f'{p}_sd']:.3f}) [{_blank(r[f'{p}_plim'])}]"),
            ("coverage", lambda r, p=parameter: _blank(r[f"{p}_coverage"], 2)),
        ]
        for estimator in ESTIMATORS:
            for model in MODELS:
                cells = block[(block["estimator"] == estimator) & (block["model"] == model)]
                if cells.empty:
                    continue
                for name, formatter in statistics:
                    row: dict[str, object] = {
                        "Parameter": parameter,
                        "Estimator": estimator,
                        "Model": model,
                        "Statistic": name,
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
    (output_dir / "table_variances.md").write_text(
        _to_markdown(
            variance_table(summary),
            "Variance table: the parameters the growth fits return beside rho",
            "True sigma_eps = 0.3 in every model; true sigma_nu = 0.2 in M2 and 0 elsewhere. "
            "A blank coverage is a variance pinned at zero, where the interval is degenerate.",
        )
    )
    print(f"[summary] {len(summary)} cells -> {output_dir / 'summary.csv'}")
    return summary
