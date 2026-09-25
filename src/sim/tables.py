"""LaTeX versions of the deliverable tables.

The PDF build is on hold until a TeX engine goes into mise.toml; this module
only writes the .tex sources, which compile standalone under booktabs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ESTIMATOR_LABELS, MODEL_LABELS
from .summarize import coverage_table, main_table, variance_table


def _escape(text: str) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _relabel(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["Estimator"] = out["Estimator"].map(lambda e: ESTIMATOR_LABELS.get(e, e))
    out["Model"] = out["Model"].map(lambda m: MODEL_LABELS.get(m, m))
    return out


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    table = _relabel(table)
    columns = list(table.columns)
    align = "ll" + "c" * (len(columns) - 2)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(_escape(v) for v in row) + r" \\" for row in table.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    summary = pd.read_csv(output_dir / "summary.csv")
    target = output_dir / "tables"
    target.mkdir(parents=True, exist_ok=True)

    written = []
    specs = [
        (
            main_table(summary),
            "Mean $\\hat\\rho$ (SD across replications) [analytical plim]. True $\\rho = 0.7$.",
            "tab:main",
            "table_main.tex",
        ),
        (
            coverage_table(summary),
            "Share of 95\\% confidence intervals containing $\\rho = 0.7$.",
            "tab:coverage",
            "table_coverage.tex",
        ),
        (
            variance_table(summary),
            "The variance parameters the growth fits return beside $\\rho$, as "
            "mean (SD) [plim] and coverage. True $\\sigma_\\varepsilon = 0.3$ "
            "throughout; true $\\sigma_\\nu = 0.2$ in M2 and $0$ elsewhere.",
            "tab:variances",
            "table_variances.tex",
        ),
    ]
    for table, caption, label, name in specs:
        path = target / name
        path.write_text(to_latex(table, caption, label))
        written.append(path)
        print(f"[tables] -> {path}")
    return written
