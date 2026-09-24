"""Run one design cell and write it to parquet.

One cell is one (model, T, N) design point at R replications.  All four
estimators see the same simulated panel within a replication, so the gaps
between them are not Monte Carlo noise.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ESTIMATORS, MASTER_SEED
from .dgp import seed_sequence, simulate_panel
from .estimators import ESTIMATOR_FUNCS
from .provenance import provenance

CELL_RE = re.compile(r"^(?P<model>M\d+)_T(?P<T>\d+)_N(?P<N>\d+)$")


def cell_name(model: str, T: int, N: int) -> str:
    return f"{model}_T{T}_N{N}"


def parse_cell_name(stem: str) -> tuple[str, int, int]:
    m = CELL_RE.match(stem)
    if m is None:
        raise ValueError(f"cannot parse cell name {stem!r}, expected e.g. 'M1_T10_N500'")
    return m["model"], int(m["T"]), int(m["N"])


def run_cell(
    model: str,
    T: int,
    N: int,
    R: int,
    master_seed: int = MASTER_SEED,
) -> pd.DataFrame:
    """Simulate R panels and fit all four estimators on each.

    Returns one row per (replication, estimator) with the raw estimates and their
    clustered standard errors, so new summary statistics never need a rerun.
    The sigma_eps columns are NaN for the three estimators that identify rho only.
    """
    rows: list[dict[str, object]] = []
    for rep in range(R):
        ss = seed_sequence(model, T, N, rep, master=master_seed)
        rng = np.random.default_rng(ss)
        y = simulate_panel(model, T, N, rng)
        for name in ESTIMATORS:
            fit = ESTIMATOR_FUNCS[name](y)
            rows.append(
                {
                    "model": model,
                    "T": T,
                    "N": N,
                    "R": R,
                    "rep": rep,
                    "master_seed": master_seed,
                    # Full entropy behind this draw, so one replication can be rerun alone.
                    "seed_entropy": ",".join(str(x) for x in ss.entropy),
                    "estimator": name,
                    "rho_hat": fit.rho_hat,
                    "se": fit.se,
                    "sigma_eps_hat": fit.sigma_eps_hat,
                    "sigma_eps_se": fit.sigma_eps_se,
                }
            )
    df = pd.DataFrame(rows)
    return df.astype(
        {
            "model": "string",
            "T": "int32",
            "N": "int64",
            "R": "int32",
            "rep": "int32",
            "seed_entropy": "string",
            "estimator": "string",
        }
    )


def write_cell(df: pd.DataFrame, path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Write one cell to parquet, stamping spec hash and git commit into the schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = provenance()
    if extra:
        meta.update(extra)

    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta.update({k.encode(): v.encode() for k, v in meta.items()})
    pq.write_table(table.replace_schema_metadata(schema_meta), path, compression="zstd")
    return meta


def read_cell_metadata(path: Path) -> dict[str, str]:
    """Read back the string key/value metadata stamped by `write_cell`."""
    raw = pq.read_schema(Path(path)).metadata or {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = key.decode(errors="replace")
        if name == "pandas":  # pyarrow's own blob, not ours
            continue
        out[name] = value.decode(errors="replace")
    return out


def load_cells(directory: Path) -> pd.DataFrame:
    """Read every cell parquet under `directory` into one long frame."""
    directory = Path(directory)
    paths = sorted(directory.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {directory}; run `make sim` first")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_and_write(
    model: str,
    T: int,
    N: int,
    R: int,
    out: Path,
    master_seed: int = MASTER_SEED,
    quiet: bool = False,
) -> Path:
    start = time.perf_counter()
    df = run_cell(model, T, N, R, master_seed=master_seed)
    write_cell(df, out)
    if not quiet:
        means = df.groupby("estimator", observed=True)["rho_hat"].mean()
        summary = "  ".join(f"{k}={means[k]:+.3f}" for k in ESTIMATORS)
        print(
            f"[sim] {cell_name(model, T, N)} R={R} "
            f"({time.perf_counter() - start:5.1f}s)  {summary}  -> {out}"
        )
    return Path(out)
