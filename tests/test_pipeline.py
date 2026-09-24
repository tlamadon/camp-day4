"""The grid, the cell files and the summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim.__main__ import main
from sim.check import check_outputs
from sim.config import ESTIMATORS, MODELS, RHO, T_GRID, Z95
from sim.runner import cell_name, load_cells, parse_cell_name, read_cell_metadata, run_cell, write_cell
from sim.summarize import coverage_table, main_table, summarize


def test_grid_is_two_models_by_five_T():
    assert len(MODELS) == 2
    assert T_GRID == (3, 5, 10, 20, 50)
    assert len(MODELS) * len(T_GRID) == 10


@pytest.mark.parametrize(("model", "T", "N"), [("M0", 3, 500), ("M1", 50, 5000)])
def test_cell_name_round_trips(model, T, N):
    assert parse_cell_name(cell_name(model, T, N)) == (model, T, N)


def test_run_cell_shares_draws_across_estimators():
    df = run_cell("M1", 5, 300, R=4)
    assert len(df) == 4 * len(ESTIMATORS)
    assert set(df["estimator"]) == set(ESTIMATORS)
    # Same replication index, same panel: rerunning one estimator reproduces it.
    again = run_cell("M1", 5, 300, R=4)
    assert np.allclose(df["rho_hat"], again["rho_hat"])
    assert df["se"].gt(0).all()


def test_cell_file_carries_spec_hash_and_commit(tmp_path):
    df = run_cell("M0", 3, 200, R=2)
    path = tmp_path / f"{cell_name('M0', 3, 200)}.parquet"
    written = write_cell(df, path)

    meta = read_cell_metadata(path)
    assert meta["spec_sha256"] == written["spec_sha256"]
    assert len(meta["spec_sha256"]) == 64
    assert meta["git_commit"]
    assert pd.read_parquet(path).shape == df.shape


def test_summary_metrics_and_tables(tmp_path):
    for model in MODELS:
        for T in (3, 5):
            df = run_cell(model, T, 200, R=5)
            write_cell(df, tmp_path / f"{cell_name(model, T, 200)}.parquet")

    draws = load_cells(tmp_path)
    summary = summarize(draws)

    assert len(summary) == len(MODELS) * 2 * len(ESTIMATORS)
    assert (summary["R"] == 5).all()
    assert summary["mean_bias"].equals(summary["mean_rho"] - RHO)
    assert (summary["coverage"].between(0, 1)).all()
    assert (summary["rmse"] >= summary["mean_bias"].abs() - 1e-12).all()
    assert (summary["mc_se"] <= summary["sd"] + 1e-12).all()

    # Coverage is exactly the share of CIs that cover, recomputed from the draws.
    row = summary.iloc[0]
    cell = draws[
        (draws["model"] == row["model"])
        & (draws["T"] == row["T"])
        & (draws["estimator"] == row["estimator"])
    ]
    expected = ((cell["rho_hat"] - RHO).abs() <= Z95 * cell["se"]).mean()
    assert row["coverage"] == pytest.approx(expected)

    main = main_table(summary)
    assert len(main) == len(ESTIMATORS) * len(MODELS)
    assert list(main.columns) == ["Estimator", "Model", "T=3", "T=5"]
    assert coverage_table(summary).shape == main.shape


def test_check_flags_a_stale_output(tmp_path, monkeypatch):
    out = tmp_path / "output"
    df = run_cell("M0", 3, 100, R=2)
    write_cell(df, out / "pilot" / "M0_T3_N100.parquet", extra={"spec_sha256": "0" * 64})

    ok, report = check_outputs(out, expect_cells=False)
    assert not ok
    assert any("stale" in line for line in report)


def test_cli_run_writes_a_cell(tmp_path, capsys):
    target = tmp_path / "cell.parquet"
    code = main(["run", "--model", "M0", "--T", "3", "--N", "150", "--R", "2", "--out", str(target)])
    assert code == 0
    assert target.exists()
    assert "M0_T3_N150" in capsys.readouterr().out
