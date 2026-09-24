# Dynamic panel AR(1): pooled OLS vs first differences vs within

A Monte Carlo that crosses three estimators of the autoregressive coefficient
with two data-generating processes, and tracks bias, dispersion and coverage as
the panel length `T` varies.

`SPEC.md` is the authority for the design, the estimators and every benchmark
number. This README only says how to run it.

## Quick start

```bash
mise install          # provisions python and uv (see mise.toml)
make                  # setup -> tests -> 10 cells -> summary -> figures
make -j sim           # cells in parallel; each is its own file target
make check            # outputs match the current SPEC.md and commit
```

`make check` ties every output to the exact commit it was built from, so the
order is **commit, then build**: after any commit, `make clean && make` restamps
the outputs. A run from a dirty tree is stamped `-dirty` and fails the check.

One cell on its own, which is also the command the HPC scale-up reuses:

```bash
uv run python -m sim run --model M1 --T 10 --N 500 --R 10 --seed 0
```

## What lands where

| Path | Contents |
| --- | --- |
| `output/pilot/<model>_T<T>_N<N>.parquet` | raw ρ̂ and clustered SE for every replication, stamped with the spec hash and git commit |
| `output/summary.csv` | mean bias and its Monte Carlo SE, SD, RMSE, coverage, mean SE / SD, per cell |
| `output/table_main.md` | main table: mean ρ̂ (SD) [plim], rows = estimator × model, columns = T |
| `output/table_coverage.md` | coverage table, same layout |
| `output/figures/fig1_mean_rho_by_T.*` | Figure 1: mean ρ̂ against T, two panels, dashed plims |
| `output/figures/fig2_density_T10.*` | Figure 2: densities of ρ̂ at T = 10 for the six cells |
| `output/tables/*.tex` | LaTeX versions (`make tables`); the PDF build waits on a TeX engine |

## Layout

```
src/sim/
  config.py       design constants -- the only place a parameter value is written
  analytics.py    the plims: pooled, first differences, Nickell
  dgp.py          seeds and the N x (T+1) panel
  estimators.py   the three estimators as closed-form ratios of sums
  runner.py       one design cell -> one parquet, with provenance
  summarize.py    draws -> summary.csv and the two deliverable tables
  figures.py      Figures 1 and 2
  tables.py       LaTeX tables
  check.py        the mechanical half of the spec audit
tests/            the spec's own numbers, including the N = 10^6 plim gate
.claude/skills/spec-check/   the Claude Code skill that audits the repo against SPEC.md
```

## Two choices the spec leaves open

- **Cluster correction.** SPEC.md fixes clustering by individual but not the
  finite-sample factor. The code uses `G/(G-1)` for all three estimators; a
  degrees-of-freedom factor counting the N absorbed effects would inflate the
  within SE by ~22% at T = 3 and blur the mean-SE-over-SD diagnostic.
- **`setup` and `test` are stamp files** under `.make/`, with phony aliases, so
  that the cell targets depending on them stay incremental.
