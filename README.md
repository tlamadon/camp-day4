# Dynamic panel AR(1): pooled OLS, first differences, within, growth-covariance GMM

A Monte Carlo that crosses five estimators of the autoregressive coefficient
with three data-generating processes, and tracks bias, dispersion and coverage
as the panel length `T` varies. The three regressions are inconsistent somewhere;
the two GMM fits of the full covariance matrix of growth are the reference, and
they recover the variance parameters along the way. M2 splits them: it puts
measurement error in the data, and only the fit whose Ω carries a term for it
survives.

`SPEC.md` is the authority for the design, the estimators and every benchmark
number. This README only says how to run it.

## Quick start

```bash
mise install          # provisions python, uv and elan (see mise.toml)
make                  # setup -> tests -> 10 cells -> summary -> figures
make -j sim           # cells in parallel; each is its own file target
make report           # one-page HTML summary at output/report.html
make check            # outputs match the current SPEC.md and commit
make proof            # machine-check the growth autocovariance matrix (Lean)
```

`report` and `tables` sit outside `all`, so the target graph SPEC.md describes is
unchanged; run them when you want the page or the LaTeX.

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
| `output/pilot/<model>_T<T>_N<N>.parquet` | raw ρ̂ and clustered SE for every replication (and σ̂_ε, σ̂_ν where the estimator reports them), stamped with the spec hash and git commit |
| `output/summary.csv` | mean bias and its Monte Carlo SE, SD, RMSE, coverage, mean SE / SD, per cell — for ρ̂ and for each variance parameter |
| `output/table_main.md` | main table: mean ρ̂ (SD) [plim], rows = estimator × model, columns = T |
| `output/table_coverage.md` | coverage table, same layout |
| `output/table_variances.md` | mean σ̂ (SD) [plim] and coverage for σ_ε and σ_ν; growth-fit rows only |
| `output/figures/fig1_mean_rho_by_T.*` | Figure 1: mean ρ̂ against T, two panels, dashed plims |
| `output/figures/fig2_density_T10.*` | Figure 2: densities of ρ̂ at T = 10, one panel per model |
| `output/tables/*.tex` | LaTeX versions (`make tables`); the PDF build waits on a TeX engine |
| `output/report.html` | one-page HTML summary of the whole pilot (`make report`) -- open it straight from disk |
| `output/artifact.html` | the same page as a fragment, for publishing as an Artifact |

## Layout

```
src/sim/
  config.py       design constants -- the only place a parameter value is written
  growth.py       Omega, its two shapes, and the criterion that fits them
  analytics.py    the plims, closed-form for M0/M1 and general for all three
  dgp.py          seeds and the N x (T+1) panel
  estimators.py   three closed-form ratios of sums, plus the two growth fits
  runner.py       one design cell -> one parquet, with provenance
  summarize.py    draws -> summary.csv and the three deliverable tables
  figures.py      Figures 1 and 2
  tables.py       LaTeX tables
  report.py       the one-page HTML summary, charts drawn as inline SVG
  check.py        the mechanical half of the spec audit
tests/            the spec's own numbers, including the N = 10^6 plim gate
proofs/           Lean 4 + Mathlib proof of the autocovariance matrix of Dy_it,
                  with the derivation in proofs/README.md (`make proof`)
.claude/skills/spec-check/   the Claude Code skill that audits the repo against SPEC.md
```

## Two choices the spec leaves open

- **Cluster correction.** SPEC.md fixes clustering by individual but not the
  finite-sample factor. The code uses `G/(G-1)` for all three estimators; a
  degrees-of-freedom factor counting the N absorbed effects would inflate the
  within SE by ~22% at T = 3 and blur the mean-SE-over-SD diagnostic.
- **`setup` and `test` are stamp files** under `.make/`, with phony aliases, so
  that the cell targets depending on them stay incremental.
- **Golden-section iterations.** SPEC.md fixes the grid and the refinement but
  not how far to refine. 40 iterations take the bracket to ~3e-11, well past the
  ~1e-8 to which the location of a smooth maximum is resolvable at all.
- **Non-negative variances.** SPEC.md fixes the criterion but not what to do
  when it wants a negative variance. The fit holds both at zero or above, which
  is why σ̂_ν is pinned at 0 in M0 and M1 rather than going negative there.
