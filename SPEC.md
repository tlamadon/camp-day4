# Spec: Dynamic panel AR(1) simulation — pooled OLS vs first differences vs within

Sep 24, 2026 · @Thibaut

## Purpose

The simulation shows how three simple estimators of the autoregressive coefficient behave in a panel AR(1), with and without individual effects. It crosses 3 estimators × 2 data-generating processes and tracks bias, dispersion and coverage as T varies.

The expected story: pooled OLS is fine without effects and badly biased upward with them; OLS in first differences is inconsistent in both cases; the within estimator carries the Nickell bias in both cases, shrinking at rate 1/T.

## Data-generating process

Both models share one equation, for i = 1, …, N and t = 1, …, T:

```latex
y_{it} = \alpha_i + \rho\, y_{i,t-1} + \varepsilon_{it}, \qquad \varepsilon_{it} \overset{iid}{\sim} N(0, \sigma_\varepsilon^2)
```

| Model | Individual effect | Parameters |
| --- | --- | --- |
| M0: no effects | α\_i = 0 for all i | ρ = 0.7, σ\_ε = 0.3 |
| M1: fixed effects | α\_i \~ N(0, σ\_α²), iid, independent of ε | ρ = 0.7, σ\_ε = 0.3, σ\_α = 0.5 |

**Initial condition.** Draw y\_i0 from the stationary distribution so results do not depend on start-up transients:

```latex
y_{i0} = \frac{\alpha_i}{1-\rho} + u_{i0}, \qquad u_{i0} \sim N\!\left(0, \frac{\sigma_\varepsilon^2}{1-\rho^2}\right)
```

With these values the long-run mean μ\_i = α\_i / (1 − ρ) has variance 2.78, while the transitory component has variance 0.176. Between-individual variation dominates by a factor of about 16, which drives the pooled OLS bias in M1.

The observed sample is y\_i0, …, y\_iT, so every estimator uses T regression periods (t = 1, …, T). Differencing estimators lose one more period and use t = 2, …, T.

## Estimators

Each estimator is run on every simulated panel from both models, giving 6 estimator × model cells.

| Estimator | Regression | Periods used | Obs. per panel |
| --- | --- | --- | --- |
| Pooled OLS | y\_it on a constant and y\_i,t−1 | t = 1, …, T | N·T |
| First-difference OLS | Δy\_it on Δy\_i,t−1, no constant | t = 2, …, T | N·(T−1) |
| Within (FE) | ỹ\_it on ỹ\_i,t−1, where ỹ is y minus its individual mean over t = 1, …, T | t = 1, …, T | N·T |

For the within estimator, demean the dependent variable and the lag separately: the mean of y\_it uses y\_i1…y\_iT, the mean of the lag uses y\_i0…y\_i,T−1.

Standard errors: cluster by individual i for all three. Report 95% confidence intervals as the estimate ± 1.96 × clustered SE.

## Design grid

Baseline: N = 500, R = 10 replications per design point for the local pilot (scaled to 1,000 on the HPC later), T varied to trace out the Nickell bias.

| Dimension | Values | Why |
| --- | --- | --- |
| T | 3, 5, 10, 20, 50 | Within bias is of order 1/T; the others do not vanish in T |
| N | 500 (robustness: 5,000) | Larger N confirms the biases are inconsistency, not finite-sample noise |
| Model | M0, M1 | With and without α\_i |
| R | 10 (pilot); 1,000 on HPC | At R = 10 the Monte Carlo SE of the mean bias is about SD/3, enough to check the code; SD/32 at R = 1,000 |

Total: 5 values of T × 2 models × R draws, with all 3 estimators run on the same draw. Share draws across estimators within a replication so differences between estimators are not Monte Carlo noise.

Seeds: one master seed; replication r in design (model, T, N) gets a seed derived deterministically from those four values, so any single cell can be rerun on its own.

## Analytical benchmarks

The simulated means should match these large-N probability limits; overlay them on every figure as a check on the code.

| Estimator | M0 (no α\_i) | M1 (with α\_i) | Source of bias |
| --- | --- | --- | --- |
| Pooled OLS | 0.700 | 0.982 | Lag correlated with α\_i |
| First-difference OLS | −0.150 | −0.150 | Δy\_i,t−1 correlated with Δε\_it through ε\_i,t−1 |
| Within | Nickell, see below | Nickell, same as M0 | Demeaned lag correlated with the mean of ε |

**Pooled OLS in M1.** Write y\_i,t−1 = μ\_i + u\_i,t−1 with μ\_i = α\_i/(1−ρ). Under stationarity:

```latex
\operatorname{plim} \hat\rho_{OLS} = \frac{\sigma_\mu^2 + \rho\,\sigma_u^2}{\sigma_\mu^2 + \sigma_u^2}, \qquad \sigma_\mu^2 = \frac{\sigma_\alpha^2}{(1-\rho)^2},\; \sigma_u^2 = \frac{\sigma_\varepsilon^2}{1-\rho^2}
```

**First differences.** Cov(Δy\_i,t−1, Δε\_it) = −σ\_ε² and Var(Δy\_i,t−1) = 2σ\_ε²/(1+ρ), so the plim is ρ − (1+ρ)/2 = (ρ−1)/2. It does not depend on T, σ\_ε or σ\_α.

**Within (Nickell 1981).** With A\_T = (1−ρ^T) / (T(1−ρ)):

```latex
\operatorname{plim} (\hat\rho_{W} - \rho) = -\frac{1+\rho}{T-1}\,(1-A_T)\left[1 - \frac{2\rho}{(1-\rho)(T-1)}(1-A_T)\right]^{-1}
```

| T | Within bias | Within plim | Leading term −(1+ρ)/(T−1) |
| --- | --- | --- | --- |
| 3 | −0.620 | 0.080 | −0.850 |
| 5 | −0.394 | 0.306 | −0.425 |
| 10 | −0.197 | 0.503 | −0.189 |
| 20 | −0.094 | 0.606 | −0.089 |
| 50 | −0.036 | 0.664 | −0.035 |

The within bias is identical in M0 and M1 because demeaning removes α\_i exactly; its value depends only on ρ and T.

## Outputs

Per (model, T, N, estimator) cell, compute over the R replications:

- **Mean bias**: mean of ρ̂ − 0.7, with its Monte Carlo SE.
- **SD**: standard deviation of ρ̂ across replications.
- **RMSE**: square root of mean (ρ̂ − 0.7)².
- **Coverage**: share of 95% CIs containing 0.7.
- **Mean clustered SE / SD**: checks whether the reported SE matches actual dispersion.

Deliverables:

1. Main table: rows = estimator × model, columns = T, cells = mean ρ̂ (SD), with the analytical plim beside each.
2. Figure 1: mean ρ̂ against T, one line per estimator, two panels (M0, M1), dashed plims and a horizontal line at 0.7.
3. Figure 2: density of ρ̂ at T = 10 for the 6 cells, to show that the biased estimators are tightly centred on the wrong value.
4. Coverage table: same layout as the main table.

## Implementation notes and extensions

**Tooling.** Python, managed with uv; tool versions and tasks pinned with mise.

- `mise.toml` pins `python` and `uv` under `[tools]`; the task graph lives in the Makefile below. A TeX engine (e.g. tectonic) gets added to `mise.toml` once we write the LaTeX tables.
- `uv init` project with `pyproject.toml` and a committed `uv.lock`; dependencies: numpy, pandas, matplotlib, pytest (dev).
- One entry point, `uv run python -m sim run --model M1 --T 10 --N 500 --R 10 --seed 0`, writing one parquet file per cell. The HPC scale-up (on hold for now) reuses this same command, so nothing in the code depends on where it runs.
- Summary and figures read all parquet files in the output folder, so pilot and HPC outputs share one pipeline.

**Makefile.** A repo-root `Makefile` holds every task and its dependencies. Simulation outputs are file targets (one parquet per cell), so `make -j` runs cells in parallel and reruns only what is stale.

| Target | Depends on | Does |
| --- | --- | --- |
| `setup` | `pyproject.toml`, `uv.lock` | `uv sync` (assumes `mise install` has been run) |
| `test` | `setup` | `uv run pytest`, including the plim unit test |
| `output/pilot/<model>_T<T>_N<N>.parquet` | `test`, `src/sim/*.py` | Pattern rule: one cell at R = 10 via the entry point |
| `sim` | all 10 cell files (2 models × 5 values of T) | Phony target collecting the pilot cells |
| `summary` | `sim` | Bias, SD, RMSE and coverage tables to `output/summary.csv` |
| `figures` | `summary` | Figures 1 and 2 to `output/figures/` |
| `tables` | `summary` | LaTeX tables (on hold, with the TeX engine) |
| `all` | `figures`, `summary` | Default target |
| `clean` | none | Removes `output/` |
| check | SPEC.md, all outputs | Verifies each output's spec hash and git commit match the current ones, and that make -q all reports nothing stale |

The design grid (models, T, N, R) is defined once as Make variables at the top, so the HPC run can override R and N from the command line.

**HPC (on hold).** When we scale up, runs go through [scripthut](https://github.com/tlamadon/scripthut) rather than hand-written Slurm scripts. The plan, to be revisited then:

- A workflow file in `.hut/workflows/` whose tasks are one per (model, T, N) cell, with dot-notation ids such as `sim.M1.T10`, each calling the entry point above with R = 1,000.
- A final `summary` task with `deps: ["sim.*"]` that builds the tables and figures once every cell finishes.
- The cluster environment built from the same `uv.lock`, so pilot and HPC use identical package versions.

* Simulate the whole panel as an N × (T+1) array, looping over t only; each replication is then a few vectorised operations.
* Code the three estimators as closed-form ratios of sums (Σxy / Σx²) rather than calling a regression package. This is faster and makes the moment conditions explicit.
* Unit test before running the grid: at N = 10⁶ and T = 10, each estimator should land within 0.005 of its plim in the benchmark table.
* Save raw ρ̂ and SE draws for every replication, so new summary statistics need no rerun.

Optional extensions, one line each:

- Add Anderson–Hsiao IV (Δy\_i,t−1 instrumented by y\_i,t−2) as a consistent reference, which would show the FD failure is an endogeneity problem, not a differencing problem.
- Replace the stationary initial condition with y\_i0 = 0 to show how pooled OLS and within results move when the start-up is not in steady state.
- Vary ρ ∈ {0.3, 0.7, 0.95} to show the within bias worsening as persistence rises.

## Spec-check skill

A Claude Code skill in the repo, `.claude/skills/spec-check/SKILL.md`, audits the project against this spec and reports pass or fail per check. It fixes nothing unless asked.

It reads a copy of this spec committed as `SPEC.md`, exported from the doc whenever the doc changes. It runs three checks, in order:

1. **Spec is internally consistent.** It recomputes every plim in the benchmark tables from the stated formulas and parameters, and checks that the grid counts (2 models × 5 values of T) match the Makefile description.
2. **Code matches the spec.** It checks the DGP (ρ, σ\_ε, σ\_α, stationary initial condition), the three estimator definitions (periods used, demeaning, clustering), the grid and seeding, the output metrics, and that every Makefile target and dependency in the spec table exists as written (`make -pn`).
3. **Outputs are built from current code and spec.** Each parquet file records the SHA-256 of `SPEC.md` and the git commit in its metadata. The skill flags any output whose hashes differ from the current ones, and requires `make -q all` to report nothing stale.

The mechanical parts of check 3 live in a `make check` target so they also run without Claude; the skill calls it and adds the judgement-based checks 1 and 2.

## To do

- [ ] Make the project a git repo: `git init`, a `.gitignore` for `.venv/` and `output/`, and a first commit with `mise.toml`, `pyproject.toml`, `uv.lock` and the `Makefile`. Scripthut clones from git, so this is also a prerequisite for the HPC runs.
- [ ] Write the spec-check skill and the `make check` target, and commit this spec as `SPEC.md`.
