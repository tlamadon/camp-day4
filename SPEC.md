# Spec: Dynamic panel AR(1) simulation — pooled OLS, first differences, within, growth-covariance GMM

Sep 24, 2026 · @Thibaut

## Purpose

The simulation shows how four estimators of the autoregressive coefficient behave in a panel AR(1), with and without individual effects. It crosses 4 estimators × 2 data-generating processes and tracks bias, dispersion and coverage as T varies.

The expected story: pooled OLS is fine without effects and badly biased upward with them; OLS in first differences is inconsistent in both cases; the within estimator carries the Nickell bias in both cases, shrinking at rate 1/T. The fourth estimator is the control: GMM on the whole variance–covariance matrix of growth recovers ρ and σ\_ε in both models, at every T, from the same differenced data that defeats first-difference OLS. What sinks FD OLS is therefore not differencing but reading a single moment ratio out of the growth distribution.

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

The observed sample is y\_i0, …, y\_iT, so every estimator has T regression periods (t = 1, …, T), and there are T growth rates Δy\_i1, …, Δy\_iT. Regressing a growth rate on its own lag loses one more period and uses t = 2, …, T.

## Estimators

Each estimator is run on every simulated panel from both models, giving 8 estimator × model cells.

| Estimator | Regression or criterion | Periods used | Obs. per panel | Parameters recovered |
| --- | --- | --- | --- | --- |
| Pooled OLS | y\_it on a constant and y\_i,t−1 | t = 1, …, T | N·T | ρ |
| First-difference OLS | Δy\_it on Δy\_i,t−1, no constant | t = 2, …, T | N·(T−1) | ρ |
| Within (FE) | ỹ\_it on ỹ\_i,t−1, where ỹ is y minus its individual mean over t = 1, …, T | t = 1, …, T | N·T | ρ |
| Growth-covariance GMM | minimum distance between the sample and model covariance matrices of Δy | t = 1, …, T | N growth vectors of length T | ρ and σ\_ε |

For the within estimator, demean the dependent variable and the lag separately: the mean of y\_it uses y\_i1…y\_iT, the mean of the lag uses y\_i0…y\_i,T−1.

Standard errors: cluster by individual i throughout. Report 95% confidence intervals as the estimate ± 1.96 × clustered SE.

### Growth-covariance GMM

Differencing removes α\_i, so the entire distribution of growth is free of the individual effect. What FD OLS then does with that distribution is read one number out of it, the ratio Cov(Δy\_it, Δy\_i,t−1) / Var(Δy\_i,t−1) — and that ratio is (ρ−1)/2, not ρ. The full covariance matrix of growth carries much more, and it is the same matrix in M0 and M1.

**The moments.** Stack growth into Δy\_i = (Δy\_i1, …, Δy\_iT)′, which uses every period including t = 1 since y\_i0 is observed. Under stationarity its variance–covariance matrix is

```latex
\Omega_{ts}(\rho, \sigma_\varepsilon^2) =
\begin{cases}
\dfrac{2\sigma_\varepsilon^2}{1+\rho}, & t = s,\\[6pt]
-\dfrac{\sigma_\varepsilon^2 (1-\rho)}{1+\rho}\,\rho^{|t-s|-1}, & t \neq s.
\end{cases}
```

Ω is Toeplitz, free of α\_i, of σ\_α and of T (T only sets its size): growth is the ARMA(1,1) (1 − ρL) Δy\_it = (1 − L) ε\_it. `proofs/PanelAR1/GrowthACov.lean` derives Ω from the innovation representation of the model and checks the FD OLS ratio as a corollary, both machine-verified.

**Identification.** Ω\_{t,t−1} / Ω\_tt = (ρ−1)/2 is strictly increasing in ρ, so ρ is pinned for every T ≥ 2, and σ\_ε² then follows from the diagonal. σ\_α is *not* identified: Ω does not contain it. That absence is exactly why the estimator is consistent in M1 as well as M0.

**The estimator.** With C the sample second-moment matrix of growth, C\_ts = N⁻¹ Σ\_i Δy\_it Δy\_is — raw, not centred, since E Δy\_it = 0 under the stationary initial condition — set

```latex
\hat\theta = \arg\min_{\rho,\,\sigma_\varepsilon^2}\; \bigl\| C - \Omega(\rho, \sigma_\varepsilon^2) \bigr\|_F^2 ,
\qquad \hat\sigma_\varepsilon = \sqrt{\hat\sigma_\varepsilon^2}.
```

That is GMM on all T² moment conditions E[Δy\_it Δy\_is − Ω\_ts(θ)] = 0 with an identity weight matrix. Two consequences of the fixed weight:

- It is not the efficient weight. The efficient one needs the inverse of the covariance matrix of the moments, which does not exist here: the moment vector repeats each off-diagonal pair, and at T = 50 there are 1,275 distinct entries against N = 500 independent draws. The identity weight never inverts it — it enters only as the 2 × 2 matrix G′ŜG — at the cost of efficiency, not consistency. The overidentification test is dropped for the same reason.
- Uninformative moments down-weight themselves. ∂Ω\_ts/∂θ decays like ρ^{|t−s|}, so entries far off the diagonal enter the influence function with weight near zero however many of them there are.

**Computation.** Ω is linear in σ\_ε², so write Ω(ρ, σ²) = σ² A(ρ) and concentrate σ² out: σ̂²(ρ) = ⟨A(ρ), C⟩ / ⟨A(ρ), A(ρ)⟩, leaving a smooth profiled criterion in the single parameter ρ ∈ (−1, 1). A 1,001-point grid on [−0.995, 0.995] followed by golden-section refinement on the bracketing interval solves it, so no optimiser dependency is added.

**Standard errors.** The GMM sandwich

```latex
\widehat{\operatorname{Var}}(\hat\theta) = (G'G)^{-1} G'\hat{S} G\, (G'G)^{-1} / N,
\qquad G = \partial\,\mathrm{vec}\,\Omega / \partial\theta',
```

with Ŝ the sample covariance of the per-individual moment vectors vec(Δy\_i Δy\_i′). One individual contributes one moment vector, so this clusters by individual by construction, matching the other three estimators. The standard error of σ̂\_ε follows by the delta method, se(σ̂\_ε) = se(σ̂\_ε²) / (2σ̂\_ε).

## Design grid

Baseline: N = 500, R = 10 replications per design point for the local pilot (scaled to 1,000 on the HPC later), T varied to trace out the Nickell bias.

| Dimension | Values | Why |
| --- | --- | --- |
| T | 3, 5, 10, 20, 50 | Within bias is of order 1/T; the others do not vanish in T |
| N | 500 (robustness: 5,000) | Larger N confirms the biases are inconsistency, not finite-sample noise |
| Model | M0, M1 | With and without α\_i |
| R | 10 (pilot); 1,000 on HPC | At R = 10 the Monte Carlo SE of the mean bias is about SD/3, enough to check the code; SD/32 at R = 1,000 |

Total: 5 values of T × 2 models × R draws, with all 4 estimators run on the same draw. Share draws across estimators within a replication so differences between estimators are not Monte Carlo noise.

Seeds: one master seed; replication r in design (model, T, N) gets a seed derived deterministically from those four values, so any single cell can be rerun on its own.

## Analytical benchmarks

The simulated means should match these large-N probability limits; overlay them on every figure as a check on the code.

| Estimator | M0 (no α\_i) | M1 (with α\_i) | Source of bias |
| --- | --- | --- | --- |
| Pooled OLS | 0.700 | 0.982 | Lag correlated with α\_i |
| First-difference OLS | −0.150 | −0.150 | Δy\_i,t−1 correlated with Δε\_it through ε\_i,t−1 |
| Within | Nickell, see below | Nickell, same as M0 | Demeaned lag correlated with the mean of ε |
| Growth-covariance GMM | 0.700 | 0.700 | None; σ̂\_ε has plim 0.300 in both |

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

**Growth-covariance GMM.** Consistent, so the plim is the truth: ρ in both models at every T, and σ\_ε alongside it. The moment conditions E[Δy\_it Δy\_is] = Ω\_ts(ρ, σ\_ε²) hold exactly under the stationary initial condition, Ω is free of α\_i, and the map θ ↦ Ω(θ) is injective for T ≥ 2. Its plim line on Figure 1 is therefore flat at 0.700 in both panels — the reference the other three are measured against. Growth in M1 has the same distribution as growth in M0, so this estimator and the FD and within estimators have identical sampling distributions across the two models; only the seed differs.

## Outputs

Per (model, T, N, estimator) cell, compute over the R replications:

- **Mean bias**: mean of ρ̂ − 0.7, with its Monte Carlo SE.
- **SD**: standard deviation of ρ̂ across replications.
- **RMSE**: square root of mean (ρ̂ − 0.7)².
- **Coverage**: share of 95% CIs containing 0.7.
- **Mean clustered SE / SD**: checks whether the reported SE matches actual dispersion.

The same five statistics are computed for σ̂\_ε in the GMM cells, against σ\_ε = 0.3. The other three estimators leave those columns empty: they estimate ρ only.

Deliverables:

1. Main table: rows = estimator × model, columns = T, cells = mean ρ̂ (SD), with the analytical plim beside each.
2. Figure 1: mean ρ̂ against T, one line per estimator, two panels (M0, M1), dashed plims and a horizontal line at 0.7.
3. Figure 2: density of ρ̂ at T = 10 for the 8 cells, to show that the biased estimators are tightly centred on the wrong value while the GMM density sits on 0.7.
4. Coverage table: same layout as the main table.
5. σ\_ε table: mean σ̂\_ε (SD) [0.300] and its coverage, rows = model, columns = T. GMM cells only.

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
| `summary` | `sim` | Bias, SD, RMSE and coverage tables to `output/summary.csv`, for ρ̂ and for σ̂\_ε |
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
* Code the three regression estimators as closed-form ratios of sums (Σxy / Σx²) rather than calling a regression package. This is faster and makes the moment conditions explicit.
* The GMM estimator is the one exception, and it still needs no new dependency: its criterion profiles down to one parameter, and the sample moment matrix is a single T × T Gram matrix, C = ΔY′ΔY / N. Never build the N × T² matrix of per-individual moments; the sandwich needs only the two N-vectors Δy\_i′ (∂Ω/∂θ\_p) Δy\_i, each one matmul.
* Unit test before running the grid: at N = 10⁶ and T = 10, each estimator should land within 0.005 of its plim in the benchmark table, and σ̂\_ε within 0.005 of 0.3. Also check that feeding the population Ω(θ) in place of C returns θ back, to the ~10⁻⁸ a derivative-free maximum of a smooth criterion allows.
* Save raw ρ̂ and SE draws for every replication, plus σ̂\_ε and its SE where the estimator produces one, so new summary statistics need no rerun.

Optional extensions, one line each:

- Add the levels moment Var(y\_it) = σ\_α²/(1−ρ)² + σ\_ε²/(1−ρ²) to the growth moments, which would identify σ\_α as well; growth alone cannot see it.
- Add Anderson–Hsiao IV (Δy\_i,t−1 instrumented by y\_i,t−2) as a second consistent reference, one that uses the differenced data a single lag at a time rather than all of it at once.
- Use the efficient GMM weight where it is feasible (T(T+1)/2 < N, so T ≤ 30 at N = 500) and report how much of the GMM standard error the identity weight costs.
- Replace the stationary initial condition with y\_i0 = 0 to show how pooled OLS and within results move when the start-up is not in steady state.
- Vary ρ ∈ {0.3, 0.7, 0.95} to show the within bias worsening as persistence rises.

## Spec-check skill

A Claude Code skill in the repo, `.claude/skills/spec-check/SKILL.md`, audits the project against this spec and reports pass or fail per check. It fixes nothing unless asked.

It reads a copy of this spec committed as `SPEC.md`, exported from the doc whenever the doc changes. It runs three checks, in order:

1. **Spec is internally consistent.** It recomputes every plim in the benchmark tables from the stated formulas and parameters, and checks that the grid counts (2 models × 5 values of T) match the Makefile description.
2. **Code matches the spec.** It checks the DGP (ρ, σ\_ε, σ\_α, stationary initial condition), the four estimator definitions (periods used, demeaning, clustering, and for the GMM the Ω formula, the identity weight and the concentrated σ\_ε²), the grid and seeding, the output metrics, and that every Makefile target and dependency in the spec table exists as written (`make -pn`).
3. **Outputs are built from current code and spec.** Each parquet file records the SHA-256 of `SPEC.md` and the git commit in its metadata. The skill flags any output whose hashes differ from the current ones, and requires `make -q all` to report nothing stale.

The mechanical parts of check 3 live in a `make check` target so they also run without Claude; the skill calls it and adds the judgement-based checks 1 and 2.

## To do

- [ ] Make the project a git repo: `git init`, a `.gitignore` for `.venv/` and `output/`, and a first commit with `mise.toml`, `pyproject.toml`, `uv.lock` and the `Makefile`. Scripthut clones from git, so this is also a prerequisite for the HPC runs.
- [ ] Write the spec-check skill and the `make check` target, and commit this spec as `SPEC.md`.
