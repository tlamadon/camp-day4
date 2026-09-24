---
name: spec-check
description: Audit this dynamic-panel AR(1) simulation project against SPEC.md and report pass or fail for each check. Use when asked to check the project against the spec, verify the code still matches SPEC.md, or confirm the outputs are current. It reports only; it fixes nothing unless the user asks.
---

# Spec check

Audit the repository against `SPEC.md` and report **PASS** or **FAIL** per check,
with the evidence for each. **Fix nothing unless the user explicitly asks.** When
a check fails, say what the spec requires, what the repo does, and where.

Run the three checks in order. Later checks stay meaningful even if an earlier
one fails, so run them all and report all results.

Start by reading `SPEC.md` — it is the authority, not this file and not the code.

## Check 1 — the spec is internally consistent

Recompute every number the spec states from the formulas and parameters the spec
states, and compare.

```bash
uv run python -m sim plims
```

That prints the plim of each estimator in each model at each `T` from the
implementation. Independently recompute the same numbers **from the formulas
written in SPEC.md** (do the arithmetic yourself; do not take the code's word for
it) and compare against these tables in the spec:

- "Analytical benchmarks": pooled OLS 0.700 (M0) and 0.982 (M1), first
  differences −0.150 in both, growth-covariance GMM 0.700 in both (it is
  consistent, so its plim is ρ itself, and σ̂_ε has plim 0.300).
- "Within (Nickell 1981)": the bias and plim columns at T = 3, 5, 10, 20, 50, and
  that the leading term −(1+ρ)/(T−1) approaches the exact bias as T grows.
- The variance note: σ²\_μ = σ²\_α/(1−ρ)² ≈ 2.78, σ²\_u = σ²\_ε/(1−ρ²) ≈ 0.176,
  ratio ≈ 16.

Also check the grid counts: the spec's design grid (2 models × 5 values of T =
10 cells) must equal the number of cells the Makefile builds.

```bash
make -Bn sim | grep -c 'sim run'   # expect 10
```

(`-B` forces the dry run to print every recipe even when the cells are already
built; a plain `make -n sim` prints nothing once everything is up to date.)

Report each recomputed number that disagrees with the spec by more than the
precision the spec prints.

## Check 2 — the code matches the spec

Read the source and compare it with the spec section by section. This is the
judgement-heavy check; quote the line you are judging.

| What the spec fixes | Where the code should show it |
| --- | --- |
| ρ = 0.7, σ\_ε = 0.3, σ\_α = 0 (M0) / 0.5 (M1) | `src/sim/config.py` |
| Stationary initial condition y\_i0 = α\_i/(1−ρ) + u\_i0, u\_i0 ~ N(0, σ²\_ε/(1−ρ²)) | `src/sim/dgp.py` |
| Panel simulated as one N × (T+1) array, looping over t only | `src/sim/dgp.py` |
| Pooled OLS: y\_it on a constant and the lag, t = 1..T | `src/sim/estimators.py` |
| First differences: Δy\_it on Δy\_i,t−1, no constant, t = 2..T | `src/sim/estimators.py` |
| Within: y and its lag demeaned **separately** (y over t = 1..T, lag over t = 0..T−1) | `src/sim/estimators.py` |
| Ω: 2σ²\_ε/(1+ρ) on the diagonal, −σ²\_ε(1−ρ)/(1+ρ)·ρ^(\|t−s\|−1) off it | `src/sim/analytics.py` |
| GMM: identity weight on all T² growth moments, σ²\_ε concentrated out, grid then golden section over ρ, growth at t = 1..T (not 2..T) | `src/sim/estimators.py` |
| GMM standard errors: the sandwich (G′G)⁻¹G′ŜG(G′G)⁻¹/N, one moment vector per individual | `src/sim/estimators.py` |
| The three regressions written as closed-form ratios of sums, not a regression package | `src/sim/estimators.py` |
| SEs clustered by individual; CI = ρ̂ ± 1.96 × SE | `src/sim/estimators.py`, `src/sim/summarize.py` |
| Grid T ∈ {3,5,10,20,50}, N = 500, R = 10 pilot | `src/sim/config.py`, `Makefile` |
| Four estimators, so 8 estimator × model cells | `src/sim/config.py` |
| One master seed; replication seed derived from (model, T, N, r) | `src/sim/dgp.py` |
| All four estimators share the same draw within a replication | `src/sim/runner.py` |
| Raw ρ̂ and SE saved per replication, plus σ̂_ε and its SE where there is one | `src/sim/runner.py` |
| Outputs: mean bias + its Monte Carlo SE, SD, RMSE, coverage, mean SE / SD — and the same five for σ̂_ε | `src/sim/summarize.py` |
| Deliverables: main table, Figure 1, Figure 2, coverage table, σ\_ε table | `src/sim/summarize.py`, `src/sim/figures.py` |

Then confirm every Makefile target and dependency in the spec's Makefile table
exists as written:

```bash
make -pn 2>/dev/null | grep -E '^(setup|test|sim|summary|figures|tables|all|clean|check):'
```

Check the dependency edges too, not just the target names: `test` on `setup`,
each cell on `test` and `src/sim/*.py`, `sim` on all 10 cells, `summary` on
`sim`, `figures` on `summary`, `all` on `figures` and `summary`.

Two deviations are deliberate and documented; treat them as PASS but mention
them once:

- `test` and `setup` are stamp files under `.make/` with phony aliases, so that
  file targets depending on them stay incremental.
- The spec does not pin a finite-sample correction for the clustered SE; the code
  uses G/(G−1), explained in the `src/sim/estimators.py` docstring.
- The spec does not pin how far to refine the GMM search; the code runs 40
  golden-section iterations, which is past the precision of a smooth maximum.

Finally, run the tests — they encode the spec's own numbers:

```bash
make test
```

## Check 3 — outputs are built from the current code and spec

Mechanical, so it also runs without Claude:

```bash
make check
```

It verifies that every parquet under `output/pilot/` carries the SHA-256 of the
current `SPEC.md` and the current git commit, that the working tree is clean,
that every expected cell and deliverable exists, and that `make -q all` reports
nothing stale. Report its output verbatim on failure.

A `-dirty` commit stamp or a mismatched spec hash means the outputs predate the
current state of the repo: the fix is a commit followed by `make all`, but only
do that if the user asks.

## Reporting

Report as a short list, one line per check, then the details of any failure:

```
Check 1  spec internally consistent      PASS
Check 2  code matches spec               FAIL  (2 findings)
Check 3  outputs current                 PASS
```

State what is wrong and where. Do not change any file unless the user asks for a
fix.
