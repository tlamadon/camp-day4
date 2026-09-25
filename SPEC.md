# Spec: Dynamic panel AR(1) simulation — pooled OLS, first differences, within, growth-covariance GMM

Sep 24, 2026 · @Thibaut

## Purpose

The simulation shows how five estimators of the autoregressive coefficient behave in a panel AR(1), with and without individual effects and with and without measurement error. It crosses 5 estimators × 3 data-generating processes and tracks bias, dispersion and coverage as T varies.

The expected story: pooled OLS is fine without effects and badly biased upward with them; OLS in first differences is inconsistent in every case; the within estimator carries the Nickell bias, shrinking at rate 1/T. The last two are the controls, and they split. GMM on the whole variance–covariance matrix of growth recovers ρ and σ\_ε from the same differenced data that defeats first-difference OLS — so what sinks FD OLS is not differencing but reading a single moment ratio out of the growth distribution. But that only holds while the matrix being fitted is the right one: put measurement error in the data and the AR(1) version of the fit loses a third of ρ, while the version that carries a measurement-error term keeps it. Fitting all of the moments is not enough; they have to be the model's moments.

## Data-generating process

Both models share one equation, for i = 1, …, N and t = 1, …, T:

```latex
y_{it} = \alpha_i + \rho\, y_{i,t-1} + \varepsilon_{it}, \qquad \varepsilon_{it} \overset{iid}{\sim} N(0, \sigma_\varepsilon^2)
```

| Model | Individual effect | Measurement error | Parameters |
| --- | --- | --- | --- |
| M0: no effects | α\_i = 0 for all i | none | ρ = 0.7, σ\_ε = 0.3 |
| M1: fixed effects | α\_i \~ N(0, σ\_α²), iid, independent of ε, as an **intercept** | none | ρ = 0.7, σ\_ε = 0.3, σ\_α = 0.5 |
| M2: effects + measurement error | α\_i \~ N(0, σ\_α²), as a **level** | ν\_it \~ iid N(0, σ\_ν²) | ρ = 0.7, σ\_ε = 0.3, σ\_α = 5/3, σ\_ν = 0.2 |

**Initial condition.** Draw y\_i0 from the stationary distribution so results do not depend on start-up transients:

```latex
y_{i0} = \frac{\alpha_i}{1-\rho} + u_{i0}, \qquad u_{i0} \sim N\!\left(0, \frac{\sigma_\varepsilon^2}{1-\rho^2}\right)
```

With these values the long-run mean μ\_i = α\_i / (1 − ρ) has variance 2.78, while the transitory component has variance 0.176. Between-individual variation dominates by a factor of about 16, which drives the pooled OLS bias in M1.

The observed sample is y\_i0, …, y\_iT, so every estimator has T regression periods (t = 1, …, T), and there are T growth rates Δy\_i1, …, Δy\_iT. Regressing a growth rate on its own lag loses one more period and uses t = 2, …, T.

### M2: the persistence is latent

M0 and M1 observe the AR(1) itself. M2 hides it behind noise: the persistent component is a latent p, and what is recorded is p plus a level plus an error.

```latex
p_{it} = \rho\, p_{i,t-1} + \varepsilon_{it}, \qquad y_{it} = \alpha_i + p_{it} + \nu_{it},
\qquad \nu_{it} \overset{iid}{\sim} N(0, \sigma_\nu^2)
```

with ν independent of ε and of α. y is then no longer an AR(1) — it is an ARMA(1,1) around a level — so **none of the closed forms written for M0 and M1 apply to it**, including the Nickell bias.

**Notation.** This model was asked for as p = ρp + μ, y = α + p + ε. The spec already spends ε on the innovation to the persistent component, in M0 and M1, so that reading is kept here and the measurement error is ν: the μ above is ε, and the ε above is ν. The persistent part is then literally the same process, with the same σ\_ε = 0.3, in all three models, and the only new symbol is σ\_ν.

**The effect is a level, not an intercept.** M1's α\_i enters the recursion, and is worth α\_i/(1−ρ) once as a level. M2's enters as the level directly. σ\_α = 0.5/(1−ρ) = 5/3 therefore gives M2 the same between-individual variance as M1, 2.78, so the two models differ by the measurement error and nothing else.

**Initial condition.** p\_i0 from the stationary distribution exactly as above, and ν\_i0 drawn like any other period: y\_i0 is an observation, not a latent state.

**One family.** With ℓ\_i for the level, all three models are

```latex
p_{it} = \rho\, p_{i,t-1} + \varepsilon_{it}, \qquad y_{it} = \ell_i + p_{it} + \nu_{it}
```

at (σ\_ℓ, σ\_ν) = (0, 0) for M0, (5/3, 0) for M1 and (5/3, 0.2) for M2 — M1 is M2 with the error switched off. Writing them this way is what lets one set of formulas cover all three, below.

## Estimators

Each estimator is run on every simulated panel from every model, giving 15 estimator × model cells.

| Estimator | Regression or criterion | Periods used | Obs. per panel | Parameters recovered |
| --- | --- | --- | --- | --- |
| Pooled OLS | y\_it on a constant and y\_i,t−1 | t = 1, …, T | N·T | ρ |
| First-difference OLS | Δy\_it on Δy\_i,t−1, no constant | t = 2, …, T | N·(T−1) | ρ |
| Within (FE) | ỹ\_it on ỹ\_i,t−1, where ỹ is y minus its individual mean over t = 1, …, T | t = 1, …, T | N·T | ρ |
| Growth-covariance GMM | minimum distance between the sample and model covariance matrices of Δy | t = 1, …, T | N growth vectors of length T | ρ and σ\_ε |
| Growth-covariance GMM, with ME | the same, against a model matrix that carries a measurement-error term | t = 1, …, T | N growth vectors of length T | ρ, σ\_ε and σ\_ν |

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

### The same fit, with measurement error

Δy\_it = Δp\_it + Δν\_it, and the two pieces are independent, so Ω is the sum of their matrices. Differencing white noise leaves an MA(1) — variance 2σ\_ν², first autocovariance −σ\_ν², nothing beyond — so

```latex
\Omega(\rho, \sigma_\varepsilon^2, \sigma_\nu^2) = \sigma_\varepsilon^2 A(\rho) + \sigma_\nu^2 B,
\qquad B_{ts} = \begin{cases} 2, & t = s \\ -1, & |t-s| = 1 \\ 0, & \text{otherwise.}\end{cases}
```

A is the matrix above divided by σ\_ε². The two shapes are told apart by how far they reach: A's off-diagonals decay geometrically and never stop, B's stop after one. That is what separates the two variances — and it is also why ρ = 0 is the single point at which they cannot be separated, since A = B exactly there and white noise is white noise however it is labelled.

**Identification.** Past the first off-diagonal only A contributes, so Ω\_{k+1} / Ω\_k = ρ for every k ≥ 2, which pins ρ as soon as T ≥ 4. At T = 3 the three bands are three equations in three unknowns, and Ω\_2 / (Ω\_0 + 2Ω\_1) = (ρ−1)/2 still gives ρ. T = 2 is one equation short: the fit is defined for T ≥ 3.

**Estimation.** The same criterion, ‖C − Ω(θ)‖\_F², with both variances concentrated out — for fixed ρ it is a 2 × 2 linear least squares — and the profile again searched over ρ alone. Variances are held non-negative; with two coefficients the non-negative solution is one of four candidates (both free, either alone, neither), so all four are evaluated and the best feasible one taken.

**The boundary.** σ\_ν = 0 is the truth in M0 and M1, and it is the edge of the parameter space. The fit is pinned there in roughly half the replications; those have no usable interval, so they report no standard error and drop out of σ\_ν's coverage rather than being counted either way. ρ is interior in every model, so its interval is unaffected.

**What the extra parameter costs and buys.** In M0 and M1 this fit is still consistent, but in short panels it pays for the third parameter twice. At T = 3, N = 500 its dispersion is half again the AR(1) fit's, and its mean sits about 0.045 above ρ — a finite-sample bias, not an inconsistency: it falls to 0.014 at N = 5,000 and 0.005 at N = 50,000, and to 0.011 at T = 10 and N = 500. It comes from the non-negativity constraint, which binds in roughly half the replications when there is no measurement error to find, and from T = 3 leaving the fit exactly identified. In M2 it is the only one of the five estimators that is consistent at all.

**Standard errors.** The GMM sandwich

```latex
\widehat{\operatorname{Var}}(\hat\theta) = (G'G)^{-1} G'\hat{S} G\, (G'G)^{-1} / N,
\qquad G = \partial\,\mathrm{vec}\,\Omega / \partial\theta',
```

with Ŝ the sample covariance of the per-individual moment vectors vec(Δy\_i Δy\_i′). One individual contributes one moment vector, so this clusters by individual by construction, matching the other three estimators. θ is (ρ, σ\_ε²) for the AR(1) fit and (ρ, σ\_ε², σ\_ν²) with measurement error; either way G has one T × T block per parameter, and the standard error of an SD follows by the delta method, se(σ̂) = se(σ̂²) / (2σ̂).

## Design grid

Baseline: N = 500, R = 10 replications per design point for the local pilot (scaled to 1,000 on the HPC later), T varied to trace out the Nickell bias.

| Dimension | Values | Why |
| --- | --- | --- |
| T | 3, 5, 10, 20, 50 | Within bias is of order 1/T; the others do not vanish in T |
| N | 500 (robustness: 5,000) | Larger N confirms the biases are inconsistency, not finite-sample noise |
| Model | M0, M1, M2 | With and without α\_i, and with α\_i but measuring y with error |
| R | 10 (pilot); 1,000 on HPC | At R = 10 the Monte Carlo SE of the mean bias is about SD/3, enough to check the code; SD/32 at R = 1,000 |

Total: 5 values of T × 3 models × R draws, with all 5 estimators run on the same draw. Share draws across estimators within a replication so differences between estimators are not Monte Carlo noise.

Seeds: one master seed; replication r in design (model, T, N) gets a seed derived deterministically from those four values, so any single cell can be rerun on its own.

## Analytical benchmarks

The simulated means should match these large-N probability limits; overlay them on every figure as a check on the code.

| Estimator | M0 | M1 | M2 | Source of bias |
| --- | --- | --- | --- | --- |
| Pooled OLS | 0.700 | 0.982 | 0.969 | Lag correlated with α\_i |
| First-difference OLS | −0.150 | −0.150 | −0.301 | Δy\_i,t−1 correlated with Δε\_it through ε\_i,t−1 |
| Within | Nickell, see below | Nickell, same as M0 | Nickell **and** attenuation, see below | Demeaned lag correlated with the mean of ε |
| Growth-covariance GMM | 0.700 | 0.700 | 0.370 | None in M0 and M1; in M2 it fits the wrong Ω |
| Growth-covariance GMM, with ME | 0.700 | 0.700 | 0.700 | None |

For the variance parameters: σ̂\_ε has plim 0.300 everywhere except the AR(1) fit under M2, where it absorbs the noise it has no other place for and settles at 0.354; σ̂\_ν has plim 0.200 in M2 and 0 in M0 and M1.

### One set of formulas for all three models

Every model here is y\_it = ℓ\_i + v\_it with ℓ\_i time-invariant and v stationary and mean zero. Write σ\_ℓ² for the variance of the level and γ(k) for the autocovariance of v:

```latex
\sigma_\ell^2 = \begin{cases}\sigma_\alpha^2/(1-\rho)^2 & \text{M0, M1}\\ \sigma_\alpha^2 & \text{M2}\end{cases}
\qquad
\gamma(k) = \frac{\sigma_\varepsilon^2 \rho^{k}}{1-\rho^2} + \sigma_\nu^2 \mathbf{1}\{k = 0\}
```

Measurement error is white, so it lands on γ(0) alone and leaves every other lag untouched. That one spike is the whole difference between M1 and M2, and it is enough to move all three regressions:

| Estimator | plim, in terms of σ\_ℓ² and γ |
| --- | --- |
| Pooled OLS | (σ\_ℓ² + γ(1)) / (σ\_ℓ² + γ(0)) |
| First-difference OLS | (2γ(1) − γ(0) − γ(2)) / (2γ(0) − 2γ(1)) |
| Within | trace(Q X Γ Y′) / trace(Q X Γ X′), with Γ the (T+1) × (T+1) matrix of γ, X and Y selecting v\_0…v\_{T−1} and v\_1…v\_T, and Q demeaning over the T periods |
| Both growth fits | whatever their own criterion returns at the population Ω |

The closed forms below are the M0/M1 special case of the first three; the code runs the general versions and the tests pin them to the closed forms wherever both are defined. The growth fits get no formula at all: their plim is computed by running the estimator's own criterion on the population Ω, which is exact rather than approximate and is also the only honest way to state a pseudo-true value.

**Pooled OLS in M1.** Write y\_i,t−1 = μ\_i + u\_i,t−1 with μ\_i = α\_i/(1−ρ). Under stationarity:

```latex
\operatorname{plim} \hat\rho_{OLS} = \frac{\sigma_\mu^2 + \rho\,\sigma_u^2}{\sigma_\mu^2 + \sigma_u^2}, \qquad \sigma_\mu^2 = \frac{\sigma_\alpha^2}{(1-\rho)^2},\; \sigma_u^2 = \frac{\sigma_\varepsilon^2}{1-\rho^2}
```

**First differences.** Cov(Δy\_i,t−1, Δε\_it) = −σ\_ε² and Var(Δy\_i,t−1) = 2σ\_ε²/(1+ρ), so the plim is ρ − (1+ρ)/2 = (ρ−1)/2. It does not depend on T, σ\_ε or σ\_α — but it does depend on σ\_ν, which is the point of M2: the general formula gives −0.301 there, twice the bias.

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

**Within in M2.** Measurement error puts attenuation on top of the Nickell bias, and unlike the Nickell bias it does not vanish in T: as T grows the demeaning stops mattering and the plim tends to γ(1)/γ(0) = ρ σ\_p²/(σ\_p² + σ\_ν²) = 0.571, not ρ.

| T | 3 | 5 | 10 | 20 | 50 |
| --- | --- | --- | --- | --- | --- |
| Within plim, M2 | −0.077 | 0.144 | 0.347 | 0.461 | 0.528 |

**The growth fits.** Each is consistent exactly where the shape it fits is the shape that generated Ω. The moment conditions E[Δy\_it Δy\_is] = Ω\_ts(θ) hold exactly under the stationary initial condition, Ω is free of α\_i, and θ ↦ Ω(θ) is injective — for the AR(1) fit at T ≥ 2, for the measurement-error fit at T ≥ 3.

- In M0 and M1 both are consistent, and their plim lines on Figure 1 are flat at 0.700. Growth in M1 has the same distribution as growth in M0, so these two estimators and the FD and within estimators have identical sampling distributions across those two models; only the seed differs.
- In M2 the AR(1) fit is misspecified. It has no σ\_ν² to put the noise in, so it loads it onto σ\_ε² and buys the resulting shape mismatch by lowering ρ: the pseudo-true pair is (0.370, 0.354), near enough T-free from T = 5 on. It is the cleanest failure in the whole grid — the estimator that was the reference in M0 and M1 is, with one extra term in the DGP and none in the model, worse than the within estimator at T = 50.

## Outputs

Per (model, T, N, estimator) cell, compute over the R replications:

- **Mean bias**: mean of ρ̂ − 0.7, with its Monte Carlo SE.
- **SD**: standard deviation of ρ̂ across replications.
- **RMSE**: square root of mean (ρ̂ − 0.7)².
- **Coverage**: share of 95% CIs containing 0.7.
- **Mean clustered SE / SD**: checks whether the reported SE matches actual dispersion.

The same five statistics are computed for each variance parameter an estimator reports, against that parameter's true value in that model: σ\_ε = 0.3 everywhere, σ\_ν = 0.2 in M2 and 0 in M0 and M1. Estimators that report no variance leave those columns empty. A replication whose variance was pinned at zero has no usable interval and drops out of that parameter's coverage and mean SE rather than being counted either way.

Deliverables:

1. Main table: rows = estimator × model, columns = T, cells = mean ρ̂ (SD), with the analytical plim beside each.
2. Figure 1: mean ρ̂ against T, one line per estimator, one panel per model, dashed plims and a horizontal line at 0.7.
3. Figure 2: density of ρ̂ at T = 10 for the 15 cells, one panel per model, to show that the biased estimators are tightly centred on the wrong value while the consistent ones sit on 0.7.
4. Coverage table: same layout as the main table.
5. Variance table: mean σ̂ (SD) [plim] and coverage, one block per (parameter, estimator, model), columns = T. Only the growth fits appear.
6. Empirical growth-covariance table: the 5 × 5 matrix of family earnings growth as a LaTeX table, from the companion section below.

## Empirical companion: family earnings growth

The growth-covariance GMM consumes exactly one object, the variance–covariance matrix of Δy. Everything above feeds it a matrix this spec generated. This section builds the same object from real household earnings, so there is one matrix in the project that the DGP did not produce, and so the Ω restrictions can be looked at against data that has no obligation to satisfy them.

The source is the replication package for Blundell, Pistaferri and Saporta-Eksten (2016), *Consumption Inequality and Family Labor Supply*, AER 106(2) 387–435 — PSID waves 1999–2009, a panel of married couples with both earners' annual earnings.

### Acquiring the archive

The data link on the AEA article page redirects to openICPSR (doi:10.3886/E116142V1), which requires an account. The original archive is still served, unauthenticated, from AEA's own asset host, which is what the prep step fetches:

```
https://assets.aeaweb.org/asset-server/articles-attachments/aer/data/10602/20121549_data.zip
```

| Property | Value |
| --- | --- |
| Size | 5,736,291 bytes; 51 files, 16 MB extracted |
| SHA-256 | `a3cc2b7cf60dddff9a8d02612d6c15b90c79093db0f91ddecb3db11c7542e28c` |
| Lands in | `data/bps2016/` (gitignored — the checksum, not the bytes, is what the repo carries) |

The prep step is idempotent: it verifies the checksum and re-downloads only on mismatch or absence. The archive ships the built estimation file `output/data4estimation.dta` (10,479 observations, 3,008 households, 6 waves), so no raw PSID extract is needed. It does **not** ship `data4estimation_nopart.dta` or `data4sample3.dta`, so nothing here can use the paper's wider samples.

### Variable and sample

**Family earnings** are head plus spouse annual earnings, deflated: the file's `log_toty` equals log(`ly` + `wly`) − log(`price`) to 1e−6, and is used directly. Growth is the biennial log change under `tsset person year`, where `year` advances in steps of two; each change is dated at its end point, giving Δ2000 … Δ2008.

**Balance.** Six waves of levels yield five growth rates, so the balanced panel is T = 5, not 6. Households are kept when family earnings are non-missing in all six waves: **695** households for the raw series, **692** after the first stage below. This is the sample dimension the GMM section calls N; the spec's Ω is T × T with T = 5 here.

Balancing selects stable households and the spec should not pretend otherwise. Pooled over waves, the variance of the biennial change is 0.1538 on all households and 0.1356 on the balanced ones — about 12% of the dispersion is the balance restriction, before any household enters the matrix.

**Scale check against the published paper.** The same pipeline gives 0.2626 for the variance of head-only earnings growth on all households; √0.263 = 0.513 is the figure printed in the paper's Table 1, which is how this prep step is validated. Family earnings are much smoother than head earnings because two earners pool.

### First stage

BPS do not use raw growth; they use the residual from a projection of growth on demographics, and the matrix inherits that convention. Following the `dlog_toty` regression in `residual_measures.do`: year, cohort (head and spouse), education (head and spouse), race, family size, number of children, employment status, state, out-of-household kids, metropolitan status and an other-income indicator, entered in **both levels and differences**, plus the education, race, employment and metropolitan interactions with year. 7,063 observations, 336 columns, rank 256, R² = 0.066.

Two implementation points, both load-bearing:

- **Order.** Residualize on the full estimation sample, then restrict to the balanced households. Residualizing within the balanced panel would absorb variation the estimator is supposed to see.
- **Parameterization is irrelevant.** OLS residuals depend only on the column space of the design, so collinear dummy sets are kept whole and resolved by least squares, rather than reproducing whichever reference categories Stata's `xi` happens to drop. The rank deficiency (336 → 256) is expected and is not an error to fix.

Both the raw and the residualized matrices are produced; the residualized one is the deliverable.

### The matrix

Covariances of residualized biennial log family earnings growth, 692 balanced households:

| | Δ2000 | Δ2002 | Δ2004 | Δ2006 | Δ2008 |
| --- | --- | --- | --- | --- | --- |
| **Δ2000** | **0.1035** | −0.0353 | 0.0025 | −0.0067 | 0.0013 |
| **Δ2002** | −0.0353 | **0.1449** | −0.0800 | 0.0089 | 0.0007 |
| **Δ2004** | 0.0025 | −0.0800 | **0.1511** | −0.0476 | 0.0000 |
| **Δ2006** | −0.0067 | 0.0089 | −0.0476 | **0.1089** | −0.0378 |
| **Δ2008** | 0.0013 | 0.0007 | 0.0000 | −0.0378 | **0.1424** |

Band means: 0.1302 on the diagonal, −0.0502 at lag 1, then 0.0038, −0.0030, 0.0013.

### What it says about Ω

Two things the simulated matrices cannot show:

**The empirical matrix is not Toeplitz.** The spec's Ω is, by stationarity — one value per band. The diagonal here runs 0.104 to 0.151, a factor of 1.5 across five waves, because real earnings risk moves with the business cycle. Any minimum-distance fit is therefore matching a Toeplitz surface to a non-Toeplitz target, and the residual criterion is not pure sampling noise.

**The ratio statistic is the same number under two incompatible readings.** Under the spec's AR(1), Ω\_{t,t−1}/Ω\_tt = (ρ−1)/2, so ρ = 1 + 2·(lag-1 band)/(diagonal). Under the permanent-plus-transitory model standard in this literature, Δy\_it = ζ\_it + ε\_it − ε\_i,t−1, the permanent share σ²\_ζ/Var(Δy) equals **that same expression**. Here both are 0.229. The two models are separated not by the ratio but by the higher bands: AR(1) requires geometric decay ρ^{|t−s|−1}, permanent-plus-transitory requires exactly zero.

The data side with the second. Fitting the spec's Ω by the criterion in the GMM section gives ρ̂ = 0.111 and σ̂\_ε = 0.264 — the full-matrix fit is pulled well below the ratio-implied 0.229 because it is also trying to flatten the higher bands — and it still wants −0.0062 at lag 2 where the data show +0.0038. The criterion at the optimum is 0.00564 against ‖C‖²\_F = 0.10964.

This is the empirical case for the extension already listed below: a stationary AR(1) in levels has one parameter doing two jobs, and household earnings need a unit-root component beside a transitory one.

### The LaTeX table

Deliverable 6, generated rather than hand-maintained, so it moves when the numbers move.

| Requirement | Why |
| --- | --- |
| `booktabs` the only package | No `siunitx`, no `threeparttable`; the table drops into any document |
| Decimal alignment via `\phantom{-}` on positives | Right-aligned `r` columns otherwise misalign signed and unsigned entries |
| Diagonal in `\mathbf` | The variances are what the reader looks up first |
| Notes measured to the table, not the page | Set the tabular into `\sbox0`, then the notes in a `minipage{\wd0}`. Putting the notes in a `\multicolumn` *inside* the tabular forces the tabular to the notes' width and dumps all the slack into the last column — the failure mode that produced the first draft |
| Notes state the sample, the first stage and the source | The table is meant to travel out of this repo |

Compile check: the table must build under `pdflatex` with `booktabs` alone, from a wrapper that `\input`s the generated file, so what is verified is the artifact and not a copy of it.

## Implementation notes and extensions

**Tooling.** Python, managed with uv; tool versions and tasks pinned with mise.

- `mise.toml` pins `python` and `uv` under `[tools]`; the task graph lives in the Makefile below. The empirical table compiles today against whatever `pdflatex` is on PATH; pinning a TeX engine in `mise.toml` is still open, and until it is done `tables` stays off the default target.
- `uv init` project with `pyproject.toml` and a committed `uv.lock`; dependencies: numpy, pandas, matplotlib, pytest (dev).
- One entry point, `uv run python -m sim run --model M1 --T 10 --N 500 --R 10 --seed 0`, writing one parquet file per cell. The HPC scale-up (on hold for now) reuses this same command, so nothing in the code depends on where it runs.
- Summary and figures read all parquet files in the output folder, so pilot and HPC outputs share one pipeline.

**Makefile.** A repo-root `Makefile` holds every task and its dependencies. Simulation outputs are file targets (one parquet per cell), so `make -j` runs cells in parallel and reruns only what is stale.

| Target | Depends on | Does |
| --- | --- | --- |
| `setup` | `pyproject.toml`, `uv.lock` | `uv sync` (assumes `mise install` has been run) |
| `test` | `setup` | `uv run pytest`, including the plim unit test |
| `output/pilot/<model>_T<T>_N<N>.parquet` | `test`, `src/sim/*.py` | Pattern rule: one cell at R = 10 via the entry point |
| `sim` | all 15 cell files (3 models × 5 values of T) | Phony target collecting the pilot cells |
| `summary` | `sim` | Bias, SD, RMSE and coverage tables to `output/summary.csv`, for ρ̂ and for each variance parameter |
| `figures` | `summary` | Figures 1 and 2 to `output/figures/` |
| `tables` | `summary` | LaTeX tables (on hold, with the TeX engine) |
| `data/bps2016/20121549_data.zip` | none | Fetches the BPS archive from the AEA asset host and verifies the SHA-256; no-op when the file is present and matches |
| `data/bps2016/AER_2012_1549_data/` | the zip | Unpacks it |
| `data/bps2016/vcov_family_earnings_growth_residual.{csv,tex}` | the unpacked archive, `scripts/bps_vcov.py` | Builds the balanced panel, runs the first stage, writes the matrix and the LaTeX table |
| `empirical` | the two files above | Phony target for the companion section |
| `all` | `figures`, `summary` | Default target |
| `clean` | none | Removes `output/`; leaves `data/` alone, since re-downloading the archive is the slow step |
| check | SPEC.md, all outputs | Verifies each output's spec hash and git commit match the current ones, and that make -q all reports nothing stale |

The design grid (models, T, N, R) is defined once as Make variables at the top, so the HPC run can override R and N from the command line.

**HPC (on hold).** When we scale up, runs go through [scripthut](https://github.com/tlamadon/scripthut) rather than hand-written Slurm scripts. The plan, to be revisited then:

- A workflow file in `.hut/workflows/` whose tasks are one per (model, T, N) cell, with dot-notation ids such as `sim.M1.T10`, each calling the entry point above with R = 1,000.
- A final `summary` task with `deps: ["sim.*"]` that builds the tables and figures once every cell finishes.
- The cluster environment built from the same `uv.lock`, so pilot and HPC use identical package versions.

* Simulate the whole panel as an N × (T+1) array, looping over t only; each replication is then a few vectorised operations.
* Code the three regression estimators as closed-form ratios of sums (Σxy / Σx²) rather than calling a regression package. This is faster and makes the moment conditions explicit.
* The growth fits are the exception, and they still need no new dependency: the criterion profiles down to one parameter whichever shape is fitted, and the sample moment matrix is a single T × T Gram matrix, C = ΔY′ΔY / N. Never build the N × T² matrix of per-individual moments; the sandwich needs only one N-vector per parameter, Δy\_i′ (∂Ω/∂θ\_p) Δy\_i, each one matmul.
* Both shapes are Toeplitz, so the criterion sees C only through its T band sums. That is what keeps a 1,001-point search over ρ cheap at T = 50, and it is worth computing the bands once rather than rebuilding a T × T matrix per grid point.
* Put Ω, the two shapes and the criterion in one module that both the estimators and the plims import. The pseudo-true values are then the same code as the estimates, run on a population matrix instead of a sample one, and cannot drift apart from them.
* Unit test before running the grid: at N = 10⁶ and T = 10, each estimator should land within 0.005 of its plim in the benchmark table, and each variance within 0.005 of its own. Also check that feeding a population Ω(θ) in place of C returns θ back, to the ~10⁻⁸ a derivative-free maximum of a smooth criterion allows, for both shapes.
* Save raw ρ̂ and SE draws for every replication, plus every variance and its SE where the estimator produces one, so new summary statistics need no rerun.

Optional extensions, one line each:

- Add the levels moment Var(y\_it) to the growth moments, which would identify σ\_α as well; growth alone cannot see it.
- Let σ\_ν vary over a grid rather than sitting at one value, to trace how much measurement error the AR(1) fit tolerates before the within estimator overtakes it.
- Add Anderson–Hsiao IV (Δy\_i,t−1 instrumented by y\_i,t−2) as a second consistent reference, one that uses the differenced data a single lag at a time rather than all of it at once.
- Use the efficient GMM weight where it is feasible (T(T+1)/2 < N, so T ≤ 30 at N = 500) and report how much of the GMM standard error the identity weight costs.
- Replace the stationary initial condition with y\_i0 = 0 to show how pooled OLS and within results move when the start-up is not in steady state.
- Vary ρ ∈ {0.3, 0.7, 0.95} to show the within bias worsening as persistence rises.

## Spec-check skill

A Claude Code skill in the repo, `.claude/skills/spec-check/SKILL.md`, audits the project against this spec and reports pass or fail per check. It fixes nothing unless asked.

It reads a copy of this spec committed as `SPEC.md`, exported from the doc whenever the doc changes. It runs three checks, in order:

1. **Spec is internally consistent.** It recomputes every plim in the benchmark tables from the stated formulas and parameters, and checks that the grid counts (3 models × 5 values of T) match the Makefile description.
2. **Code matches the spec.** It checks the DGP (ρ, σ\_ε, σ\_α, σ\_ν, the stationary initial condition, and that M2's effect is a level while M0's and M1's is an intercept), the five estimator definitions (periods used, demeaning, clustering, and for the growth fits the two Ω shapes, the identity weight and the concentrated variances), the grid and seeding, the output metrics, and that every Makefile target and dependency in the spec table exists as written (`make -pn`).
3. **Outputs are built from current code and spec.** Each parquet file records the SHA-256 of `SPEC.md` and the git commit in its metadata. The skill flags any output whose hashes differ from the current ones, and requires `make -q all` to report nothing stale.

The mechanical parts of check 3 live in a `make check` target so they also run without Claude; the skill calls it and adds the judgement-based checks 1 and 2.

4. **The empirical companion reproduces.** Its outputs live under gitignored `data/`, so they carry no parquet metadata and check 3 cannot see them. They are verified differently: the archive's SHA-256 against the value in the companion section, and the pipeline against the paper itself — head earnings growth variance 0.2626, whose root rounds to the 0.513 printed in the paper's Table 1. A prep step that silently changed the sample would move that number. The check skips rather than fails when the archive has not been downloaded.

## To do

- [ ] Make the project a git repo: `git init`, a `.gitignore` for `.venv/` and `output/`, and a first commit with `mise.toml`, `pyproject.toml`, `uv.lock` and the `Makefile`. Scripthut clones from git, so this is also a prerequisite for the HPC runs.
- [ ] Write the spec-check skill and the `make check` target, and commit this spec as `SPEC.md`.
- [ ] Fold `scripts/bps_vcov.py` into the Makefile as the `empirical` targets above; it currently runs by hand and re-does the download check and the first stage on every invocation.
- [ ] Pin a TeX engine in `mise.toml` and turn the compile check into a `tables` target, so the LaTeX is verified rather than assumed.
- [ ] Decide whether the empirical matrix stays a companion or becomes a fifth "estimator × model" column: running the growth-covariance GMM on it is three lines, but interpreting ρ̂ = 0.111 needs the levels moment from the extensions list.
