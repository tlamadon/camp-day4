# The variance–autocovariance matrix of growth, and its Lean proof

The simulation in `SPEC.md` uses two moments of `Δy_it` in passing (its variance
and its first autocovariance) to get the first-difference plim.  This note
derives the **whole** matrix and checks every step with Lean 4 + Mathlib in
`PanelAR1/GrowthACov.lean`.

## The model

For i = 1, …, N and t = 1, …, T, as in `SPEC.md`,

```
y_it = α_i + ρ y_i,t-1 + ε_it,      ε_it ~ iid N(0, σ_ε²),
y_i0 = μ_i + u_i0,   μ_i = α_i/(1-ρ),   u_i0 ~ N(0, σ_ε²/(1-ρ²)),
```

with α_i ⊥ ε, |ρ| < 1, and everything independent across i.  M0 is α_i ≡ 0;
M1 has α_i ~ N(0, σ_α²).  Growth is `Δy_it = y_it - y_i,t-1`, for t = 1, …, T —
all T of them, since y_i0 is observed.

## Derivation

**1. Centre the process.**  With μ_i = α_i/(1-ρ), i.e. α_i = (1-ρ)μ_i, put
u_it = y_it - μ_i.  Then

```
u_it = y_it - μ_i = α_i + ρ(u_i,t-1 + μ_i) + ε_it - μ_i = ρ u_i,t-1 + ε_it,
```

a zero-mean AR(1) with the same ρ.  (Lean: `centered_recursion`.)

**2. Growth never sees the individual effect.**  μ_i is time-invariant, so

```
Δy_it = (μ_i + u_it) - (μ_i + u_i,t-1) = Δu_it.
```

α_i cancels *exactly*, whatever its distribution.  Hence the matrix below is
identical in M0 and M1, and does not involve σ_α.  (Lean:
`growth_eq_centered_growth`.)

**3. Innovation representation.**  Unrolling the recursion,

```
u_it = ρ^t u_i0 + Σ_{j=1}^{t} ρ^{t-j} ε_ij,
```

so u_it is a linear combination of the uncorrelated innovations
ξ_0 = u_i0, ξ_j = ε_ij with variances σ_ε²/(1-ρ²) and σ_ε².  (Lean: `load`,
and `innovation_representation`, which checks that these loadings really do
solve the AR(1) recursion.)

**4. Levels are stationary.**  Because the start-up draw has exactly the
stationary variance,

```
Var(u_it) = ρ^{2t} σ_ε²/(1-ρ²) + σ_ε² Σ_{j=1}^{t} ρ^{2(t-j)}
          = σ_ε² [ρ^{2t} + 1 - ρ^{2t}]/(1-ρ²) = σ_ε²/(1-ρ²) ≡ σ_u²,
```

for every t, and, projecting u_it on u_i,t-h,

```
γ_u(h) ≡ Cov(u_it, u_i,t-h) = ρ^h σ_u² = σ_ε² ρ^h/(1-ρ²),   h ≥ 0.
```

(Lean: `cov_diag` for stationarity, `cov_closed` for γ_u.)

**5. Difference it.**  For t ≥ s, with h = t - s, bilinearity of covariance gives

```
Cov(Δy_it, Δy_is) = Cov(u_it - u_i,t-1, u_is - u_i,s-1)
                  = 2γ_u(h) - γ_u(h+1) - γ_u(h-1).
```

(Lean: `gcov_expand`.)  Evaluating with γ_u(h) = σ_u² ρ^h:

* **h = 0:** `2σ_u²(1 - ρ) = 2σ_ε²/(1+ρ)` — since σ_u² = σ_ε²/((1-ρ)(1+ρ)).
* **h ≥ 1:** `σ_u² ρ^{h-1}(2ρ - ρ² - 1) = -σ_u² ρ^{h-1}(1-ρ)²
  = -σ_ε² ρ^{h-1}(1-ρ)/(1+ρ)`.

(Lean: `gcov_self` and `gcov_lag`.)

## The matrix

E[Δy_it] = 0, growth is independent across i, and the T × T matrix
Ω = Var(Δy_i1, …, Δy_iT) is Toeplitz:

```
             ⎧  2σ_ε²/(1+ρ)                        t = s
Ω_ts  =      ⎨
             ⎩ -σ_ε² (1-ρ)/(1+ρ) · ρ^{|t-s|-1}     t ≠ s
```

or, factored,

```
Ω = σ_ε²/(1+ρ) · [ 2 I_T − (1−ρ) K ],     K_ts = ρ^{|t-s|-1} 1{t ≠ s}.
```

(Lean: `gcov_toeplitz`; as a matrix, `Omega` and `Omega_apply`.)  It is free of
α_i, of σ_α and of T — the T × T matrix is the leading block of the (T+1) ×
(T+1) one.  Written out:

```
        σ_ε²   ⎡  2        -(1-ρ)      -(1-ρ)ρ     -(1-ρ)ρ²   ⋯ ⎤
Ω =   ------- ⎢ -(1-ρ)      2          -(1-ρ)      -(1-ρ)ρ    ⋯ ⎥
        1+ρ   ⎢ -(1-ρ)ρ    -(1-ρ)       2          -(1-ρ)     ⋯ ⎥
              ⎣    ⋮          ⋮            ⋮            ⋮        ⎦
```

### What it says

* **ARMA(1,1).**  `(1 - ρL) Δy_it = (1 - L) ε_it`, so past lag 1 the
  autocovariances decay geometrically at rate ρ: γ_Δ(h+1) = ρ γ_Δ(h) for h ≥ 1.
  (Lean: `gcov_decay`.)
* **Over-differencing.**  Every off-diagonal is negative (for 0 < ρ < 1) and
  the autocovariances sum to zero,
  `γ_Δ(0) + 2 Σ_{h≥1} γ_Δ(h) = 2σ_ε²/(1+ρ) - 2σ_ε²/(1+ρ) = 0`: differencing a
  stationary series puts a unit root in the MA part, so the spectral density of
  Δy vanishes at frequency zero.  Ω itself is still positive definite — it is
  `D Var(u_i0, …, u_iT) Dᵀ` for the full-row-rank differencing matrix D.  (Lean:
  `Omega_quadForm_nonneg` proves positive semidefiniteness, by exhibiting the
  quadratic form as an innovation-weighted sum of squares.)
* **Limits.**  ρ → 1 gives Ω → σ_ε² I (a random walk differences to white
  noise); ρ = 0 gives the MA(1) matrix with 2σ_ε² on the diagonal and -σ_ε²
  next to it.
* **The first-difference plim of `SPEC.md`** falls out of the first two entries:

  ```
  plim ρ̂_FD = Ω_{t,t-1} / Ω_{t-1,t-1} = [-σ_ε²(1-ρ)/(1+ρ)] / [2σ_ε²/(1+ρ)] = (ρ-1)/2,
  ```

  free of σ_ε, T and α — at ρ = 0.7 that is −0.15, the number in the benchmark
  table.  (Lean: `fd_plim`.)

At the spec's ρ = 0.7, σ_ε = 0.3: Var(Δy) = 0.10588, γ_Δ(1) = −0.01588, and
γ_Δ(h) = −0.01588 · 0.7^{h-1} for h ≥ 1.

## How it is verified

`PanelAR1/GrowthACov.lean` (Lean 4 + Mathlib) proves all of the above.
Covariance is formalised as the innovation-variance-weighted inner product of
loading vectors — which is what the covariance of linear combinations of
uncorrelated innovations *is* — and `innovation_representation` proves those
loadings solve the model's recursion, which ties the algebra to the DGP in
`SPEC.md`.  No `sorry`, and no axioms beyond Mathlib's.

```
mise install          # provisions elan (and hence lean/lake), python, uv
make proof            # lake build, from the repo root
```

`numeric_check.py` is the independent cross-check: it draws panels from the
project's own `sim.dgp.simulate_panel` and compares the empirical covariance
matrix of Δy with the formula above, for both M0 and M1.  `make proof-numeric`
runs it.
