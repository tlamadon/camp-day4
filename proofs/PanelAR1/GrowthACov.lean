/-
# The variance–autocovariance matrix of growth in the panel AR(1)

Companion to `SPEC.md`.  The data-generating process there is, for each `i`,

    y t = α + ρ * y (t-1) + ε t,   ε t iid N(0, σ²),
    y 0 = α / (1 - ρ) + u 0,       u 0 ~ N(0, σ² / (1 - ρ²)),

and this file derives, and checks, the full autocovariance matrix of *growth*
`Δy t = y t - y (t-1)` for `t = 1, …, T`:

    Ω t s = 2σ²/(1+ρ)                      if t = s,
          = -σ²(1-ρ)/(1+ρ) * ρ^(|t-s|-1)   otherwise.

## How the probability is modelled

`Δy` is a linear combination of the uncorrelated innovations
`ξ 0 = u 0, ξ 1 = ε 1, ξ 2 = ε 2, …` with variances
`ivar 0 = σ²/(1-ρ²)` and `ivar j = σ²` for `j ≥ 1`.  For such linear
combinations the covariance *is* the `ivar`-weighted inner product of the
loading vectors, so that weighted inner product (`cov`, `gcov`) is what is
formalised below.  Section 2 proves that the loadings used here really do solve
the AR(1) recursion, which is what ties the algebra back to the model above.
-/
import Mathlib

set_option autoImplicit false

namespace PanelAR1

open Finset

variable (ρ σ : ℝ)

/-! ## 1. Reduction of the model to a centred AR(1)

Two algebraic steps: recentring at the long-run mean `μ = α/(1-ρ)` turns the
model into a zero-mean AR(1), and growth does not see `μ` at all.  The second
lemma is why `Ω` is the same in M0 (α = 0) and M1 (α ~ N(0, σ_α²)): whatever the
individual effect is, and however it is distributed, it cancels from `Δy`. -/

/-- Recentring: if `y` follows the AR(1) with intercept `α`, then
`u t = y t - α/(1-ρ)` follows the same AR(1) without intercept. -/
lemma centered_recursion (α : ℝ) (hρ : (1 : ℝ) - ρ ≠ 0) (y ε : ℕ → ℝ)
    (hy : ∀ t, y (t + 1) = α + ρ * y t + ε (t + 1)) (t : ℕ) :
    y (t + 1) - α / (1 - ρ) = ρ * (y t - α / (1 - ρ)) + ε (t + 1) := by
  rw [hy t]
  field_simp
  ring

/-- The individual effect drops out of growth: only the centred process matters. -/
lemma growth_eq_centered_growth (μ : ℝ) (y u : ℕ → ℝ) (h : ∀ t, y t = μ + u t) (t : ℕ) :
    y (t + 1) - y t = u (t + 1) - u t := by
  rw [h (t + 1), h t]; ring

/-! ## 2. Innovation representation of the centred process -/

/-- Loading of innovation `j` on `u t`, i.e. `u t = ∑ j, load ρ t j * ξ j`.
Innovation `0` is the stationary start-up draw `u 0` (loading `ρ^t`); innovation
`j ≥ 1` is `ε j` (loading `ρ^(t-j)`, and `0` before it has happened). -/
noncomputable def load (ρ : ℝ) (t j : ℕ) : ℝ :=
  if j = 0 then ρ ^ t else if j ≤ t then ρ ^ (t - j) else 0

lemma load_zero_zero : load ρ 0 0 = 1 := by simp [load]

/-- Innovations in the future carry no loading. -/
lemma load_of_lt {t j : ℕ} (h : t < j) : load ρ t j = 0 := by
  have h0 : j ≠ 0 := by omega
  have h1 : ¬ j ≤ t := by omega
  simp [load, h0, h1]

/-- One step of the AR(1), loading by loading. -/
lemma load_succ (t j : ℕ) :
    load ρ (t + 1) j = ρ * load ρ t j + (if j = t + 1 then 1 else 0) := by
  rcases Nat.eq_zero_or_pos j with hj | hj
  · subst hj
    have : (0 : ℕ) ≠ t + 1 := by omega
    simp [load, this, pow_succ]
    ring
  · rcases lt_trichotomy j (t + 1) with h | h | h
    · have hjt : j ≤ t := by omega
      have hj1 : j ≤ t + 1 := by omega
      have hne : ¬ (j = t + 1) := by omega
      have hj0 : ¬ (j = 0) := by omega
      have hsub : t + 1 - j = (t - j) + 1 := by omega
      simp only [load, if_neg hj0, if_pos hj1, if_pos hjt, if_neg hne, add_zero, hsub]
      rw [pow_succ]; ring
    · subst h
      have hj0 : ¬ (j = 0) := by omega
      simp [load, hj0, load_of_lt ρ (show t < j by omega)]
    · have h1 : ¬ (j ≤ t + 1) := by omega
      have hne : ¬ (j = t + 1) := by omega
      rw [load_of_lt ρ (show t + 1 < j by omega), load_of_lt ρ (show t < j by omega)]
      simp [hne]

/-- **The loadings solve the model.**  For *any* innovation sequence `ξ`, the
process `u t = ∑ j, load ρ t j * ξ j` satisfies `u (t+1) = ρ * u t + ξ (t+1)`.
This is the link between the algebra below and the AR(1) of `SPEC.md`. -/
theorem innovation_representation (ξ : ℕ → ℝ) (t : ℕ) :
    ∑ j ∈ range (t + 2), load ρ (t + 1) j * ξ j
      = ρ * (∑ j ∈ range (t + 2), load ρ t j * ξ j) + ξ (t + 1) := by
  have key : ∀ j ∈ range (t + 2),
      load ρ (t + 1) j * ξ j
        = ρ * (load ρ t j * ξ j) + (if j = t + 1 then ξ j else 0) := by
    intro j _
    rw [load_succ]
    by_cases h : j = t + 1 <;> simp [h] <;> ring
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, ← Finset.mul_sum]
  simp

/-! ## 3. Covariance of the levels

`ivar j` is the variance of innovation `j`; `cov ρ σ t s` is the covariance of
`u t` and `u s`, i.e. the `ivar`-weighted inner product of their loadings.  The
truncation `covAux … n` is only bookkeeping: all loadings past `t` vanish, so the
value does not depend on `n` once `n > t`. -/

noncomputable def ivar (ρ σ : ℝ) (j : ℕ) : ℝ :=
  if j = 0 then σ ^ 2 / (1 - ρ ^ 2) else σ ^ 2

noncomputable def covAux (ρ σ : ℝ) (t s n : ℕ) : ℝ :=
  ∑ j ∈ range n, ivar ρ σ j * load ρ t j * load ρ s j

/-- Covariance of `u t` and `u s`. -/
noncomputable def cov (ρ σ : ℝ) (t s : ℕ) : ℝ := covAux ρ σ t s (max t s + 1)

lemma covAux_extend {t s n m : ℕ} (ht : t < n) (hnm : n ≤ m) :
    covAux ρ σ t s n = covAux ρ σ t s m := by
  refine Finset.sum_subset (Finset.range_subset.mpr hnm) ?_
  intro x _ hx
  rw [Finset.mem_range] at hx
  rw [load_of_lt ρ (show t < x by omega)]
  ring

lemma cov_eq_covAux (t s n : ℕ) (ht : t < n) (hs : s < n) :
    cov ρ σ t s = covAux ρ σ t s n :=
  covAux_extend ρ σ (by omega) (by omega)

lemma cov_symm (t s : ℕ) : cov ρ σ t s = cov ρ σ s t := by
  unfold cov covAux
  rw [max_comm]
  exact Finset.sum_congr rfl fun j _ => by ring

lemma cov_zero_zero : cov ρ σ 0 0 = σ ^ 2 / (1 - ρ ^ 2) := by
  simp [cov, covAux, ivar, load]

/-- The lag recursion `γ(h+1) = ρ γ(h)`: one more step on the left multiplies the
covariance by `ρ`, because the new shock `ε (t+1)` is orthogonal to `u s`. -/
lemma cov_succ_left (t s : ℕ) (hst : s ≤ t) :
    cov ρ σ (t + 1) s = ρ * cov ρ σ t s := by
  rw [cov_eq_covAux ρ σ (t + 1) s (t + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ t s (t + 2) (by omega) (by omega)]
  unfold covAux
  have key : ∀ j ∈ range (t + 2),
      ivar ρ σ j * load ρ (t + 1) j * load ρ s j
        = ρ * (ivar ρ σ j * load ρ t j * load ρ s j)
          + (if j = t + 1 then ivar ρ σ j * load ρ s j else 0) := by
    intro j _
    rw [load_succ]
    by_cases h : j = t + 1 <;> simp [h] <;> ring
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [Finset.sum_ite_eq' (range (t + 2)) (t + 1) (fun j => ivar ρ σ j * load ρ s j)]
  simp [Finset.mem_range, load_of_lt ρ (show s < t + 1 by omega)]

/-- The variance recursion `Var(u (t+1)) = ρ² Var(u t) + σ²`. -/
lemma cov_succ_diag (t : ℕ) :
    cov ρ σ (t + 1) (t + 1) = ρ ^ 2 * cov ρ σ t t + σ ^ 2 := by
  rw [cov_eq_covAux ρ σ (t + 1) (t + 1) (t + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ t t (t + 2) (by omega) (by omega)]
  unfold covAux
  have key : ∀ j ∈ range (t + 2),
      ivar ρ σ j * load ρ (t + 1) j * load ρ (t + 1) j
        = ρ ^ 2 * (ivar ρ σ j * load ρ t j * load ρ t j)
          + (if j = t + 1 then ivar ρ σ j else 0) := by
    intro j _
    rw [load_succ]
    by_cases h : j = t + 1
    · subst h
      rw [load_of_lt ρ (show t < t + 1 by omega)]
      simp; ring
    · simp [h]; ring
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [Finset.sum_ite_eq' (range (t + 2)) (t + 1) (fun j => ivar ρ σ j)]
  simp [Finset.mem_range, ivar]

/-- **Stationarity.**  The start-up draw is chosen so the variance never moves. -/
theorem cov_diag (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (t : ℕ) :
    cov ρ σ t t = σ ^ 2 / (1 - ρ ^ 2) := by
  induction t with
  | zero => exact cov_zero_zero ρ σ
  | succ n ih => rw [cov_succ_diag, ih]; field_simp; ring

lemma cov_add (t d : ℕ) : cov ρ σ (t + d) t = ρ ^ d * cov ρ σ t t := by
  induction d with
  | zero => simp
  | succ k ih =>
    rw [show t + (k + 1) = (t + k) + 1 by omega,
        cov_succ_left ρ σ (t + k) t (by omega), ih]
    ring

/-- **Autocovariance of the levels**: `γ(d) = σ² ρ^d / (1 - ρ²)`. -/
theorem cov_closed (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (t d : ℕ) :
    cov ρ σ (t + d) t = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ d := by
  rw [cov_add, cov_diag ρ σ h]

/-! ## 4. The autocovariance matrix of growth -/

/-- Loading of innovation `j` on the growth `Δu (a+1) = u (a+1) - u a`. -/
noncomputable def gload (ρ : ℝ) (a j : ℕ) : ℝ := load ρ (a + 1) j - load ρ a j

lemma gload_of_lt {a j : ℕ} (h : a + 1 < j) : gload ρ a j = 0 := by
  rw [gload, load_of_lt ρ (show a + 1 < j by omega), load_of_lt ρ (show a < j by omega)]
  ring

noncomputable def gcovAux (ρ σ : ℝ) (a b n : ℕ) : ℝ :=
  ∑ j ∈ range n, ivar ρ σ j * gload ρ a j * gload ρ b j

/-- Covariance of the growth rates `Δu (a+1)` and `Δu (b+1)`. -/
noncomputable def gcov (ρ σ : ℝ) (a b : ℕ) : ℝ := gcovAux ρ σ a b (max a b + 2)

lemma gcovAux_extend {a b n m : ℕ} (ha : a + 1 < n) (hnm : n ≤ m) :
    gcovAux ρ σ a b n = gcovAux ρ σ a b m := by
  refine Finset.sum_subset (Finset.range_subset.mpr hnm) ?_
  intro x _ hx
  rw [Finset.mem_range] at hx
  rw [gload_of_lt ρ (show a + 1 < x by omega)]
  ring

lemma gcov_eq_gcovAux (a b n : ℕ) (ha : a + 1 < n) (hb : b + 1 < n) :
    gcov ρ σ a b = gcovAux ρ σ a b n :=
  gcovAux_extend ρ σ (by omega) (by omega)

lemma gcov_symm (a b : ℕ) : gcov ρ σ a b = gcov ρ σ b a := by
  unfold gcov gcovAux
  rw [max_comm]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- Bilinearity: the covariance of differences is the difference of covariances. -/
theorem gcov_expand (a b : ℕ) :
    gcov ρ σ a b
      = cov ρ σ (a + 1) (b + 1) - cov ρ σ (a + 1) b - cov ρ σ a (b + 1) + cov ρ σ a b := by
  rw [gcov_eq_gcovAux ρ σ a b (max a b + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ (a + 1) (b + 1) (max a b + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ (a + 1) b (max a b + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ a (b + 1) (max a b + 2) (by omega) (by omega),
      cov_eq_covAux ρ σ a b (max a b + 2) (by omega) (by omega)]
  unfold gcovAux covAux
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => by unfold gload; ring

/-- `1 - ρ² ≠ 0` rules out `ρ = -1`, so `1 + ρ ≠ 0`. -/
lemma one_add_ne_zero (h : (1 : ℝ) - ρ ^ 2 ≠ 0) : (1 : ℝ) + ρ ≠ 0 := by
  intro hc
  apply h
  have : ρ = -1 := by linarith
  subst this; norm_num

/-- **Variance of growth**: `Var(Δy) = 2σ²/(1+ρ)`, the diagonal of `Ω`. -/
theorem gcov_self (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (a : ℕ) :
    gcov ρ σ a a = 2 * σ ^ 2 / (1 + ρ) := by
  have h1 : (1 : ℝ) + ρ ≠ 0 := one_add_ne_zero ρ h
  have e0 : cov ρ σ a a = σ ^ 2 / (1 - ρ ^ 2) := cov_diag ρ σ h a
  have e1 : cov ρ σ (a + 1) (a + 1) = σ ^ 2 / (1 - ρ ^ 2) := cov_diag ρ σ h (a + 1)
  have e2 : cov ρ σ (a + 1) a = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ 1 := cov_closed ρ σ h a 1
  have e3 : cov ρ σ a (a + 1) = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ 1 := by
    rw [cov_symm]; exact e2
  rw [gcov_expand, e0, e1, e2, e3]
  field_simp
  ring

/-- **Autocovariance of growth at lag `k+1 ≥ 1`**: `-σ²(1-ρ)/(1+ρ) * ρ^k`. -/
theorem gcov_lag (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (b k : ℕ) :
    gcov ρ σ (b + k + 1) b = -(σ ^ 2 * (1 - ρ) / (1 + ρ)) * ρ ^ k := by
  have h1 : (1 : ℝ) + ρ ≠ 0 := one_add_ne_zero ρ h
  have e0 : cov ρ σ (b + k + 1) b = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ (k + 1) := by
    rw [show b + k + 1 = b + (k + 1) by omega]; exact cov_closed ρ σ h b (k + 1)
  have e1 : cov ρ σ (b + k + 1 + 1) (b + 1) = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ (k + 1) := by
    rw [show b + k + 1 + 1 = (b + 1) + (k + 1) by omega]; exact cov_closed ρ σ h (b + 1) (k + 1)
  have e2 : cov ρ σ (b + k + 1 + 1) b = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ (k + 2) := by
    rw [show b + k + 1 + 1 = b + (k + 2) by omega]; exact cov_closed ρ σ h b (k + 2)
  have e3 : cov ρ σ (b + k + 1) (b + 1) = σ ^ 2 / (1 - ρ ^ 2) * ρ ^ k := by
    rw [show b + k + 1 = (b + 1) + k by omega]; exact cov_closed ρ σ h (b + 1) k
  rw [gcov_expand, e0, e1, e2, e3]
  have hρ2 : (1 : ℝ) - ρ ^ 2 = (1 - ρ) * (1 + ρ) := by ring
  field_simp
  ring

/-- **The full variance–autocovariance matrix of growth.**  For `Δy t` and `Δy s`
with `t = a+1`, `s = b+1`, the `(t,s)` entry of `Ω` is

    2σ²/(1+ρ)                     if t = s,
    -σ²(1-ρ)/(1+ρ) * ρ^(|t-s|-1)  otherwise.

It is Toeplitz (depends on `|t-s|` only), free of `α` and of `T`. -/
theorem gcov_toeplitz (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (a b : ℕ) :
    gcov ρ σ a b =
      if a = b then 2 * σ ^ 2 / (1 + ρ)
      else -(σ ^ 2 * (1 - ρ) / (1 + ρ)) * ρ ^ (max a b - min a b - 1) := by
  by_cases hab : a = b
  · subst hab; simp [gcov_self ρ σ h]
  · rcases Nat.lt_or_ge a b with hlt | hge
    · obtain ⟨k, hk⟩ : ∃ k, b = a + k + 1 := ⟨b - a - 1, by omega⟩
      subst hk
      rw [if_neg hab, gcov_symm, gcov_lag ρ σ h a k]
      congr 2
      omega
    · have hgt : b < a := by omega
      obtain ⟨k, hk⟩ : ∃ k, a = b + k + 1 := ⟨a - b - 1, by omega⟩
      subst hk
      rw [if_neg hab, gcov_lag ρ σ h b k]
      congr 2
      omega

/-! ## 5. Consequences -/

/-- Rows of `Ω` decay at rate `ρ` past the first off-diagonal: growth is
ARMA(1,1), `(1 - ρL) Δy t = (1 - L) ε t`. -/
theorem gcov_decay (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (b k : ℕ) :
    gcov ρ σ (b + (k + 1) + 1) b = ρ * gcov ρ σ (b + k + 1) b := by
  rw [gcov_lag ρ σ h b (k + 1), gcov_lag ρ σ h b k]
  ring

/-- **The first-difference plim of `SPEC.md`**, read straight off `Ω`:
`Cov(Δy t, Δy (t-1)) / Var(Δy) = (ρ - 1)/2`, whatever `σ`, `T` or `α`. -/
theorem fd_plim (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (hσ : σ ≠ 0) (b : ℕ) :
    gcov ρ σ (b + 1) b / gcov ρ σ b b = (ρ - 1) / 2 := by
  have h1 : (1 : ℝ) + ρ ≠ 0 := one_add_ne_zero ρ h
  have hb : gcov ρ σ (b + 1) b = -(σ ^ 2 * (1 - ρ) / (1 + ρ)) * ρ ^ 0 := by
    have := gcov_lag ρ σ h b 0
    rwa [show b + 0 + 1 = b + 1 by omega] at this
  rw [hb, gcov_self ρ σ h]
  have hσ2 : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  field_simp
  ring

/-- The `T × T` matrix `Ω` of `(Δy 1, …, Δy T)`, row `s`, column `t`. -/
noncomputable def Omega (ρ σ : ℝ) (T : ℕ) : Matrix (Fin T) (Fin T) ℝ :=
  Matrix.of fun s t => gcov ρ σ (s : ℕ) (t : ℕ)

theorem Omega_apply (h : (1 : ℝ) - ρ ^ 2 ≠ 0) (T : ℕ) (s t : Fin T) :
    Omega ρ σ T s t =
      if (s : ℕ) = (t : ℕ) then 2 * σ ^ 2 / (1 + ρ)
      else -(σ ^ 2 * (1 - ρ) / (1 + ρ)) * ρ ^ (max (s : ℕ) (t : ℕ) - min (s : ℕ) (t : ℕ) - 1) :=
  gcov_toeplitz ρ σ h s t

theorem Omega_isSymm (T : ℕ) : (Omega ρ σ T).IsSymm := by
  ext s t
  simpa [Omega, Matrix.IsSymm] using gcov_symm ρ σ (t : ℕ) (s : ℕ)

/-- `Ω` is a genuine covariance matrix: its quadratic form is a sum of squares
weighted by the innovation variances, hence non-negative when `ρ² < 1`. -/
theorem Omega_quadForm_nonneg (hρ : ρ ^ 2 < 1) (T : ℕ) (v : Fin T → ℝ) :
    0 ≤ ∑ s : Fin T, ∑ t : Fin T, v s * Omega ρ σ T s t * v t := by
  have hivar : ∀ j, 0 ≤ ivar ρ σ j := by
    intro j
    unfold ivar
    split
    · exact div_nonneg (sq_nonneg σ) (by linarith)
    · exact sq_nonneg σ
  have hg : ∀ s t : Fin T, Omega ρ σ T s t = gcovAux ρ σ (s : ℕ) (t : ℕ) (T + 1) := by
    intro s t
    exact gcov_eq_gcovAux ρ σ (s : ℕ) (t : ℕ) (T + 1) (by omega) (by omega)
  have key : ∑ s : Fin T, ∑ t : Fin T, v s * Omega ρ σ T s t * v t
      = ∑ j ∈ range (T + 1), ivar ρ σ j * (∑ s : Fin T, v s * gload ρ (s : ℕ) j) ^ 2 := by
    simp_rw [hg, gcovAux, Finset.mul_sum, Finset.sum_mul]
    rw [Finset.sum_congr rfl (fun s _ => Finset.sum_comm), Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [sq, Finset.sum_mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun t _ => by ring
  rw [key]
  exact Finset.sum_nonneg fun j _ => mul_nonneg (hivar j) (sq_nonneg _)

end PanelAR1
