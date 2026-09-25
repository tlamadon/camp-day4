"""Numerical cross-check of the growth autocovariance matrix proved in Lean.

Draws panels from the project's own DGP and compares the empirical covariance
matrix of (Dy_i1, ..., Dy_iT) with the closed form of PanelAR1/GrowthACov.lean:

    Omega[t,s] =  2 s_eps^2 / (1 + rho)                          if t == s
               = -s_eps^2 (1 - rho) / (1 + rho) * rho^(|t-s|-1)  otherwise

Run with `make proof-numeric`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sim.config import RHO, SIGMA_EPS  # noqa: E402
from sim.dgp import simulate_panel  # noqa: E402

T = 6
N = 1_000_000
TOL = 1e-3       # ~10 Monte Carlo SEs at N = 1e6; the seed is fixed
TOL_SAME = 1e-12  # M0 vs M1: exact cancellation, up to floating point


def omega(T: int, rho: float, s_eps: float) -> np.ndarray:
    """The Lean-verified matrix (gcov_toeplitz)."""
    lag = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    off = -(s_eps**2) * (1.0 - rho) / (1.0 + rho) * rho ** np.maximum(lag - 1, 0)
    return np.where(lag == 0, 2.0 * s_eps**2 / (1.0 + rho), off)


def main() -> int:
    theory = omega(T, RHO, SIGMA_EPS)
    worst = 0.0
    empirical = {}
    for model in ("M0", "M1"):
        rng = np.random.default_rng(20260924)
        g = np.diff(simulate_panel(model, T, N, rng), axis=1)
        empirical[model] = np.cov(g, rowvar=False, bias=True)
        worst = max(worst, float(np.abs(empirical[model] - theory).max()))

    print(f"rho = {RHO}, sigma_eps = {SIGMA_EPS}, T = {T}, N = {N:,}")
    print("\ntheory (first row):     ", np.round(theory[0], 6))
    for model in ("M0", "M1"):
        print(f"empirical {model} (first row): ", np.round(empirical[model][0], 6))
    same = float(np.abs(empirical["M0"] - empirical["M1"]).max())
    print(f"\nmax |empirical - theory|      = {worst:.2e}  (tol {TOL:.0e})")
    print(f"max |M0 - M1| (alpha cancels) = {same:.2e}  (tol {TOL_SAME:.0e})")
    print(f"implied FD plim  Omega[1,0]/Omega[0,0] = {theory[1, 0] / theory[0, 0]:+.4f}"
          f"   (analytic (rho-1)/2 = {(RHO - 1) / 2:+.4f})")

    ok = worst < TOL and same < TOL_SAME
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
