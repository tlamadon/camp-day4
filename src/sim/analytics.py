"""Population objects: the probability limits, and the growth covariance matrix.

The plims are the "Analytical benchmarks" section of SPEC.md, and are what the
simulated means are checked against -- in the unit tests (at N = 10^6) and as the
dashed reference lines on every figure.  Omega, the variance-autocovariance
matrix of growth, is the thing the GMM estimator fits; it lives here because it
is a property of the model, not of the estimator.
"""

from __future__ import annotations

import numpy as np

from .config import ESTIMATES_SIGMA_EPS, RHO, SIGMA_EPS, sigma_alpha


def var_mu(rho: float = RHO, s_alpha: float = 0.0) -> float:
    """Variance of the long-run mean mu_i = alpha_i / (1 - rho)."""
    return s_alpha**2 / (1.0 - rho) ** 2


def var_u(rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """Variance of the stationary transitory component u_it."""
    return s_eps**2 / (1.0 - rho**2)


def plim_pooled(rho: float = RHO, s_eps: float = SIGMA_EPS, s_alpha: float = 0.0) -> float:
    """plim of pooled OLS: (sigma_mu^2 + rho sigma_u^2) / (sigma_mu^2 + sigma_u^2).

    With s_alpha = 0 this collapses to rho, so one formula covers M0 and M1.
    """
    smu, su = var_mu(rho, s_alpha), var_u(rho, s_eps)
    return (smu + rho * su) / (smu + su)


def plim_fd(rho: float = RHO) -> float:
    """plim of OLS in first differences: rho - (1 + rho) / 2 = (rho - 1) / 2.

    Free of T, sigma_eps and sigma_alpha.
    """
    return (rho - 1.0) / 2.0


def nickell_bias(T: int, rho: float = RHO) -> float:
    """Nickell (1981) bias of the within estimator, plim(rho_hat_W) - rho."""
    if T < 2:
        raise ValueError("the within estimator needs T >= 2")
    a_t = (1.0 - rho**T) / (T * (1.0 - rho))
    lead = -(1.0 + rho) / (T - 1) * (1.0 - a_t)
    denom = 1.0 - (2.0 * rho) / ((1.0 - rho) * (T - 1)) * (1.0 - a_t)
    return lead / denom


def plim_within(T: int, rho: float = RHO) -> float:
    """plim of the within estimator.  Identical in M0 and M1: demeaning kills alpha_i."""
    return rho + nickell_bias(T, rho)


# --- The growth covariance matrix --------------------------------------------


def growth_shape(T: int, rho: float = RHO) -> np.ndarray:
    """A(rho) = Omega / sigma_eps^2, the T x T growth covariance matrix up to scale.

    Omega is Toeplitz: 2 / (1 + rho) on the diagonal and
    -(1 - rho) / (1 + rho) * rho^(|t-s|-1) off it (SPEC.md, "Growth-covariance
    GMM"; derived in proofs/PanelAR1/GrowthACov.lean).  It is free of alpha_i and
    of sigma_alpha, which is why the GMM estimator is consistent in M1 too.
    """
    k = _lag_distance(T)
    return np.where(k == 0, 2.0, -(1.0 - rho) * rho ** np.maximum(k - 1, 0)) / (1.0 + rho)


def growth_shape_deriv(T: int, rho: float = RHO) -> np.ndarray:
    """dA/drho, the rho block of the GMM gradient.

    With f(rho) = -(1 - rho) / (1 + rho) and f'(rho) = 2 / (1 + rho)^2, the
    off-diagonal entry f(rho) rho^(k-1) differentiates to
    f'(rho) rho^(k-1) + f(rho) (k-1) rho^(k-2); the second term is absent at
    k = 1, where the entry does not depend on rho at all.
    """
    k = _lag_distance(T)
    f, f_prime = -(1.0 - rho) / (1.0 + rho), 2.0 / (1.0 + rho) ** 2
    off = f_prime * rho ** np.maximum(k - 1, 0) + f * np.where(
        k >= 2, (k - 1) * rho ** np.maximum(k - 2, 0), 0.0
    )
    return np.where(k == 0, -2.0 / (1.0 + rho) ** 2, off)


def growth_cov(T: int, rho: float = RHO, s_eps: float = SIGMA_EPS) -> np.ndarray:
    """Omega, the variance-autocovariance matrix of (Delta y_i1, ..., Delta y_iT)."""
    return s_eps**2 * growth_shape(T, rho)


def _lag_distance(T: int) -> np.ndarray:
    """|t - s| for the T x T matrix of growth rates."""
    t = np.arange(T)
    return np.abs(np.subtract.outer(t, t))


# --- Dispatch ----------------------------------------------------------------


def plim(estimator: str, model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """plim of `estimator` in `model` at panel length `T`."""
    if estimator == "pooled":
        return plim_pooled(rho, s_eps, sigma_alpha(model))
    if estimator == "fd":
        return plim_fd(rho)
    if estimator == "within":
        return plim_within(T, rho)
    if estimator == "gmm":
        # Consistent: the moment conditions hold exactly and Omega identifies rho.
        return rho
    raise ValueError(f"unknown estimator {estimator!r}")


def plim_sigma_eps(estimator: str, s_eps: float = SIGMA_EPS) -> float:
    """plim of sigma_eps-hat, or NaN for the estimators that do not produce one."""
    if estimator in ESTIMATES_SIGMA_EPS:
        return s_eps
    return float("nan")
