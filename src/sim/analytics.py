"""Large-N probability limits, from the "Analytical benchmarks" section of SPEC.md.

These are what the simulated means are checked against, both in the unit tests
(at N = 10^6) and as the dashed reference lines on every figure.
"""

from __future__ import annotations

from .config import RHO, SIGMA_EPS, sigma_alpha


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


def plim(estimator: str, model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """plim of `estimator` in `model` at panel length `T`."""
    if estimator == "pooled":
        return plim_pooled(rho, s_eps, sigma_alpha(model))
    if estimator == "fd":
        return plim_fd(rho)
    if estimator == "within":
        return plim_within(T, rho)
    raise ValueError(f"unknown estimator {estimator!r}")
