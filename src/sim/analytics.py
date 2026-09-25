"""The probability limits of SPEC.md's "Analytical benchmarks".

These are what the simulated means are checked against -- in the unit tests (at
N = 10^6) and as the dashed reference lines on every figure.

Two layers.  The closed forms come first, exactly as the spec writes them; they
hold in M0 and M1, where y is an AR(1).  Then the general versions, which take
the autocovariance function of y's time-varying part and so cover M2 as well,
where y is an ARMA(1,1) and none of the closed forms apply.  `plim` routes
everything through the general versions, and the tests pin them to the closed
forms wherever both are defined.  Omega itself lives in `growth.py`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import growth
from .config import (
    ESTIMATES_SIGMA_EPS,
    ESTIMATES_SIGMA_NU,
    RHO,
    SIGMA_EPS,
    effect_kind,
    sigma_alpha,
    sigma_nu,
)


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


# --- The model in general: a level plus a stationary process ------------------
#
# Every model here is y_it = l_i + v_it with l_i time-invariant and v_it mean
# zero and stationary.  Those two objects -- the variance of the level and the
# autocovariance function of v -- are all any of the four regression plims needs.


def level_var(model: str, rho: float = RHO) -> float:
    """Variance of the time-invariant part of y.

    M2's individual effect is already a level.  M0's and M1's is an intercept in
    the recursion, worth alpha_i/(1 - rho) as a level.
    """
    if effect_kind(model) == "level":
        return sigma_alpha(model) ** 2
    return var_mu(rho, sigma_alpha(model))


def autocov(model: str, K: int, rho: float = RHO, s_eps: float = SIGMA_EPS) -> np.ndarray:
    """gamma(0), ..., gamma(K) of y's time-varying part.

    The persistent component contributes sigma_eps^2 rho^k / (1 - rho^2) at every
    lag; measurement error is white, so it lands on gamma(0) alone.  That gap --
    geometric decay everywhere except a spike at lag zero -- is what separates
    the two variances.
    """
    gamma = var_u(rho, s_eps) * rho ** np.arange(K + 1, dtype=float)
    gamma[0] += sigma_nu(model) ** 2
    return gamma


def plim_pooled_general(model: str, rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """(sigma_l^2 + gamma(1)) / (sigma_l^2 + gamma(0))."""
    gamma = autocov(model, 1, rho, s_eps)
    level = level_var(model, rho)
    return float((level + gamma[1]) / (level + gamma[0]))


def plim_fd_general(model: str, rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """Cov(Delta v_t, Delta v_t-1) / Var(Delta v), both read off gamma."""
    gamma = autocov(model, 2, rho, s_eps)
    return float((2 * gamma[1] - gamma[0] - gamma[2]) / (2 * (gamma[0] - gamma[1])))


def plim_within_general(
    model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS
) -> float:
    """The within plim at finite T, for any gamma.

    Writing the two demeaned regressors as Q X v and Q Y v, where X and Y select
    v_0..v_T-1 and v_1..v_T out of v and Q demeans over the T periods, the plim
    is trace(Q X Gamma Y') / trace(Q X Gamma X').  With gamma the AR(1) one this
    reproduces Nickell exactly, which is how it is tested.
    """
    gamma = autocov(model, T, rho, s_eps)
    covariance = gamma[np.abs(np.subtract.outer(np.arange(T + 1), np.arange(T + 1)))]
    lag, dep = np.eye(T + 1)[:T], np.eye(T + 1)[1:]
    demean = np.eye(T) - np.ones((T, T)) / T
    lagged = demean @ lag @ covariance
    return float(np.trace(lagged @ dep.T) / np.trace(lagged @ lag.T))


@lru_cache(maxsize=None)
def growth_plim(
    estimator: str, model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS
) -> tuple[float, float, float]:
    """(rho, sigma_eps, sigma_nu) the growth fit returns at the population Omega.

    Where the shape being fitted is the one that generated Omega these are the
    true values; where it is not -- the AR(1) fit under M2 -- they are the
    pseudo-true values, the point the criterion actually converges to.  Running
    the estimator's own criterion on the population matrix is what makes that
    exact rather than approximate.
    """
    omega = growth.cov(T, rho, s_eps, sigma_nu(model))
    if estimator == "gmm":
        rho_star, var_eps = growth.fit_ar1(omega)
        return rho_star, float(np.sqrt(var_eps)), float("nan")
    rho_star, var_eps, var_nu = growth.fit_me(omega)
    return rho_star, float(np.sqrt(var_eps)), float(np.sqrt(var_nu))


# --- Dispatch ----------------------------------------------------------------


def plim(estimator: str, model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS) -> float:
    """plim of `estimator` in `model` at panel length `T`."""
    if estimator == "pooled":
        return plim_pooled_general(model, rho, s_eps)
    if estimator == "fd":
        return plim_fd_general(model, rho, s_eps)
    if estimator == "within":
        return plim_within_general(model, T, rho, s_eps)
    if estimator in ("gmm", "gmm_me"):
        return growth_plim(estimator, model, T, rho, s_eps)[0]
    raise ValueError(f"unknown estimator {estimator!r}")


def plim_param(
    parameter: str, estimator: str, model: str, T: int, rho: float = RHO, s_eps: float = SIGMA_EPS
) -> float:
    """plim of a variance parameter, or NaN where the estimator reports none."""
    reports = {"sigma_eps": ESTIMATES_SIGMA_EPS, "sigma_nu": ESTIMATES_SIGMA_NU}
    if parameter not in reports:
        raise ValueError(f"unknown parameter {parameter!r}")
    if estimator not in reports[parameter]:
        return float("nan")
    index = 1 if parameter == "sigma_eps" else 2
    return growth_plim(estimator, model, T, rho, s_eps)[index]
