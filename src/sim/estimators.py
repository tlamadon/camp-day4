"""The three estimators of rho, each a closed-form ratio of sums.

Writing them as Sigma xy / Sigma x^2 rather than calling a regression package is
faster and keeps the moment condition that each one gets wrong explicit.

Standard errors cluster by individual.  SPEC.md fixes the clustering but not the
finite-sample correction; we use c = G / (G - 1) for all three, the convention
that stays asymptotically honest as N grows with T fixed.  A degrees-of-freedom
factor counting the N absorbed effects would inflate the within standard error
by ~22% at T = 3 and would muddy the mean-SE-over-SD diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fit:
    """A point estimate of rho and its clustered standard error."""

    rho_hat: float
    se: float


def _fit(x: np.ndarray, y: np.ndarray) -> Fit:
    """Regress y on x through the origin; both arrays are already residualised.

    x and y are N x P (individuals by regression periods).  Clustering by
    individual is then a sum over rows.
    """
    sxx = float(np.einsum("ij,ij->", x, x))
    sxy = float(np.einsum("ij,ij->", x, y))
    rho_hat = sxy / sxx

    resid = y - rho_hat * x
    score = np.einsum("ij,ij->i", x, resid)  # Sigma_t x_it e_it, one per individual

    n_clusters = x.shape[0]
    correction = n_clusters / (n_clusters - 1.0)
    var = correction * float(score @ score) / sxx**2
    return Fit(rho_hat, float(np.sqrt(var)))


def pooled_ols(y: np.ndarray) -> Fit:
    """y_it on a constant and y_i,t-1, t = 1..T.

    The constant is partialled out by global demeaning, which makes the slope and
    its clustered variance exactly those of the two-regressor OLS fit.
    """
    lag, dep = y[:, :-1], y[:, 1:]
    return _fit(lag - lag.mean(), dep - dep.mean())


def first_difference(y: np.ndarray) -> Fit:
    """Delta y_it on Delta y_i,t-1 with no constant, t = 2..T."""
    d = np.diff(y, axis=1)  # columns are Delta y_i1 ... Delta y_iT
    return _fit(d[:, :-1], d[:, 1:])


def within(y: np.ndarray) -> Fit:
    """Demeaned y_it on demeaned y_i,t-1, t = 1..T.

    The two variables are demeaned separately: the mean of the dependent
    variable uses y_i1..y_iT, the mean of the lag uses y_i0..y_i,T-1.
    """
    lag, dep = y[:, :-1], y[:, 1:]
    return _fit(
        lag - lag.mean(axis=1, keepdims=True),
        dep - dep.mean(axis=1, keepdims=True),
    )


ESTIMATOR_FUNCS = {
    "pooled": pooled_ols,
    "fd": first_difference,
    "within": within,
}
