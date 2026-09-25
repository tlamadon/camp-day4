"""The five estimators of rho.

The three regressions are closed-form ratios of sums: writing them as
Sigma xy / Sigma x^2 rather than calling a regression package is faster and keeps
the moment condition that each one gets wrong explicit.  The other two fit the
covariance matrix of growth by minimum distance -- one assuming y is an AR(1),
one allowing measurement error on top of it -- and return the variance
parameters alongside rho.  The criterion they share lives in `growth.py`.

Standard errors cluster by individual.  SPEC.md fixes the clustering but not the
finite-sample correction; we use c = G / (G - 1) for all three, the convention
that stays asymptotically honest as N grows with T fixed.  A degrees-of-freedom
factor counting the N absorbed effects would inflate the within standard error
by ~22% at T = 3 and would muddy the mean-SE-over-SD diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import growth


@dataclass(frozen=True)
class Fit:
    """A point estimate of rho and its clustered standard error.

    The variance fields are NaN for the estimators that do not report them.
    """

    rho_hat: float
    se: float
    sigma_eps_hat: float = float("nan")
    sigma_eps_se: float = float("nan")
    sigma_nu_hat: float = float("nan")
    sigma_nu_se: float = float("nan")


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


# --- GMM on the growth covariance matrix -------------------------------------


def _growth_moments(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Growth, and its sample second-moment matrix C = Delta Y' Delta Y / N.

    Not centred: E Delta y_it = 0 under the stationary initial condition, so the
    raw cross-products already estimate Omega.  Growth runs over t = 1, ..., T --
    one period more than first-difference OLS, which needs a lagged difference too.
    """
    d = np.diff(y, axis=1)
    return d, d.T @ d / d.shape[0]


def _growth_sandwich(d: np.ndarray, C: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """Var(theta-hat) for a minimum-distance fit of Omega under an identity weight.

    `blocks` are the T x T matrices d Omega / d theta_p.  One individual
    contributes one moment vector, so this clusters by individual on its own.
    The N x T^2 matrix of those vectors is never formed: G'(g_i - g-bar) is one
    N-vector per parameter, a single matrix product each.  S-hat is never
    inverted either -- it appears only as the small matrix G' S-hat G, which is
    what lets the weight stay fixed when the efficient one does not exist.
    """
    n = d.shape[0]
    bread = np.array([[float((p * q).sum()) for q in blocks] for p in blocks])
    score = np.column_stack(
        [((d @ p) * d).sum(axis=1) - float((p * C).sum()) for p in blocks]
    )
    meat = score.T @ score / (n - 1.0)
    bread_inv = np.linalg.inv(bread)
    return bread_inv @ meat @ bread_inv / n


def _sd_and_se(variance: float, variance_se2: float) -> tuple[float, float]:
    """A variance estimate and its sandwich entry, as an SD and its delta-method SE.

    A variance pinned at zero is on the boundary of the parameter space, where
    the usual asymptotics do not hold; the SE is NaN rather than misleading.
    """
    sd = float(np.sqrt(variance)) if variance > 0.0 else 0.0
    if sd == 0.0:
        return 0.0, float("nan")
    return sd, float(np.sqrt(variance_se2) / (2.0 * sd))


def gmm_growth(y: np.ndarray) -> Fit:
    """Minimum distance between C and sigma_eps^2 A(rho).

    GMM on all T^2 moments E[Delta y_it Delta y_is - Omega_ts] = 0 with an
    identity weight.  Consistent wherever y really is an AR(1) around a level;
    under M2 it fits the wrong shape and converges to a pseudo-true rho instead.
    """
    d, C = _growth_moments(y)
    T = d.shape[1]
    rho_hat, var_eps = growth.fit_ar1(C)

    shape = growth.shape(T, rho_hat)
    var = _growth_sandwich(d, C, [var_eps * growth.shape_deriv(T, rho_hat), shape])
    sigma_eps, se_eps = _sd_and_se(var_eps, var[1, 1])
    return Fit(rho_hat, float(np.sqrt(var[0, 0])), sigma_eps, se_eps)


def gmm_growth_me(y: np.ndarray) -> Fit:
    """The same against sigma_eps^2 A(rho) + sigma_nu^2 B: the M2 shape.

    One more parameter buys consistency under measurement error, and costs
    precision when there is none -- in M0 and M1 the true sigma_nu is zero, a
    boundary the fit is pinned to about half the time.
    """
    d, C = _growth_moments(y)
    T = d.shape[1]
    rho_hat, var_eps, var_nu = growth.fit_me(C)

    shape, noise = growth.shape(T, rho_hat), growth.noise_shape(T)
    var = _growth_sandwich(
        d, C, [var_eps * growth.shape_deriv(T, rho_hat), shape, noise]
    )
    sigma_eps, se_eps = _sd_and_se(var_eps, var[1, 1])
    sigma_nu, se_nu = _sd_and_se(var_nu, var[2, 2])
    return Fit(rho_hat, float(np.sqrt(var[0, 0])), sigma_eps, se_eps, sigma_nu, se_nu)


ESTIMATOR_FUNCS = {
    "pooled": pooled_ols,
    "fd": first_difference,
    "within": within,
    "gmm": gmm_growth,
    "gmm_me": gmm_growth_me,
}
