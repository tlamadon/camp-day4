"""The four estimators of rho.

The three regressions are closed-form ratios of sums: writing them as
Sigma xy / Sigma x^2 rather than calling a regression package is faster and keeps
the moment condition that each one gets wrong explicit.  The fourth, GMM on the
growth covariance matrix, is the one that gets every moment right; it also
returns sigma_eps, and it is the only estimator here that needs a search.

Standard errors cluster by individual.  SPEC.md fixes the clustering but not the
finite-sample correction; we use c = G / (G - 1) for all three, the convention
that stays asymptotically honest as N grows with T fixed.  A degrees-of-freedom
factor counting the N absorbed effects would inflate the within standard error
by ~22% at T = 3 and would muddy the mean-SE-over-SD diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytics import growth_shape, growth_shape_deriv
from .config import GMM_GRID_POINTS, GMM_RHO_BOUNDS


@dataclass(frozen=True)
class Fit:
    """A point estimate of rho and its clustered standard error.

    `sigma_eps_hat` is NaN for the three estimators that identify rho alone.
    """

    rho_hat: float
    se: float
    sigma_eps_hat: float = float("nan")
    sigma_eps_se: float = float("nan")


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


def _band_sums(C: np.ndarray) -> np.ndarray:
    """Sum of C over each band |t - s| = k, for k = 0, ..., T-1.

    Omega is Toeplitz, so the criterion only ever sees C through these T numbers,
    which is what keeps a 1,001-point search over rho cheap at T = 50.
    """
    T = C.shape[0]
    out = np.empty(T)
    out[0] = float(np.trace(C))
    for k in range(1, T):
        out[k] = 2.0 * float(np.trace(C, offset=k))  # C is symmetric by construction
    return out


def _band_sizes(T: int) -> np.ndarray:
    """How many entries of the T x T matrix each band holds."""
    k = np.arange(T)
    return np.where(k == 0, float(T), 2.0 * (T - k))


def _criterion(rho: np.ndarray, bands: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The profiled criterion <A, C>^2 / <A, A>, band by band.

    Maximising it is minimising ||C - sigma^2 A(rho)||_F^2 after concentrating
    sigma^2 out.  It is invariant to the scale of A, so the 1 / (1 + rho) factor
    of A is dropped here: the criterion then stays finite as rho approaches -1.
    Branches where <A, C> < 0 would need a negative sigma^2 and are ruled out.
    """
    k = np.arange(bands.size)
    a = np.where(k == 0, 2.0, -(1.0 - rho[:, None]) * rho[:, None] ** np.maximum(k - 1, 0))
    inner = a @ bands
    return np.where(inner > 0.0, inner**2 / (a**2 @ sizes), -np.inf)


def _argmax_rho(bands: np.ndarray, sizes: np.ndarray) -> float:
    """Grid search over rho, then golden-section refinement on the bracket."""
    grid = np.linspace(*GMM_RHO_BOUNDS, GMM_GRID_POINTS)
    best = int(np.argmax(_criterion(grid, bands, sizes)))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, GMM_GRID_POINTS - 1)]

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    # 40 halvings shrink the bracket to ~3e-11; the maximum is smooth, so its
    # location is only resolvable to ~1e-8 anyway.
    for _ in range(40):
        q_c, q_d = _criterion(np.array([c, d]), bands, sizes)
        if q_c > q_d:
            hi, d, c = d, c, d - phi * (d - lo)
        else:
            lo, c, d = c, d, c + phi * (hi - c)
    return float(0.5 * (lo + hi))


def gmm_growth(y: np.ndarray) -> Fit:
    """Minimum distance between the sample and model covariance matrices of growth.

    Fits Omega(rho, sigma_eps^2) to C = Delta Y' Delta Y / N in Frobenius norm,
    i.e. GMM on all T^2 moments E[Delta y_it Delta y_is - Omega_ts] = 0 with an
    identity weight.  Uses growth at t = 1, ..., T -- one period more than
    first-difference OLS, which needs a lagged difference as well.
    """
    d = np.diff(y, axis=1)
    n, T = d.shape
    C = d.T @ d / n  # not centred: E Delta y_it = 0 under the stationary start

    bands, sizes = _band_sums(C), _band_sizes(T)
    rho_hat = _argmax_rho(bands, sizes)

    A = growth_shape(T, rho_hat)
    s2 = float((A * C).sum() / (A * A).sum())

    # G = d vec(Omega) / d theta', theta = (rho, sigma_eps^2), one T x T block each.
    g_rho, g_s2 = s2 * growth_shape_deriv(T, rho_hat), A
    bread = np.array(
        [
            [(g_rho * g_rho).sum(), (g_rho * g_s2).sum()],
            [(g_rho * g_s2).sum(), (g_s2 * g_s2).sum()],
        ]
    )
    # One moment vector per individual, so the sandwich clusters by i on its own.
    # The N x T^2 matrix of those vectors is never formed: G' (g_i - g-bar) is
    # these two N-vectors, one matrix product each.
    score = np.column_stack(
        [
            ((d @ g_rho) * d).sum(axis=1) - (g_rho * C).sum(),
            ((d @ g_s2) * d).sum(axis=1) - (g_s2 * C).sum(),
        ]
    )
    meat = score.T @ score / (n - 1.0)
    bread_inv = np.linalg.inv(bread)
    var = bread_inv @ meat @ bread_inv / n

    sigma = np.sqrt(s2) if s2 > 0.0 else 0.0
    return Fit(
        rho_hat,
        float(np.sqrt(var[0, 0])),
        float(sigma),
        float(np.sqrt(var[1, 1]) / (2.0 * sigma)) if sigma > 0.0 else float("nan"),
    )


ESTIMATOR_FUNCS = {
    "pooled": pooled_ols,
    "fd": first_difference,
    "within": within,
    "gmm": gmm_growth,
}
