"""Omega, the variance-autocovariance matrix of growth, and the fit of it.

SPEC.md, "Growth-covariance GMM".  Growth never sees the individual effect, so
Omega is the same object in every model; what changes across models is which
parameters it holds.  Two nested shapes:

    Omega = sigma_eps^2 A(rho)                  no measurement error (M0, M1)
    Omega = sigma_eps^2 A(rho) + sigma_nu^2 B   with it (M2)

A is the growth covariance of the AR(1) itself, Toeplitz and geometric; B is the
MA(1) left by differencing white noise, non-zero only on the diagonal and the
first off-diagonal.  Both are linear in the variances, so the variances
concentrate out of the criterion and only rho is searched over.

This module is the population side and the criterion; `estimators.py` wraps it
around a sample second-moment matrix and adds the sandwich, and `analytics.py`
feeds it a population Omega to get plims -- including the pseudo-true value of
the misspecified fit, which is what the criterion returns when the Omega it is
given did not come from the shape it fits.
"""

from __future__ import annotations

import numpy as np

from .config import GMM_GRID_POINTS, GMM_RHO_BOUNDS, RHO, SIGMA_EPS

# --- The two shapes ----------------------------------------------------------


def lag_distance(T: int) -> np.ndarray:
    """|t - s| for the T x T matrix of growth rates."""
    t = np.arange(T)
    return np.abs(np.subtract.outer(t, t))


def shape(T: int, rho: float = RHO) -> np.ndarray:
    """A(rho): 2/(1+rho) on the diagonal, -(1-rho)/(1+rho) rho^(k-1) off it."""
    k = lag_distance(T)
    return np.where(k == 0, 2.0, -(1.0 - rho) * rho ** np.maximum(k - 1, 0)) / (1.0 + rho)


def shape_deriv(T: int, rho: float = RHO) -> np.ndarray:
    """dA/drho, the rho block of the GMM gradient.

    With f(rho) = -(1 - rho)/(1 + rho) and f'(rho) = 2/(1 + rho)^2, the
    off-diagonal entry f(rho) rho^(k-1) differentiates to
    f'(rho) rho^(k-1) + f(rho) (k-1) rho^(k-2); the second term is absent at
    k = 1, where the entry does not depend on rho at all.
    """
    k = lag_distance(T)
    f, f_prime = -(1.0 - rho) / (1.0 + rho), 2.0 / (1.0 + rho) ** 2
    off = f_prime * rho ** np.maximum(k - 1, 0) + f * np.where(
        k >= 2, (k - 1) * rho ** np.maximum(k - 2, 0), 0.0
    )
    return np.where(k == 0, -2.0 / (1.0 + rho) ** 2, off)


def noise_shape(T: int) -> np.ndarray:
    """B: the growth covariance of iid measurement error, 2 on the diagonal and
    -1 on the first off-diagonal.  Free of rho -- which is what identifies it
    apart from A, whose off-diagonals decay geometrically instead of stopping."""
    k = lag_distance(T)
    return np.where(k == 0, 2.0, np.where(k == 1, -1.0, 0.0))


def cov(T: int, rho: float = RHO, s_eps: float = SIGMA_EPS, s_nu: float = 0.0) -> np.ndarray:
    """Omega for (Delta y_i1, ..., Delta y_iT)."""
    omega = s_eps**2 * shape(T, rho)
    if s_nu:
        omega = omega + s_nu**2 * noise_shape(T)
    return omega


# --- Reduction of a covariance matrix to its bands ---------------------------
#
# Both shapes are Toeplitz, so the criterion sees a T x T matrix only through
# these T numbers.  That is what keeps a 1,001-point search cheap at T = 50.


def band_sums(C: np.ndarray) -> np.ndarray:
    """Sum of C over each band |t - s| = k, for k = 0, ..., T-1."""
    T = C.shape[0]
    out = np.empty(T)
    out[0] = float(np.trace(C))
    for k in range(1, T):
        out[k] = 2.0 * float(np.trace(C, offset=k))  # C is symmetric by construction
    return out


def band_sizes(T: int) -> np.ndarray:
    """How many entries of the T x T matrix each band holds."""
    k = np.arange(T)
    return np.where(k == 0, float(T), 2.0 * (T - k))


def _a_bands(rho: np.ndarray, T: int) -> np.ndarray:
    """(1 + rho) A(rho), band by band, for a whole grid of rho at once.

    The criterion is invariant to the scale of A, so the 1/(1 + rho) factor is
    dropped here and put back only when a variance is read off: the bands then
    stay finite as rho approaches -1.
    """
    k = np.arange(T)
    rho = np.asarray(rho, dtype=float)[:, None]
    return np.where(k == 0, 2.0, -(1.0 - rho) * rho ** np.maximum(k - 1, 0))


def _b_bands(T: int) -> np.ndarray:
    k = np.arange(T)
    return np.where(k == 0, 2.0, np.where(k == 1, -1.0, 0.0))


# --- The profiled criterion --------------------------------------------------


def criterion_ar1(rho: np.ndarray, bands: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """<A, C>^2 / <A, A>, after concentrating sigma_eps^2 out.

    Maximising it minimises ||C - sigma_eps^2 A(rho)||_F^2.  Branches where
    <A, C> < 0 would need a negative variance and are ruled out.
    """
    a = _a_bands(rho, bands.size)
    inner = a @ bands
    return np.where(inner > 0.0, inner**2 / ((a * a) @ sizes), -np.inf)


def criterion_me(
    rho: np.ndarray, bands: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same, with both variances concentrated out and held non-negative.

    For fixed rho the criterion is linear least squares in (sigma_eps^2,
    sigma_nu^2), so it is the projection of C onto span{A, B} -- except that a
    variance may not go negative.  With two coefficients the non-negative
    solution is one of four candidates (both free, either one alone, neither),
    so they are all evaluated and the best feasible one taken.

    Returns the attained value of 2 c'g - c'Gc, which is ||C||^2 minus the
    criterion, along with the coefficients on the *scaled* A and on B.
    """
    T = bands.size
    a, b = _a_bands(rho, T), _b_bands(T)
    g_a, g_b = a @ bands, float(b @ bands)
    gram_aa, gram_ab = (a * a) @ sizes, (a * b) @ sizes
    gram_bb = float((b * b) @ sizes)

    def value(c_a: np.ndarray, c_b: np.ndarray) -> np.ndarray:
        return 2.0 * (c_a * g_a + c_b * g_b) - (
            c_a**2 * gram_aa + 2.0 * c_a * c_b * gram_ab + c_b**2 * gram_bb
        )

    # A and B coincide at rho = 0 -- white noise is white noise however it is
    # labelled -- so the two-coefficient solve is singular there and is skipped.
    det = gram_aa * gram_bb - gram_ab**2
    regular = det > 1e-12 * gram_aa * gram_bb
    safe_det = np.where(regular, det, 1.0)
    free_a = np.where(regular, (gram_bb * g_a - gram_ab * g_b) / safe_det, -1.0)
    free_b = np.where(regular, (gram_aa * g_b - gram_ab * g_a) / safe_det, -1.0)

    candidates = [
        (np.where((free_a >= 0) & (free_b >= 0), free_a, 0.0),
         np.where((free_a >= 0) & (free_b >= 0), free_b, 0.0)),
        (np.maximum(g_a / gram_aa, 0.0), np.zeros_like(g_a)),
        (np.zeros_like(g_a), np.full_like(g_a, max(g_b / gram_bb, 0.0))),
    ]
    best_value = np.zeros_like(g_a)  # the (0, 0) candidate, worth exactly 0
    best_a, best_b = np.zeros_like(g_a), np.zeros_like(g_a)
    for c_a, c_b in candidates:
        v = value(c_a, c_b)
        take = v > best_value
        best_value, best_a, best_b = (
            np.where(take, v, best_value),
            np.where(take, c_a, best_a),
            np.where(take, c_b, best_b),
        )
    return best_value, best_a, best_b


def _argmax_rho(criterion) -> float:
    """Grid search over rho, then golden-section refinement on the bracket."""
    grid = np.linspace(*GMM_RHO_BOUNDS, GMM_GRID_POINTS)
    best = int(np.argmax(criterion(grid)))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, GMM_GRID_POINTS - 1)]

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    # 40 halvings shrink the bracket to ~3e-11; the maximum is smooth, so its
    # location is only resolvable to ~1e-8 anyway.
    for _ in range(40):
        q_c, q_d = criterion(np.array([c, d]))
        if q_c > q_d:
            hi, d, c = d, c, d - phi * (d - lo)
        else:
            lo, c, d = c, d, c + phi * (hi - c)
    return float(0.5 * (lo + hi))


# --- Fits --------------------------------------------------------------------


def fit_ar1(C: np.ndarray) -> tuple[float, float]:
    """Minimum distance between C and sigma_eps^2 A(rho).  Returns (rho, sigma_eps^2)."""
    T = C.shape[0]
    bands, sizes = band_sums(C), band_sizes(T)
    rho = _argmax_rho(lambda r: criterion_ar1(r, bands, sizes))
    A = shape(T, rho)
    return rho, max(float((A * C).sum() / (A * A).sum()), 0.0)


def fit_me(C: np.ndarray) -> tuple[float, float, float]:
    """The same against sigma_eps^2 A(rho) + sigma_nu^2 B.

    Returns (rho, sigma_eps^2, sigma_nu^2).  Needs T >= 3: three parameters
    against the T distinct bands of a Toeplitz matrix, so T = 2 leaves one
    equation short.
    """
    T = C.shape[0]
    if T < 3:
        raise ValueError("the measurement-error fit needs T >= 3 to identify three parameters")
    bands, sizes = band_sums(C), band_sizes(T)
    rho = _argmax_rho(lambda r: criterion_me(r, bands, sizes)[0])
    _, c_a, c_b = criterion_me(np.array([rho]), bands, sizes)
    return rho, float(c_a[0]) * (1.0 + rho), float(c_b[0])
