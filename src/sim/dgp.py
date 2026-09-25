"""Simulation of the panel.

M0 and M1 are the AR(1) in levels, y_it = alpha_i + rho y_i,t-1 + eps_it.  M2
puts the persistence in a latent component and adds measurement error on top,

    p_it = rho p_i,t-1 + eps_it,   y_it = alpha_i + p_it + nu_it,

so y is no longer an AR(1) -- it is an ARMA(1,1) around a level.  Both paths
build one N x (T+1) array with the only loop over t, so a replication costs a
handful of vectorised operations.
"""

from __future__ import annotations

import numpy as np

from .analytics import var_u
from .config import (
    MASTER_SEED,
    MODEL_CODE,
    RHO,
    SIGMA_EPS,
    effect_kind,
    sigma_alpha,
    sigma_nu,
)


def seed_sequence(model: str, T: int, N: int, rep: int, master: int = MASTER_SEED) -> np.random.SeedSequence:
    """Deterministic seed for replication `rep` of design (model, T, N).

    Derived from the master seed plus the four design coordinates, so any single
    cell -- or any single replication inside a cell -- can be rerun on its own.
    """
    if model not in MODEL_CODE:
        raise ValueError(f"unknown model {model!r}")
    return np.random.SeedSequence([master, MODEL_CODE[model], T, N, rep])


def simulate_panel(
    model: str,
    T: int,
    N: int,
    rng: np.random.Generator,
    rho: float = RHO,
    s_eps: float = SIGMA_EPS,
) -> np.ndarray:
    """Draw one panel.

    Returns an N x (T+1) array holding y_i0, ..., y_iT.  Column 0 is the initial
    condition, drawn from the stationary distribution so nothing depends on
    start-up transients.
    """
    if T < 2:
        raise ValueError("T must be at least 2 (the differencing estimators need t = 2..T)")
    s_alpha = sigma_alpha(model)

    if effect_kind(model) == "level":
        return _simulate_with_measurement_error(model, T, N, rng, rho, s_eps, s_alpha)

    # Drawn even when s_alpha == 0, so M0 and M1 differ only through this scale.
    alpha = s_alpha * rng.standard_normal(N)

    y = np.empty((N, T + 1), dtype=np.float64)
    y[:, 0] = alpha / (1.0 - rho) + np.sqrt(var_u(rho, s_eps)) * rng.standard_normal(N)

    eps = s_eps * rng.standard_normal((N, T))
    for t in range(1, T + 1):
        y[:, t] = alpha + rho * y[:, t - 1] + eps[:, t - 1]
    return y


def _simulate_with_measurement_error(
    model: str,
    T: int,
    N: int,
    rng: np.random.Generator,
    rho: float,
    s_eps: float,
    s_alpha: float,
) -> np.ndarray:
    """M2: a latent AR(1) plus a level plus noise, observed only through y.

    alpha_i is a level here rather than an intercept, so it needs no 1/(1 - rho);
    the persistent component p is exactly the process M0 and M1 observe directly.
    The measurement error is drawn for t = 0 as well -- y_i0 is an observation
    like any other.
    """
    alpha = s_alpha * rng.standard_normal(N)

    p = np.empty((N, T + 1), dtype=np.float64)
    p[:, 0] = np.sqrt(var_u(rho, s_eps)) * rng.standard_normal(N)
    eps = s_eps * rng.standard_normal((N, T))
    for t in range(1, T + 1):
        p[:, t] = rho * p[:, t - 1] + eps[:, t - 1]

    nu = sigma_nu(model) * rng.standard_normal((N, T + 1))
    return alpha[:, None] + p + nu
