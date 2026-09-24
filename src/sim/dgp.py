"""Simulation of the panel AR(1), y_it = alpha_i + rho y_i,t-1 + eps_it.

The whole panel is one N x (T+1) array; the only loop is over t, so a
replication costs a handful of vectorised operations.
"""

from __future__ import annotations

import numpy as np

from .analytics import var_u
from .config import MASTER_SEED, MODEL_CODE, RHO, SIGMA_EPS, sigma_alpha


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

    # Drawn even when s_alpha == 0, so M0 and M1 differ only through this scale.
    alpha = s_alpha * rng.standard_normal(N)

    y = np.empty((N, T + 1), dtype=np.float64)
    y[:, 0] = alpha / (1.0 - rho) + np.sqrt(var_u(rho, s_eps)) * rng.standard_normal(N)

    eps = s_eps * rng.standard_normal((N, T))
    for t in range(1, T + 1):
        y[:, t] = alpha + rho * y[:, t - 1] + eps[:, t - 1]
    return y
