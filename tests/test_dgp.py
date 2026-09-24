"""The panel must be stationary from t = 0 and reproducible from its seed."""

from __future__ import annotations

import numpy as np
import pytest

from sim.analytics import var_mu, var_u
from sim.config import RHO, SIGMA_ALPHA, SIGMA_EPS
from sim.dgp import seed_sequence, simulate_panel


def panel(model: str, T: int = 8, N: int = 200_000, rep: int = 0):
    rng = np.random.default_rng(seed_sequence(model, T, N, rep))
    return simulate_panel(model, T, N, rng)


@pytest.mark.parametrize("model", ["M0", "M1"])
def test_initial_condition_is_stationary(model):
    """Column means and variances must not drift between t = 0 and t = T."""
    y = panel(model)
    total_var = var_mu(RHO, SIGMA_ALPHA[model]) + var_u(RHO, SIGMA_EPS)
    variances = y.var(axis=0)
    assert variances == pytest.approx(total_var, rel=0.02)
    assert y.mean(axis=0) == pytest.approx(np.zeros(y.shape[1]), abs=0.02)


def test_first_order_autocovariance_matches_rho():
    y = panel("M0")
    rho_hat = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    assert rho_hat == pytest.approx(RHO, abs=0.01)


def test_seed_is_deterministic_and_design_specific():
    a = panel("M1", T=5, N=1_000, rep=0)
    b = panel("M1", T=5, N=1_000, rep=0)
    assert np.array_equal(a, b)

    for changed in (panel("M1", T=5, N=1_000, rep=1), panel("M0", T=5, N=1_000, rep=0)):
        assert not np.array_equal(a, changed)


def test_seed_entropy_carries_every_design_coordinate():
    ss = seed_sequence("M1", 10, 500, 3, master=0)
    assert list(ss.entropy) == [0, 1, 10, 500, 3]


def test_shape_and_short_panels_rejected():
    assert panel("M0", T=3, N=50).shape == (50, 4)
    with pytest.raises(ValueError):
        panel("M0", T=1, N=50)
