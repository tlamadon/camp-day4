"""The plims must reproduce the benchmark tables in SPEC.md exactly as printed."""

from __future__ import annotations

import pytest

from sim.analytics import nickell_bias, plim, plim_fd, plim_pooled, plim_within, var_mu, var_u
from sim.config import ESTIMATORS, MODELS, RHO, SIGMA_ALPHA, SIGMA_EPS, T_GRID

# SPEC.md, "Data-generating process": the variance decomposition quoted there.
VARIANCE_NOTE = {"var_mu_M1": 2.78, "var_u": 0.176, "ratio": 16}

# SPEC.md, "Analytical benchmarks": plim by estimator and model.
PLIM_TABLE = {
    ("pooled", "M0"): 0.700,
    ("pooled", "M1"): 0.982,
    ("fd", "M0"): -0.150,
    ("fd", "M1"): -0.150,
    ("gmm", "M0"): 0.700,
    ("gmm", "M1"): 0.700,
}

# SPEC.md, "Within (Nickell 1981)": bias and plim by T.
NICKELL_TABLE = {
    3: (-0.620, 0.080),
    5: (-0.394, 0.306),
    10: (-0.197, 0.503),
    20: (-0.094, 0.606),
    50: (-0.036, 0.664),
}


def test_variance_decomposition_matches_spec():
    assert var_mu(RHO, SIGMA_ALPHA["M1"]) == pytest.approx(VARIANCE_NOTE["var_mu_M1"], abs=5e-3)
    assert var_u(RHO, SIGMA_EPS) == pytest.approx(VARIANCE_NOTE["var_u"], abs=5e-4)
    ratio = var_mu(RHO, SIGMA_ALPHA["M1"]) / var_u(RHO, SIGMA_EPS)
    assert ratio == pytest.approx(VARIANCE_NOTE["ratio"], abs=0.5)


@pytest.mark.parametrize(("estimator", "model"), sorted(PLIM_TABLE))
def test_pooled_and_fd_plims_match_spec(estimator, model):
    expected = PLIM_TABLE[(estimator, model)]
    for T in T_GRID:  # neither one depends on T
        assert plim(estimator, model, T) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("T", sorted(NICKELL_TABLE))
def test_nickell_table_matches_spec(T):
    bias, level = NICKELL_TABLE[T]
    assert nickell_bias(T) == pytest.approx(bias, abs=5e-4)
    assert plim_within(T) == pytest.approx(level, abs=5e-4)


@pytest.mark.parametrize("T", sorted(NICKELL_TABLE))
def test_within_plim_is_the_same_in_both_models(T):
    """Demeaning removes alpha_i exactly, so M0 and M1 share the Nickell bias."""
    assert plim("within", "M0", T) == plim("within", "M1", T)


def test_leading_term_approximates_the_bias_as_T_grows():
    """The -(1+rho)/(T-1) column of the SPEC table is the large-T approximation."""
    errors = [abs(nickell_bias(T) - (-(1 + RHO) / (T - 1))) for T in (10, 20, 50)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3


def test_pooled_plim_collapses_to_rho_without_effects():
    assert plim_pooled(RHO, SIGMA_EPS, 0.0) == pytest.approx(RHO)


def test_fd_plim_is_free_of_T_sigma_eps_and_sigma_alpha():
    assert plim_fd(RHO) == pytest.approx((RHO - 1) / 2)
    assert plim_fd(0.3) == pytest.approx(-0.35)


def test_every_grid_cell_has_a_plim():
    values = [plim(e, m, T) for e in ESTIMATORS for m in MODELS for T in T_GRID]
    assert len(values) == len(ESTIMATORS) * len(MODELS) * len(T_GRID)
    assert all(abs(v) < 2 for v in values)


def test_the_gmm_plim_is_the_truth_in_both_models_at_every_T():
    """Consistency: Omega is free of alpha_i, so M0 and M1 share one plim."""
    for model in MODELS:
        for T in T_GRID:
            assert plim("gmm", model, T) == RHO
