"""The growth covariance matrix, and the GMM estimator that fits it.

Omega is the one population object in SPEC.md that no other estimator uses, so
it gets its own checks: that it is what the simulated panels actually show, that
the individual effect is nowhere in it, and that inverting it recovers the
parameters it was built from.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.analytics import growth_cov, growth_shape, growth_shape_deriv, plim_fd, plim_sigma_eps
from sim.config import MODELS, RHO, SIGMA_EPS
from sim.dgp import seed_sequence, simulate_panel
from sim.estimators import Fit, _argmax_rho, _band_sizes, _band_sums, first_difference, gmm_growth

TOL = 0.005


def panel(model: str, T: int, N: int, rep: int = 0) -> np.ndarray:
    return simulate_panel(model, T, N, np.random.default_rng(seed_sequence(model, T, N, rep)))


# --- Omega, the population object --------------------------------------------


@pytest.mark.parametrize("T", [2, 3, 10])
def test_omega_matches_the_formula_in_spec(T):
    """2 sigma^2/(1+rho) on the diagonal, -sigma^2 (1-rho)/(1+rho) rho^(k-1) off it."""
    omega = growth_cov(T, RHO, SIGMA_EPS)
    assert omega == pytest.approx(omega.T)
    assert np.diag(omega) == pytest.approx(2 * SIGMA_EPS**2 / (1 + RHO))
    for k in range(1, T):
        band = np.diag(omega, k)
        expected = -SIGMA_EPS**2 * (1 - RHO) / (1 + RHO) * RHO ** (k - 1)
        assert band == pytest.approx(expected)


def test_omega_is_toeplitz_and_free_of_T():
    """The T x T matrix is the leading block of any larger one."""
    big = growth_cov(20)
    assert big[:6, :6] == pytest.approx(growth_cov(6))


def test_omega_is_a_covariance_matrix():
    assert np.linalg.eigvalsh(growth_cov(12)).min() > 0


def test_the_fd_plim_is_the_first_lag_ratio_of_omega():
    """SPEC.md's (rho-1)/2, read off Omega: the one number FD OLS uses."""
    omega = growth_cov(5)
    assert omega[1, 0] / omega[0, 0] == pytest.approx(plim_fd(RHO))


def test_omega_matches_the_simulated_growth_at_large_N():
    T, N = 6, 500_000
    for model in MODELS:
        d = np.diff(panel(model, T, N), axis=1)
        assert d.T @ d / N == pytest.approx(growth_cov(T), abs=2e-3)


def test_the_derivative_matches_a_numerical_one():
    step = 1e-6
    for rho in (-0.4, 0.0, 0.7, 0.9):
        numerical = (growth_shape(7, rho + step) - growth_shape(7, rho - step)) / (2 * step)
        assert growth_shape_deriv(7, rho) == pytest.approx(numerical, abs=1e-6)


# --- The estimator -----------------------------------------------------------


@pytest.mark.parametrize("T", [2, 3, 5, 50])
@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.7, 0.95])
def test_feeding_the_population_omega_returns_its_parameters(T, rho):
    """The gate SPEC.md asks for, at the precision a smooth maximum allows."""
    omega = growth_cov(T, rho, SIGMA_EPS)
    rho_hat = _argmax_rho(_band_sums(omega), _band_sizes(T))
    shape = growth_shape(T, rho_hat)

    assert rho_hat == pytest.approx(rho, abs=1e-6)
    sigma2 = (shape * omega).sum() / (shape * shape).sum()
    assert np.sqrt(sigma2) == pytest.approx(SIGMA_EPS, abs=1e-6)


def test_the_individual_effect_is_invisible_to_it():
    """Shifting every individual by a constant leaves the estimate untouched.

    That is why the GMM is consistent in M1: alpha_i enters y only through the
    level mu_i = alpha_i/(1-rho), and growth never sees a level.  The cancellation
    is exact in real arithmetic; in floating point the shift costs the last bits
    of each difference, and a smooth maximum turns 1e-16 in C into ~1e-8 in rho.
    """
    y = panel("M0", 8, 400)
    rng = np.random.default_rng(0)
    shifted = y + 5.0 * rng.standard_normal((y.shape[0], 1))

    base, moved = gmm_growth(y), gmm_growth(shifted)
    assert moved.rho_hat == pytest.approx(base.rho_hat, abs=1e-6)
    assert moved.sigma_eps_hat == pytest.approx(base.sigma_eps_hat, rel=1e-6)
    assert moved.se == pytest.approx(base.se, rel=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_it_recovers_both_parameters_at_large_N(model):
    fit = gmm_growth(panel(model, 10, 1_000_000))
    assert fit.rho_hat == pytest.approx(RHO, abs=TOL)
    assert fit.sigma_eps_hat == pytest.approx(SIGMA_EPS, abs=TOL)


def test_it_beats_first_differences_on_the_same_data():
    """Same differenced panel, one moment ratio against the whole matrix."""
    y = panel("M1", 10, 200_000)
    assert abs(gmm_growth(y).rho_hat - RHO) < 0.01
    assert first_difference(y).rho_hat < 0.0


@pytest.mark.parametrize("T", [3, 20])
def test_standard_errors_are_positive_and_shrink_with_N(T):
    small, large = gmm_growth(panel("M1", T, 500)), gmm_growth(panel("M1", T, 50_000))
    for fit in (small, large):
        assert fit.se > 0 and fit.sigma_eps_se > 0
    assert large.se < small.se
    assert large.sigma_eps_se < small.sigma_eps_se


def test_only_the_gmm_reports_a_sigma_eps():
    assert plim_sigma_eps("gmm") == SIGMA_EPS
    for estimator in ("pooled", "fd", "within"):
        assert np.isnan(plim_sigma_eps(estimator))
    assert np.isnan(Fit(0.7, 0.01).sigma_eps_hat)
