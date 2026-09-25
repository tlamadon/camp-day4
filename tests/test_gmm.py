"""The growth covariance matrix, and the two estimators that fit it.

Omega is the one population object in SPEC.md that no regression uses, so it
gets its own checks: that it is what the simulated panels actually show, that
the individual effect is nowhere in it, and that inverting it recovers the
parameters it was built from.  The two fits differ only in which shape they
invert, which is what M2 is there to expose.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import growth
from sim.analytics import growth_plim, plim_fd, plim_param
from sim.config import MODELS, RHO, SIGMA_EPS, SIGMA_NU, sigma_nu
from sim.dgp import seed_sequence, simulate_panel
from sim.estimators import Fit, first_difference, gmm_growth, gmm_growth_me

TOL = 0.005


def panel(model: str, T: int, N: int, rep: int = 0) -> np.ndarray:
    return simulate_panel(model, T, N, np.random.default_rng(seed_sequence(model, T, N, rep)))


# --- Omega, the population object --------------------------------------------


@pytest.mark.parametrize("T", [2, 3, 10])
def test_omega_matches_the_formula_in_spec(T):
    """2 sigma^2/(1+rho) on the diagonal, -sigma^2 (1-rho)/(1+rho) rho^(k-1) off it."""
    omega = growth.cov(T, RHO, SIGMA_EPS)
    assert omega == pytest.approx(omega.T)
    assert np.diag(omega) == pytest.approx(2 * SIGMA_EPS**2 / (1 + RHO))
    for k in range(1, T):
        band = np.diag(omega, k)
        expected = -(SIGMA_EPS**2) * (1 - RHO) / (1 + RHO) * RHO ** (k - 1)
        assert band == pytest.approx(expected)


def test_measurement_error_adds_an_MA1_and_nothing_else():
    """Differencing white noise leaves 2 sigma_nu^2 and -sigma_nu^2, then zero."""
    s_nu = 0.2
    extra = growth.cov(8, RHO, SIGMA_EPS, s_nu) - growth.cov(8, RHO, SIGMA_EPS)
    assert np.diag(extra) == pytest.approx(2 * s_nu**2)
    assert np.diag(extra, 1) == pytest.approx(-(s_nu**2))
    for k in range(2, 8):
        assert np.diag(extra, k) == pytest.approx(0.0, abs=1e-15)


def test_omega_is_toeplitz_and_free_of_T():
    """The T x T matrix is the leading block of any larger one."""
    assert growth.cov(20)[:6, :6] == pytest.approx(growth.cov(6))
    assert growth.cov(20, s_nu=0.2)[:6, :6] == pytest.approx(growth.cov(6, s_nu=0.2))


@pytest.mark.parametrize("s_nu", [0.0, 0.2])
def test_omega_is_a_covariance_matrix(s_nu):
    assert np.linalg.eigvalsh(growth.cov(12, s_nu=s_nu)).min() > 0


def test_the_fd_plim_is_the_first_lag_ratio_of_omega():
    """SPEC.md's (rho-1)/2, read off Omega: the one number FD OLS uses."""
    omega = growth.cov(5)
    assert omega[1, 0] / omega[0, 0] == pytest.approx(plim_fd(RHO))


@pytest.mark.parametrize("model", MODELS)
def test_omega_matches_the_simulated_growth_at_large_N(model):
    T, N = 6, 500_000
    d = np.diff(panel(model, T, N), axis=1)
    assert d.T @ d / N == pytest.approx(growth.cov(T, s_nu=sigma_nu(model)), abs=2e-3)


def test_the_derivative_matches_a_numerical_one():
    step = 1e-6
    for rho in (-0.4, 0.0, 0.7, 0.9):
        numerical = (growth.shape(7, rho + step) - growth.shape(7, rho - step)) / (2 * step)
        assert growth.shape_deriv(7, rho) == pytest.approx(numerical, abs=1e-6)


# --- The fits ----------------------------------------------------------------


@pytest.mark.parametrize("T", [2, 3, 5, 50])
@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.7, 0.95])
def test_the_ar1_fit_returns_the_parameters_of_its_own_omega(T, rho):
    """The gate SPEC.md asks for, at the precision a smooth maximum allows."""
    rho_hat, var_eps = growth.fit_ar1(growth.cov(T, rho, SIGMA_EPS))
    assert rho_hat == pytest.approx(rho, abs=1e-6)
    assert np.sqrt(var_eps) == pytest.approx(SIGMA_EPS, abs=1e-6)


@pytest.mark.parametrize("T", [3, 5, 50])
@pytest.mark.parametrize("rho", [-0.5, 0.3, 0.7, 0.95])
def test_the_me_fit_returns_all_three_of_its_own(T, rho):
    rho_hat, var_eps, var_nu = growth.fit_me(growth.cov(T, rho, SIGMA_EPS, 0.2))
    assert rho_hat == pytest.approx(rho, abs=1e-6)
    assert np.sqrt(var_eps) == pytest.approx(SIGMA_EPS, abs=1e-6)
    assert np.sqrt(var_nu) == pytest.approx(0.2, abs=1e-6)


def test_the_me_fit_needs_three_periods():
    """Three parameters against the T distinct bands of a Toeplitz matrix."""
    with pytest.raises(ValueError):
        growth.fit_me(growth.cov(2, RHO, SIGMA_EPS, 0.2))


def test_the_me_fit_finds_no_error_when_there_is_none():
    """sigma_nu = 0 is the boundary, and the fit sits on it rather than crossing."""
    for T in (3, 10, 50):
        rho_hat, var_eps, var_nu = growth.fit_me(growth.cov(T, RHO, SIGMA_EPS))
        assert rho_hat == pytest.approx(RHO, abs=1e-5)
        assert np.sqrt(var_eps) == pytest.approx(SIGMA_EPS, abs=1e-5)
        assert 0.0 <= np.sqrt(var_nu) < 1e-3


def test_ignoring_measurement_error_costs_the_ar1_fit_a_third_of_rho():
    """The pseudo-true value under M2: the criterion's own answer, not an approximation."""
    for T in (10, 50):
        rho_star, sigma_star, _ = growth_plim("gmm", "M2", T)
        assert rho_star == pytest.approx(0.370, abs=0.002)
        # With no sigma_nu to hold it, the noise is loaded onto sigma_eps instead.
        assert sigma_star > SIGMA_EPS


def test_the_individual_effect_is_invisible_to_both_fits():
    """Shifting every individual by a constant leaves the estimates untouched.

    That is why they are consistent with fixed effects: alpha_i enters y only as
    a level, and growth never sees a level.  The cancellation is exact in real
    arithmetic; in floating point the shift costs the last bits of each
    difference, and a smooth maximum turns 1e-16 in C into ~1e-8 in rho.
    """
    y = panel("M0", 8, 400)
    rng = np.random.default_rng(0)
    shifted = y + 5.0 * rng.standard_normal((y.shape[0], 1))

    for estimator in (gmm_growth, gmm_growth_me):
        base, moved = estimator(y), estimator(shifted)
        assert moved.rho_hat == pytest.approx(base.rho_hat, abs=1e-6)
        assert moved.sigma_eps_hat == pytest.approx(base.sigma_eps_hat, rel=1e-6)
        assert moved.se == pytest.approx(base.se, rel=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_the_me_fit_recovers_every_parameter_at_large_N(model):
    fit = gmm_growth_me(panel(model, 10, 1_000_000))
    assert fit.rho_hat == pytest.approx(RHO, abs=TOL)
    assert fit.sigma_eps_hat == pytest.approx(SIGMA_EPS, abs=TOL)
    assert fit.sigma_nu_hat == pytest.approx(SIGMA_NU[model], abs=0.02)


def test_only_the_me_fit_survives_measurement_error():
    """The point of M2: same data, same moments, one fits the right shape."""
    y = panel("M2", 10, 500_000)
    assert gmm_growth_me(y).rho_hat == pytest.approx(RHO, abs=0.01)
    assert gmm_growth(y).rho_hat == pytest.approx(0.370, abs=0.01)
    assert first_difference(y).rho_hat == pytest.approx(-0.301, abs=0.01)


@pytest.mark.parametrize("estimator", [gmm_growth, gmm_growth_me])
@pytest.mark.parametrize("T", [3, 20])
def test_standard_errors_are_positive_and_shrink_with_N(estimator, T):
    small, large = estimator(panel("M2", T, 500)), estimator(panel("M2", T, 50_000))
    for fit in (small, large):
        assert fit.se > 0 and fit.sigma_eps_se > 0
    assert large.se < small.se
    assert large.sigma_eps_se < small.sigma_eps_se


def test_a_variance_pinned_at_zero_reports_no_standard_error():
    """The boundary is where the usual asymptotics stop, so the SE is NaN not small."""
    sd, se = Fit(0.7, 0.01).sigma_eps_hat, Fit(0.7, 0.01).sigma_eps_se
    assert np.isnan(sd) and np.isnan(se)

    fits = [gmm_growth_me(panel("M0", 5, 300, rep)) for rep in range(20)]
    pinned = [f for f in fits if f.sigma_nu_hat == 0.0]
    assert pinned, "with no measurement error some replications should hit the boundary"
    assert all(np.isnan(f.sigma_nu_se) for f in pinned)


def test_which_estimators_report_which_parameters():
    for model in MODELS:
        assert plim_param("sigma_eps", "gmm_me", model, 10) == pytest.approx(SIGMA_EPS)
        assert plim_param("sigma_nu", "gmm_me", model, 10) == pytest.approx(
            SIGMA_NU[model], abs=1e-4
        )
        assert np.isnan(plim_param("sigma_nu", "gmm", model, 10))
        for estimator in ("pooled", "fd", "within"):
            assert np.isnan(plim_param("sigma_eps", estimator, model, 10))
