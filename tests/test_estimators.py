"""The estimators must equal their textbook regressions, and hit their plims at large N.

The large-N test is the gate SPEC.md puts before running the grid: at N = 10^6
and T = 10 each estimator should land within 0.005 of its benchmark plim.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.analytics import plim
from sim.config import MODELS, RHO
from sim.dgp import seed_sequence, simulate_panel
from sim.estimators import ESTIMATOR_FUNCS, first_difference, pooled_ols, within

BIG_N = 1_000_000
BIG_T = 10
TOL = 0.005


@pytest.fixture(scope="module")
def big_panels():
    out = {}
    for model in MODELS:
        rng = np.random.default_rng(seed_sequence(model, BIG_T, BIG_N, 0))
        out[model] = simulate_panel(model, BIG_T, BIG_N, rng)
    return out


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("estimator", ["pooled", "fd", "within"])
def test_estimator_hits_its_plim_at_large_N(big_panels, estimator, model):
    fit = ESTIMATOR_FUNCS[estimator](big_panels[model])
    assert fit.rho_hat == pytest.approx(plim(estimator, model, BIG_T), abs=TOL)


def test_the_expected_story_holds_at_large_N(big_panels):
    """Pooled OLS is fine without effects and badly biased with them; FD and within are not."""
    assert pooled_ols(big_panels["M0"]).rho_hat == pytest.approx(RHO, abs=TOL)
    assert pooled_ols(big_panels["M1"]).rho_hat > 0.95
    for model in MODELS:
        assert first_difference(big_panels[model]).rho_hat < 0.0
        assert within(big_panels[model]).rho_hat < RHO - 0.15


# --- equivalence with the corresponding least-squares fits --------------------


def small_panel(model: str = "M1", T: int = 6, N: int = 40):
    rng = np.random.default_rng(seed_sequence(model, T, N, 7))
    return simulate_panel(model, T, N, rng)


def cluster_se(X: np.ndarray, resid: np.ndarray, n_clusters: int, per_cluster: int) -> np.ndarray:
    """Textbook cluster sandwich, written out in full for the comparison."""
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros_like(xtx_inv)
    for i in range(n_clusters):
        block = slice(i * per_cluster, (i + 1) * per_cluster)
        score = X[block].T @ resid[block]
        meat += np.outer(score, score)
    correction = n_clusters / (n_clusters - 1.0)
    return np.sqrt(np.diag(correction * xtx_inv @ meat @ xtx_inv))


def test_pooled_matches_ols_with_a_constant():
    y = small_panel()
    N, T = y.shape[0], y.shape[1] - 1
    X = np.column_stack([np.ones(N * T), y[:, :-1].ravel()])
    dep = y[:, 1:].ravel()

    beta = np.linalg.lstsq(X, dep, rcond=None)[0]
    se = cluster_se(X, dep - X @ beta, N, T)

    fit = pooled_ols(y)
    assert fit.rho_hat == pytest.approx(beta[1], rel=1e-12)
    assert fit.se == pytest.approx(se[1], rel=1e-10)


def test_first_difference_matches_ols_on_differences_without_a_constant():
    y = small_panel()
    N, T = y.shape[0], y.shape[1] - 1
    d = np.diff(y, axis=1)
    X = d[:, :-1].reshape(-1, 1)
    dep = d[:, 1:].ravel()

    beta = np.linalg.lstsq(X, dep, rcond=None)[0]
    se = cluster_se(X, dep - X @ beta, N, T - 1)

    fit = first_difference(y)
    assert fit.rho_hat == pytest.approx(beta[0], rel=1e-12)
    assert fit.se == pytest.approx(se[0], rel=1e-10)


def test_within_matches_the_dummy_variable_regression():
    """Separate demeaning of y and its lag reproduces LSDV exactly."""
    y = small_panel(T=5, N=25)
    N, T = y.shape[0], y.shape[1] - 1
    dummies = np.repeat(np.eye(N), T, axis=0)
    X = np.column_stack([y[:, :-1].ravel(), dummies])
    dep = y[:, 1:].ravel()

    beta = np.linalg.lstsq(X, dep, rcond=None)[0]
    assert within(y).rho_hat == pytest.approx(beta[0], rel=1e-10)


def test_estimators_use_the_periods_the_spec_says():
    """FD drops one more period: truncating the panel by one must reproduce it."""
    y = small_panel(T=7, N=30)
    assert first_difference(y).rho_hat == pytest.approx(
        pooled_ols_on_differences(y), rel=1e-12
    )


def pooled_ols_on_differences(y: np.ndarray) -> float:
    d = np.diff(y, axis=1)
    x, dep = d[:, :-1], d[:, 1:]
    return float((x * dep).sum() / (x * x).sum())


@pytest.mark.parametrize("estimator", ["pooled", "fd", "within"])
def test_standard_errors_are_positive_and_shrink_with_N(estimator):
    small = ESTIMATOR_FUNCS[estimator](small_panel(T=6, N=200))
    large = ESTIMATOR_FUNCS[estimator](small_panel(T=6, N=20_000))
    assert small.se > 0 and large.se > 0
    assert large.se < small.se
