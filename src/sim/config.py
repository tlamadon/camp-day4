"""Design constants for the dynamic panel AR(1) Monte Carlo.

Every number here is fixed by SPEC.md; nothing else in the package hard-codes a
parameter value.  The Makefile can override the grid dimensions (N, R) on the
command line, which is how the HPC scale-up reuses the same code.
"""

from __future__ import annotations

# --- Data-generating process -------------------------------------------------

RHO = 0.7

#: Innovation to the persistent component, in every model.  In M0 and M1 that
#: component is y itself, so this is the sigma_eps of their AR(1) recursion.
SIGMA_EPS = 0.3

#: Transitory measurement error.  Only M2 has any.
SIGMA_NU: dict[str, float] = {"M0": 0.0, "M1": 0.0, "M2": 0.2}

#: How the individual effect enters: as the intercept of the AR(1) recursion
#: (M0, M1) or directly as a level (M2, where y is not an AR(1) at all).
EFFECT_KIND: dict[str, str] = {"M0": "intercept", "M1": "intercept", "M2": "level"}

#: Scale of the individual effect, in whichever way it enters.  M2's value is
#: M1's intercept read as the level it implies, alpha_i/(1 - rho), so the two
#: models share a between-individual variance and differ only by the noise.
SIGMA_ALPHA: dict[str, float] = {"M0": 0.0, "M1": 0.5, "M2": 0.5 / (1.0 - RHO)}

MODELS: tuple[str, ...] = ("M0", "M1", "M2")

#: Stable integer per model, mixed into the seed so the models get different draws.
MODEL_CODE: dict[str, int] = {"M0": 0, "M1": 1, "M2": 2}

# --- Design grid -------------------------------------------------------------

T_GRID: tuple[int, ...] = (3, 5, 10, 20, 50)
N_BASELINE = 500
R_PILOT = 10
MASTER_SEED = 0

# --- Estimators --------------------------------------------------------------

ESTIMATORS: tuple[str, ...] = ("pooled", "fd", "within", "gmm", "gmm_me")

ESTIMATOR_LABELS: dict[str, str] = {
    "pooled": "Pooled OLS",
    "fd": "First-difference OLS",
    "within": "Within (FE)",
    "gmm": "Growth-covariance GMM",
    "gmm_me": "Growth-covariance GMM, with ME",
}

#: Which variance parameters each estimator reports beside rho.
ESTIMATES_SIGMA_EPS: tuple[str, ...] = ("gmm", "gmm_me")
ESTIMATES_SIGMA_NU: tuple[str, ...] = ("gmm_me",)

#: Short forms, for the direct labels on the figures where space is tight.
ESTIMATOR_SHORT_LABELS: dict[str, str] = {
    "pooled": "Pooled OLS",
    "fd": "First differences",
    "within": "Within (FE)",
    "gmm": "Growth GMM",
    "gmm_me": "Growth GMM + ME",
}

MODEL_LABELS: dict[str, str] = {
    "M0": "M0: no effects",
    "M1": "M1: fixed effects",
    "M2": "M2: effects + measurement error",
}

#: Normal quantile behind the 95% confidence intervals (SPEC.md fixes 1.96).
Z95 = 1.96

# --- Growth-covariance GMM ---------------------------------------------------

#: The profiled criterion is searched exactly as SPEC.md fixes it: this grid of
#: rho, then golden-section refinement on the interval bracketing the best point.
GMM_RHO_BOUNDS: tuple[float, float] = (-0.995, 0.995)
GMM_GRID_POINTS = 1001


def _lookup(table: dict[str, float | str], model: str):
    try:
        return table[model]
    except KeyError:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}") from None


def sigma_alpha(model: str) -> float:
    return float(_lookup(SIGMA_ALPHA, model))


def sigma_nu(model: str) -> float:
    return float(_lookup(SIGMA_NU, model))


def effect_kind(model: str) -> str:
    return str(_lookup(EFFECT_KIND, model))
