"""Design constants for the dynamic panel AR(1) Monte Carlo.

Every number here is fixed by SPEC.md; nothing else in the package hard-codes a
parameter value.  The Makefile can override the grid dimensions (N, R) on the
command line, which is how the HPC scale-up reuses the same code.
"""

from __future__ import annotations

# --- Data-generating process -------------------------------------------------

RHO = 0.7
SIGMA_EPS = 0.3

#: sigma_alpha per model.  M0 has no individual effects, M1 has alpha_i ~ N(0, 0.5^2).
SIGMA_ALPHA: dict[str, float] = {"M0": 0.0, "M1": 0.5}

MODELS: tuple[str, ...] = ("M0", "M1")

#: Stable integer per model, mixed into the seed so M0 and M1 get different draws.
MODEL_CODE: dict[str, int] = {"M0": 0, "M1": 1}

# --- Design grid -------------------------------------------------------------

T_GRID: tuple[int, ...] = (3, 5, 10, 20, 50)
N_BASELINE = 500
R_PILOT = 10
MASTER_SEED = 0

# --- Estimators --------------------------------------------------------------

ESTIMATORS: tuple[str, ...] = ("pooled", "fd", "within")

ESTIMATOR_LABELS: dict[str, str] = {
    "pooled": "Pooled OLS",
    "fd": "First-difference OLS",
    "within": "Within (FE)",
}

MODEL_LABELS: dict[str, str] = {
    "M0": "M0: no effects",
    "M1": "M1: fixed effects",
}

#: Normal quantile behind the 95% confidence intervals (SPEC.md fixes 1.96).
Z95 = 1.96


def sigma_alpha(model: str) -> float:
    try:
        return SIGMA_ALPHA[model]
    except KeyError:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}") from None
