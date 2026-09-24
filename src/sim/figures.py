"""Figures 1 and 2 of SPEC.md.

Figure 1: mean rho-hat against T, one line per estimator, two panels (M0, M1),
dashed analytical plims and a reference line at the true rho.
Figure 2: densities of rho-hat at T = 10 for the six estimator x model cells.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analytics import plim  # noqa: E402
from .config import ESTIMATOR_LABELS, ESTIMATORS, MODEL_LABELS, MODELS, N_BASELINE, RHO  # noqa: E402
from .runner import load_cells  # noqa: E402

# Categorical slots 1-3 of the validated default palette: the three-slot set
# clears the all-pairs CVD and normal-vision floors in both modes.  Aqua sits
# below 3:1 on the light surface, so both figures carry direct labels (and the
# markdown tables are the table view) as the required relief.
SERIES = {"pooled": "#2a78d6", "fd": "#eb6834", "within": "#1baf7a"}
MARKERS = {"pooled": "o", "fd": "s", "within": "^"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e4df"

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _style_axes(ax, *, y_grid: bool = True, x_grid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=3, width=0.8)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    if x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, target: Path, stem: str) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        path = target / f"{stem}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    print(f"[figures] -> {target / stem}.png, .pdf")
    return written


def figure_mean_by_T(summary: pd.DataFrame, target: Path, N: int = N_BASELINE) -> list[Path]:
    """Figure 1: mean rho-hat against T, two panels."""
    block = summary[summary["N"] == N]
    t_values = sorted(block["T"].unique())

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)

        for ax, model in zip(axes, MODELS, strict=True):
            _style_axes(ax)
            ax.axhline(RHO, color=INK_MUTED, linewidth=1.0, zorder=1)
            ax.annotate(
                f"true $\\rho$ = {RHO}",
                xy=(t_values[0], RHO),
                xytext=(0, 4),
                textcoords="offset points",
                color=INK_MUTED,
                fontsize=8,
                ha="left",
                va="bottom",
            )

            for estimator in ESTIMATORS:
                cell = block[(block["model"] == model) & (block["estimator"] == estimator)]
                cell = cell.sort_values("T")
                if cell.empty:
                    continue
                color = SERIES[estimator]

                ax.plot(
                    cell["T"],
                    cell["plim"],
                    color=color,
                    linewidth=1.4,
                    linestyle=(0, (4, 3)),
                    alpha=0.55,
                    zorder=2,
                )
                ax.errorbar(
                    cell["T"],
                    cell["mean_rho"],
                    yerr=1.96 * cell["mc_se"],
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=0,
                    linewidth=2.0,
                    marker=MARKERS[estimator],
                    markersize=5.5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    label=ESTIMATOR_LABELS[estimator],
                    zorder=3,
                )

                if model == MODELS[-1]:  # direct labels once, on the right panel
                    last = cell.iloc[-1]
                    # Drop the label below the line when it would sit on the
                    # rho = 0.7 reference.
                    near_reference = abs(last["mean_rho"] - RHO) < 0.08
                    ax.annotate(
                        ESTIMATOR_LABELS[estimator],
                        xy=(last["T"], last["mean_rho"]),
                        xytext=(-2, -8 if near_reference else 7),
                        textcoords="offset points",
                        color=INK_SECONDARY,
                        fontsize=8.5,
                        ha="right",
                        va="top" if near_reference else "bottom",
                    )

            ax.set_xscale("log")
            ax.set_xticks(t_values)
            ax.set_xticklabels([str(t) for t in t_values])
            ax.minorticks_off()
            ax.set_xlim(t_values[0] * 0.82, t_values[-1] * 1.22)
            ax.set_xlabel("Panel length $T$")
            ax.set_title(MODEL_LABELS[model], color=INK, loc="left", pad=8)

        axes[0].set_ylim(-0.33, 1.14)
        axes[0].set_ylabel(r"Mean $\hat\rho$ across replications")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

        reps = int(block["R"].max())
        fig.suptitle(
            r"Solid: simulated mean $\hat\rho$ (95% Monte Carlo interval).  "
            r"Dashed: analytical plim."
            f"   N = {N}, R = {reps}.",
            x=0.0,
            y=1.04,
            ha="left",
            fontsize=8.5,
            color=INK_MUTED,
        )
        fig.tight_layout()
        return _save(fig, target, "fig1_mean_rho_by_T")


def _bandwidth(sample: np.ndarray) -> float:
    """Silverman's rule of thumb."""
    n = sample.size
    sd = float(sample.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(sample, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(spread) or spread <= 0:
        spread = max(sd, 1e-4)
    return 0.9 * spread * n ** (-0.2)


def _gaussian_kde(sample: np.ndarray, pad: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Peak-normalised Gaussian KDE on a grid local to the sample (no scipy).

    The grid stops a few bandwidths either side of the data, so a density never
    draws a flat line across the parts of the axis where it has no mass.
    """
    sample = np.asarray(sample, dtype=float)
    h = _bandwidth(sample)
    grid = np.linspace(sample.min() - pad * h, sample.max() + pad * h, 801)

    z = (grid[:, None] - sample[None, :]) / h
    density = np.exp(-0.5 * z**2).sum(axis=1) / (sample.size * h * np.sqrt(2 * np.pi))
    peak = density.max()
    return grid, (density / peak if peak > 0 else density)


def figure_density_at_T(
    draws: pd.DataFrame, target: Path, T: int = 10, N: int = N_BASELINE
) -> list[Path]:
    """Figure 2: densities of rho-hat at one T, for the six estimator x model cells."""
    block = draws[(draws["T"] == T) & (draws["N"] == N)]
    if block.empty:
        raise ValueError(f"no draws at T={T}, N={N}")

    lo, hi = float(block["rho_hat"].min()), float(block["rho_hat"].max())
    pad = 0.06 * max(hi - lo, 0.2)
    xlim = (min(lo, RHO) - pad, max(hi, RHO) + pad)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True, sharey=True)

        for ax, model in zip(axes, MODELS, strict=True):
            _style_axes(ax, y_grid=False, x_grid=True)
            ax.axvline(RHO, color=INK_MUTED, linewidth=1.0, zorder=1)

            for estimator in ESTIMATORS:
                sample = block[(block["model"] == model) & (block["estimator"] == estimator)][
                    "rho_hat"
                ].to_numpy()
                if sample.size < 2:
                    continue
                color = SERIES[estimator]
                grid, density = _gaussian_kde(sample)

                ax.axvline(
                    plim(estimator, model, T),
                    color=color,
                    linewidth=1.2,
                    linestyle=(0, (4, 3)),
                    alpha=0.55,
                    zorder=2,
                )
                ax.fill_between(grid, density, color=color, alpha=0.18, linewidth=0, zorder=3)
                ax.plot(grid, density, color=color, linewidth=2.0, zorder=4,
                        label=ESTIMATOR_LABELS[estimator])
                # Rug: every replication drawn, so a narrow spike cannot be missed.
                ax.plot(
                    sample,
                    np.full(sample.shape, -0.045),
                    marker="|",
                    markersize=6,
                    markeredgewidth=1.2,
                    linestyle="none",
                    color=color,
                    zorder=4,
                )
                ax.annotate(
                    ESTIMATOR_LABELS[estimator],
                    xy=(float(np.mean(sample)), 1.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=INK_SECONDARY,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

            ax.set_yticks([])
            ax.set_ylim(-0.1, 1.30)
            ax.set_xlim(*xlim)
            ax.set_ylabel("Density")
            ax.set_title(MODEL_LABELS[model], color=INK, loc="left", pad=6)

        # Labelled on the lower panel, where the space right of the line is free.
        axes[-1].annotate(
            f"true $\\rho$ = {RHO}",
            xy=(RHO, 1.27),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8,
            ha="left",
            va="top",
        )
        axes[-1].set_xlabel(r"$\hat\rho$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.05))

        reps = int(block["R"].max())
        fig.suptitle(
            f"Densities of $\\hat\\rho$ at T = {T}, N = {N}, R = {reps}.  "
            "Dashed: analytical plim.  Ticks: the individual replications.  "
            "Each density is scaled to its own peak.",
            x=0.0,
            y=1.03,
            ha="left",
            fontsize=8.5,
            color=INK_MUTED,
        )
        fig.tight_layout()
        return _save(fig, target, "fig2_density_T10")


def build(output_dir: Path, draws_dir: Path | None = None) -> list[Path]:
    output_dir = Path(output_dir)
    draws_dir = Path(draws_dir) if draws_dir is not None else output_dir / "pilot"
    target = output_dir / "figures"

    summary = pd.read_csv(output_dir / "summary.csv")
    draws = load_cells(draws_dir)

    written = figure_mean_by_T(summary, target)
    written += figure_density_at_T(draws, target)
    return written
