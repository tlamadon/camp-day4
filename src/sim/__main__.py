"""Command line entry point.

    uv run python -m sim run --model M1 --T 10 --N 500 --R 10 --seed 0

The same command serves the local pilot and the HPC scale-up; only R, N and the
output path change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import MASTER_SEED, MODELS, N_BASELINE, R_PILOT, RHO, T_GRID
from .runner import cell_name, run_and_write


def _default_output(model: str, T: int, N: int) -> Path:
    return Path("output") / "pilot" / f"{cell_name(model, T, N)}.parquet"


def cmd_run(args: argparse.Namespace) -> int:
    out = args.out or _default_output(args.model, args.T, args.N)
    run_and_write(args.model, args.T, args.N, args.R, Path(out), master_seed=args.seed)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from . import summarize

    summarize.build(args.output_dir, args.draws_dir)
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from . import figures

    figures.build(args.output_dir)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    report.build(args.output_dir)
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    from . import tables

    tables.build(args.output_dir)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .check import check_outputs

    ok, report = check_outputs(args.output_dir)
    print("\n".join(report))
    print("\n[check] provenance " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_plims(args: argparse.Namespace) -> int:
    from .analytics import plim

    print(f"rho = {RHO}")
    header = f"{'estimator':<10}{'model':<7}" + "".join(f"T={t:<8}" for t in T_GRID)
    print(header)
    for estimator in ("pooled", "fd", "within"):
        for model in MODELS:
            cells = "".join(f"{plim(estimator, model, t):<10.3f}" for t in T_GRID)
            print(f"{estimator:<10}{model:<7}{cells}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one design cell and write it to parquet")
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--T", type=int, required=True, help="number of regression periods")
    run.add_argument("--N", type=int, default=N_BASELINE, help="number of individuals")
    run.add_argument("--R", type=int, default=R_PILOT, help="replications")
    run.add_argument("--seed", type=int, default=MASTER_SEED, help="master seed")
    run.add_argument("--out", type=Path, default=None, help="parquet path")
    run.set_defaults(func=cmd_run)

    for name, func, help_text in [
        ("summary", cmd_summary, "collapse the raw draws into summary.csv and tables"),
        ("figures", cmd_figures, "draw Figures 1 and 2"),
        ("tables", cmd_tables, "write the LaTeX tables"),
        ("report", cmd_report, "write the one-page HTML summary"),
        ("check", cmd_check, "verify outputs match the current spec and commit"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--output-dir", type=Path, default=Path("output"))
        if name == "summary":
            p.add_argument("--draws-dir", type=Path, default=None)
        p.set_defaults(func=func)

    plims = sub.add_parser("plims", help="print the analytical benchmark table")
    plims.set_defaults(func=cmd_plims)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
