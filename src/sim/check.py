"""Mechanical half of the spec audit: are the outputs built from current code and spec?

`make check` runs this and then asks make itself whether anything is stale.  The
spec-check skill layers the judgement-based checks on top.
"""

from __future__ import annotations

from pathlib import Path

from .config import MODELS, N_BASELINE, T_GRID
from .provenance import git_commit, repo_root, spec_sha256
from .runner import cell_name, read_cell_metadata

#: Files `make all` is expected to leave behind, relative to the output folder.
REQUIRED_DELIVERABLES = (
    "summary.csv",
    "table_main.md",
    "table_coverage.md",
    "table_sigma_eps.md",
    "figures/fig1_mean_rho_by_T.pdf",
    "figures/fig1_mean_rho_by_T.png",
    "figures/fig2_density_T10.pdf",
    "figures/fig2_density_T10.png",
)


def check_outputs(output_dir: Path, expect_cells: bool = True) -> tuple[bool, list[str]]:
    """Compare every output's stamped provenance against the current spec and commit."""
    output_dir = Path(output_dir)
    root = repo_root()
    want_spec = spec_sha256()
    want_commit = git_commit()

    report = [
        f"SPEC.md sha256 : {want_spec}",
        f"git commit     : {want_commit}",
        "",
    ]
    ok = True

    if want_spec == "missing":
        ok = False
        report.append("FAIL  SPEC.md not found at the repo root")
    if want_commit == "unknown":
        ok = False
        report.append("FAIL  not a git repository, so outputs cannot be tied to a commit")
    elif want_commit.endswith("-dirty"):
        ok = False
        report.append("FAIL  working tree has uncommitted changes to tracked files")

    paths = sorted((output_dir / "pilot").rglob("*.parquet"))
    if not paths:
        ok = False
        report.append(f"FAIL  no outputs under {output_dir / 'pilot'}")

    for path in paths:
        meta = read_cell_metadata(path)
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        problems = []
        if meta.get("spec_sha256") != want_spec:
            problems.append(f"built against spec {meta.get('spec_sha256', 'missing')[:12]}")
        if meta.get("git_commit") != want_commit:
            # Printed whole: a truncated sha would hide a '-dirty' suffix.
            problems.append(f"built at commit {meta.get('git_commit', 'missing')}")
        if problems:
            ok = False
            report.append(f"FAIL  {rel}: stale ({', '.join(problems)})")
        else:
            report.append(f"ok    {rel}")

    if expect_cells:
        for rel in REQUIRED_DELIVERABLES:
            if (output_dir / rel).is_file():
                report.append(f"ok    {output_dir / rel}")
            else:
                ok = False
                report.append(f"FAIL  missing deliverable: {output_dir / rel}")

        have = {p.stem for p in paths}
        missing = [
            cell_name(m, t, N_BASELINE)
            for m in MODELS
            for t in T_GRID
            if cell_name(m, t, N_BASELINE) not in have
        ]
        if missing:
            ok = False
            report.append(f"FAIL  missing pilot cells: {', '.join(missing)}")
        else:
            report.append(f"ok    all {len(MODELS) * len(T_GRID)} pilot cells present")

    return ok, report
