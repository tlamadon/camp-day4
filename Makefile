# Dynamic panel AR(1) Monte Carlo -- the whole task graph.
#
# Every simulation cell is a file target, so `make -j` runs cells in parallel and
# reruns only what is stale.  The design grid is defined here once; the HPC
# scale-up overrides it on the command line, e.g.
#
#     make sim R=1000 N=5000
#
# Assumes `mise install` has provisioned python and uv (see mise.toml).

# --- design grid -------------------------------------------------------------

MODELS := M0 M1 M2
TS     := 3 5 10 20 50
N      := 500
R      := 10
SEED   := 0

# --- paths -------------------------------------------------------------------

PY     := uv run python
OUT    := output
PILOT  := $(OUT)/pilot
STAMP  := .make
LEAN   := proofs
SRC    := $(wildcard src/sim/*.py)
TESTS  := $(wildcard tests/*.py)

CELLS  := $(foreach m,$(MODELS),$(foreach t,$(TS),$(PILOT)/$(m)_T$(t)_N$(N).parquet))

SUMMARY    := $(OUT)/summary.csv
SUMMARY_MD := $(OUT)/table_main.md $(OUT)/table_coverage.md $(OUT)/table_variances.md
FIG1       := $(OUT)/figures/fig1_mean_rho_by_T.pdf
FIG2       := $(OUT)/figures/fig2_density_T10.pdf
TABLE1     := $(OUT)/tables/table_main.tex
TABLE2     := $(OUT)/tables/table_coverage.tex
TABLE3     := $(OUT)/tables/table_variances.tex
REPORT     := $(OUT)/report.html
FRAGMENT   := $(OUT)/artifact.html

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.PHONY: all setup test sim summary figures tables report proof proof-numeric check clean distclean help

## all: the default target -- figures and summary tables
all: figures summary

## setup: install the pinned dependencies
setup: $(STAMP)/setup
$(STAMP)/setup: pyproject.toml uv.lock
	uv sync
	@mkdir -p $(STAMP) && touch $@

## test: unit tests, including the plim gate at N = 10^6
test: $(STAMP)/test
$(STAMP)/test: $(STAMP)/setup $(SRC) $(TESTS)
	$(PY) -m pytest
	@mkdir -p $(STAMP) && touch $@

# One cell per (model, T, N); the stem carries the design point, e.g. M1_T10_N500.
$(PILOT)/%.parquet: $(STAMP)/test $(SRC) SPEC.md
	$(PY) -m sim run \
	  --model $(word 1,$(subst _, ,$*)) \
	  --T $(patsubst T%,%,$(word 2,$(subst _, ,$*))) \
	  --N $(patsubst N%,%,$(word 3,$(subst _, ,$*))) \
	  --R $(R) --seed $(SEED) --out $@

## sim: every pilot cell (2 models x 5 values of T)
sim: $(CELLS)

## summary: bias, SD, RMSE and coverage tables, for rho-hat and sigma-eps-hat
summary: $(SUMMARY) $(SUMMARY_MD)
$(SUMMARY): $(CELLS) $(SRC)
	$(PY) -m sim summary --output-dir $(OUT)
$(SUMMARY_MD): $(SUMMARY) ;

## figures: Figures 1 and 2
figures: $(FIG1) $(FIG2)
$(FIG1): $(SUMMARY) $(SRC)
	$(PY) -m sim figures --output-dir $(OUT)
$(FIG2): $(FIG1) ;

## tables: LaTeX tables (PDF build on hold until a TeX engine is pinned)
tables: $(TABLE1) $(TABLE2) $(TABLE3)
$(TABLE1): $(SUMMARY) $(SRC)
	$(PY) -m sim tables --output-dir $(OUT)
$(TABLE2) $(TABLE3): $(TABLE1) ;

## report: one-page HTML summary (open output/report.html, or publish the fragment)
report: $(REPORT) $(FRAGMENT)
$(REPORT): $(SUMMARY) $(SRC)
	$(PY) -m sim report --output-dir $(OUT)
$(FRAGMENT): $(REPORT) ;

## proof: machine-check the growth autocovariance matrix (Lean 4 + Mathlib)
# elan (pinned in mise.toml) provides `lake`.  When elan comes from mise its
# proxies are not always on PATH, so fall back to `elan run` with the toolchain
# pinned in proofs/lean-toolchain.  See proofs/README.md.
proof:
	@cd $(LEAN) && \
	  if command -v lake >/dev/null 2>&1; then LAKE="lake"; \
	  else LAKE="elan run --install $$(cat lean-toolchain) lake"; fi; \
	  $$LAKE exe cache get && $$LAKE build

## proof-numeric: cross-check the proved matrix against simulated panels
proof-numeric: $(STAMP)/setup
	$(PY) $(LEAN)/numeric_check.py

## check: outputs match the current spec and commit, and nothing is stale
check: SPEC.md $(CELLS) $(SUMMARY) $(FIG1) $(FIG2)
	$(PY) -m sim check --output-dir $(OUT)
	@$(MAKE) --no-print-directory -q all \
	  || { echo "[check] FAIL  make -q all reports stale targets"; exit 1; }
	@echo "[check] staleness PASS"

## clean: remove the output folder
clean:
	rm -rf $(OUT)

## distclean: also drop the build stamps and the virtualenv
distclean: clean
	rm -rf $(STAMP) .venv .pytest_cache

## help: list the targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## /  /'
