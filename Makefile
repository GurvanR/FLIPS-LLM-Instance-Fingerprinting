# FLIPS Makefile: one target per reproduction tier (docs/reproduction.md).
#
#   tier 0  test, smoke, smoke-open-set, gate-a, gate-golden   CPU, in-repo fixture, about 2 min
#   tier 1  fetch, paper                                       CPU, results archive (4.4 MB), about 20 s
#   tier 2  fetch-generations, grid, recompute-closed          CPU, release data archive (6.4 GB), CPU-hours
#   tier 3  recompute-open                                     CPU, same input, hundreds of core-hours
#   tier 4  generate                                           GPU, vLLM, Hugging Face weights
#
# All paths are relative to the repository root. Override on the command line, e.g.
#   make paper FLIPS_DATA=/path/to/unpacked/archive PYTHON=.venv/bin/python
#
# The targets need `audit_llm` importable: run them inside the project's environment
# (`poetry run make ...`, or `poetry shell`), or pass PYTHON=<interpreter that has it>.
# `make` with no target prints `help`.

PYTHON     ?= python
FLIPS_DATA ?= data
FLIPS_OUT  ?= paper/out
OUT        ?= out

# Tier 2 / 3 inputs: the answers of the release data archive (make fetch-generations), or of your own run.
ANSWERS ?= Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers
GRID    ?= $(OUT)/grid
JOBS    ?= $(shell nproc 2>/dev/null || echo 4)

# Tier 4 inputs.
MODEL        ?=
QUANTIZATION ?= no_quantized
SUB_RUN      ?= default
GPUS         ?= 1
DEVICE       ?=
DATASET      ?= FLiPS_ICML

FIXTURE := tests/fixtures/answers.parquet
PIPELINE := $(PYTHON) -m audit_llm.pipeline

.DEFAULT_GOAL := help
.PHONY: help fetch fetch-generations fetch-check paper paper-list test smoke smoke-open-set gate-a gate-golden \
        grid recompute-closed recompute-open generate clean

help:
	@echo "FLIPS targets (PYTHON=$(PYTHON))"
	@echo ""
	@echo "  Tier 0 - in-repo fixture, CPU:"
	@echo "    test             run the test suite (unit tests and regression tests on the fixture)"
	@echo "    smoke            closed-set pipeline on the fixture (XP_configs/smoke.yaml)"
	@echo "    smoke-open-set   open-set pipeline on the fixture (XP_configs/smoke_open_set.yaml)"
	@echo "    gate-a           smoke run plus sanity checks on its accuracies"
	@echo "    gate-golden      smoke run compared with the stored reference summary"
	@echo ""
	@echo "  Tier 1 - analysis from the results archive, CPU:"
	@echo "    fetch            download and verify the results archive into data/results/"
	@echo "    fetch-check      verify the files on disk against data/manifest.sha256 (no network)"
	@echo "    paper            regenerate the computed figures, tables and numbers, then compare with paper/expected/"
	@echo "    paper-list       print the build steps of 'paper'"
	@echo ""
	@echo "  Tiers 2-3 - recompute from the generations (ANSWERS=$(ANSWERS)):"
	@echo "    fetch-generations download and verify the release data archive (generations, LLMmap run) into Productions/"
	@echo "    grid             compute the feature grid           -> GRID=$(GRID)"
	@echo "    recompute-closed closed-set protocol of the paper   -> $(OUT)/closed_set"
	@echo "    recompute-open   open-set protocol of the paper     -> $(OUT)/open_set"
	@echo ""
	@echo "  Tier 4 - generation (GPU, vLLM):"
	@echo "    generate         one instance family: make generate MODEL=<hf id> [QUANTIZATION=fp8|bitsandbytes_int4] SUB_RUN=<name> [GPUS=4] [DEVICE=<GPU ids, default 0..GPUS-1>]"
	@echo ""
	@echo "    clean            remove out/, paper/out/ and xp_logs/"

# --------------------------------------------------------------------------- #
# Tier 0                                                                       #
# --------------------------------------------------------------------------- #
test:
	$(PYTHON) -m pytest tests -q

smoke:
	$(PIPELINE) run --config XP_configs/smoke.yaml --answers $(FIXTURE) --out $(OUT)/smoke --n-jobs $(JOBS)

smoke-open-set:
	$(PIPELINE) run --config XP_configs/smoke_open_set.yaml --answers $(FIXTURE) --out $(OUT)/smoke_open_set --n-jobs $(JOBS)

gate-a:
	$(PYTHON) scripts/gates/gate_a.py

gate-golden:
	$(PYTHON) scripts/gates/gate_golden.py

# --------------------------------------------------------------------------- #
# Tier 1                                                                       #
# --------------------------------------------------------------------------- #
fetch:
	$(PYTHON) scripts/fetch_data.py

fetch-check:
	$(PYTHON) scripts/fetch_data.py --check-only

paper:
	FLIPS_DATA=$(FLIPS_DATA) FLIPS_OUT=$(FLIPS_OUT) $(PYTHON) paper/make_paper.py --check

paper-list:
	$(PYTHON) paper/make_paper.py --list

# --------------------------------------------------------------------------- #
# Tiers 2-3: recompute. Input: the generations of the release data archive (docs/reproduction/recompute.md). #
# --------------------------------------------------------------------------- #
fetch-generations:
	$(PYTHON) scripts/fetch_data.py --with-generations

define require_answers
@if [ ! -d "$(ANSWERS)" ]; then \
	echo "No answers at ANSWERS=$(ANSWERS): run 'make fetch-generations' first, or set ANSWERS=<dir of answers parquet files> (your own run)."; \
	echo "See docs/reproduction/recompute.md."; \
	exit 2; \
fi
endef

define require_grid
@if [ ! -d "$(GRID)" ]; then \
	echo "No feature grid at GRID=$(GRID): run 'make grid' first, or set GRID=<grid dir>."; \
	exit 2; \
fi
endef

grid:
	$(require_answers)
	$(PIPELINE) build-grid --answers $(ANSWERS) --out $(GRID) --n-jobs $(JOBS)

recompute-closed:
	$(require_grid)
	$(PIPELINE) run --config XP_configs/closed_set.yaml --grid $(GRID) --out $(OUT)/closed_set

recompute-open:
	$(require_grid)
	$(PIPELINE) run --config XP_configs/open_set.yaml --grid $(GRID) --out $(OUT)/open_set

# --------------------------------------------------------------------------- #
# Tier 4: generation. Needs a GPU, vLLM and the model weights (docs/reproduction/generation.md). #
# --------------------------------------------------------------------------- #
generate:
	@if [ -z "$(MODEL)" ]; then \
		echo "MODEL is not set: make generate MODEL=<Hugging Face id> [QUANTIZATION=fp8|bitsandbytes_int4] SUB_RUN=<name>"; \
		exit 2; \
	fi
	@$(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('vllm') else 1)" \
		|| { echo "generate needs torch and vLLM (poetry install --extras generation, then vLLM: docs/reproduction/generation.md)."; exit 2; }
	$(PYTHON) scripts/Run_Inferences.py --dataset $(DATASET) --model $(MODEL) --quantization $(QUANTIZATION) \
		--sub_run $(SUB_RUN) --gpu $(GPUS) $(if $(DEVICE),--device $(DEVICE)) --parse_gen

# --------------------------------------------------------------------------- #
clean:
	rm -rf out paper/out xp_logs
