.PHONY: sync cuda-sync lock format lint typecheck test scripts check cpu-check cuda-check notebook lab

UV_RUN ?= uv run --locked
PLOT_CACHE ?= /tmp/reasoned-icrl-plot-cache
PLOT_ENV = MPLCONFIGDIR="$(PLOT_CACHE)/matplotlib" XDG_CACHE_HOME="$(PLOT_CACHE)"

# The xland extra is part of the verified environment (decision 15): the
# XLand-MiniGrid simulator runs on the CPU beside the learner.
sync:
	uv sync --frozen --extra xland

cuda-sync:
	uv sync --frozen --extra cuda --extra xland

cpu-check: sync
	ACCELERATE_USE_CPU=true $(MAKE) check

cuda-check: cuda-sync
	ACCELERATE_USE_CPU=false $(UV_RUN) --extra cuda pytest -m cuda --require-cuda

lock:
	uv lock

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

lint:
	$(UV_RUN) ruff format --check .
	$(UV_RUN) ruff check .

typecheck:
	$(UV_RUN) mypy --strict reasoned_icrl scripts

test:
	$(UV_RUN) pytest -m "not slow"

scripts:
	$(UV_RUN) python scripts/train.py --help
	$(UV_RUN) python scripts/evaluate.py --help

check: lint typecheck test scripts

# Execute every figure and table notebook from a fresh kernel over the saved
# records (REASONED_ICRL_OUTPUTS, default outputs/); outputs go to notebooks/outputs/.
notebook:
	$(PLOT_ENV) $(UV_RUN) python notebooks/run_figures.py

# Open the notebooks interactively.
lab:
	$(PLOT_ENV) $(UV_RUN) jupyter lab notebooks/
