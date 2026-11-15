.PHONY: setup libomp data train simulate api test lint all

UV ?= uv
DATASET ?= x5

setup:      ## Install dependencies (and, on a Mac without Homebrew, the OpenMP runtime)
	$(UV) sync --all-extras
	@$(UV) run python -c "import lightgbm" 2>/dev/null || $(MAKE) libomp

# LightGBM's macOS wheel expects libomp from Homebrew (/usr/local/opt/libomp), which Intel
# Macs on recent macOS can't install. scikit-learn's wheel ships the same LLVM runtime, so
# that file is linked where LightGBM looks for it. Asks for your password once.
libomp:
	@lib=$$($(UV) run python -c "import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), '.dylibs', 'libomp.dylib'))"); \
	test -f "$$lib" || { echo "scikit-learn's libomp not found at $$lib"; exit 1; }; \
	echo "LightGBM needs libomp: linking $$lib into /usr/local/opt/libomp/lib (sudo)"; \
	sudo mkdir -p /usr/local/opt/libomp/lib && sudo ln -sf "$$lib" /usr/local/opt/libomp/lib/libomp.dylib
	$(UV) run python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'OK')"

data:       ## Download X5 RetailHero (~1 GB) and build features with DuckDB (~5 min)
	$(UV) run uplift prepare --dataset $(DATASET)

train:      ## Fit all models, evaluate on test, write results/, README block and memo
	$(UV) run uplift train --dataset $(DATASET)

all: data train

simulate:   ## Simulation study with known true uplift (no download, ~2 min)
	$(UV) run uplift prepare --dataset synthetic
	$(UV) run uplift train --dataset synthetic

api:        ## Scoring API on :8000
	UPLIFT_DATASET=$(DATASET) $(UV) run uvicorn uplift_optimizer.api:app --port 8000

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
