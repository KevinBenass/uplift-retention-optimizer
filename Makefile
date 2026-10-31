.PHONY: setup data train simulate api test lint all

UV ?= uv
DATASET ?= x5

setup:      ## Install dependencies
	$(UV) sync --all-extras

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
