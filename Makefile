# Activate the environment first:  conda activate brain-mri-unet
PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: help env env-hook data demo-model train eval benchmark app test test-fast lint format coverage clean

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

env:            ## create the conda env (then: conda activate brain-mri-unet && make env-hook)
	conda env create -f environment.yml

env-hook:       ## let TensorFlow find the pip-installed CUDA libs inside the active env
	mkdir -p $$CONDA_PREFIX/etc/conda/activate.d
	cp scripts/conda_cuda_hook.sh $$CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh

data:           ## download the LGG MRI dataset into Dataset/ (public HF mirror)
	$(PY) scripts/download_data.py

demo-model:     ## ~1 min: train a small model on synthetic phantoms so the app works instantly
	$(PY) scripts/make_demo_model.py

train:          ## train on the real dataset (override with CONFIG=configs/smoke.yaml)
	$(PY) -m brainseg.train --config $(CONFIG)

eval:           ## evaluate a model: make eval MODEL=runs/<run>/best.keras
	$(PY) -m brainseg.evaluate --model $(MODEL) --config $(CONFIG)

benchmark:      ## inference / training speed for 16/32/64 base filters
	$(PY) -m brainseg.benchmark

app:            ## launch the Streamlit demo
	streamlit run app/streamlit_app.py

test:           ## full suite (unit, e2e training, UI, perf)
	$(PY) -m pytest

test-fast:      ## skip the slow end-to-end and perf tests
	$(PY) -m pytest -m "not slow and not perf"

coverage:       ## suite with coverage report
	$(PY) -m pytest --cov=brainseg --cov-report=term-missing

lint:           ## static checks
	ruff check .

format:         ## auto-fix lint + import order
	ruff check --fix .

clean:          ## remove caches and run artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov runs build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
