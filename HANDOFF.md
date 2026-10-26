# Handoff

State of the project as of 2026-10-26, and what is left.

## Environment

- Conda env `brain-mri-unet` (Python 3.11, TensorFlow 2.21 / Keras 3.15, GPU works).
- The env has an activate hook that puts the pip-installed CUDA libs on `LD_LIBRARY_PATH`
  (`scripts/conda_cuda_hook.sh`, installed with `make env-hook`). Without it TensorFlow silently uses the CPU.
- Dataset is already downloaded to `Dataset/lgg-mri-segmentation/` (3929 pairs, 110 patients, git-ignored).
- GPU: GTX 1650 Ti, 4 GB. This limits model size and batch size (see below).

## Done

- Cleanup: removed `.pyc`, TensorBoard events, `.vscode`, duplicate notebook; outputs stripped from the archived one.
- New package `src/brainseg/`, configs, CLIs, Streamlit app, Makefile, CI workflow, `environment.yml`.
- 81 tests, 95% coverage (unit, data, e2e training, UI via Streamlit `AppTest`, perf budgets, real-data checks).
- Benchmarks in `docs/assets/benchmark_*.json`; design notes and change list in `docs/architecture.md`.

## Not done: tuned reference training on real data

Only the baseline below exists. What I found:

1. **Per-slice Dice loss collapsed** to "predict no tumour" (val Dice stuck at 0.63, the all-background
   baseline). Fixed by switching to batch-level Dice in `losses.py`; regression test added.
2. **Baseline run: batch Dice, constant LR 1e-3, 16 filters, 30 epochs** (`runs/lgg_unet16`, ~40 min, git-ignored).
   Train Dice rose from 0.09 to about 0.69; validation Dice was very noisy (0.37-0.57, best 0.68) because the
   plateau scheduler never fired. Held-out test (16 patients, 549 slices): Dice 0.815 overall, 0.703 on tumour
   slices, detection recall 0.923, false-positive rate on empty slices 0.130, per-patient Dice 0.818
   (full numbers in `runs/lgg_unet16/test_metrics.json`, also in the README).
3. **Cosine LR schedule** was added afterwards (`train.lr_schedule`, default `cosine`) but has not been run on
   real data yet. Only the unit test for the schedule has been run.

Next steps, in order:

- `make train` with the cosine default and compare with the baseline above.
- If validation is still weak: try `configs/large.yaml` (32 filters; use `batch_size: 4` if it runs out of memory),
  more epochs, or a lower LR.
- Original notebook reported Dice ~0.91 but its protocol was flawed (see `docs/architecture.md`); the literature
  figure for patient-disjoint splits on this dataset is roughly 0.82-0.90, so there is headroom.
- Once a run is good: add the real numbers and a results table to `README.md`, run
  `python -m brainseg.evaluate`, and then take a UI screenshot if wanted.

## Hardware notes (measured)

- 32 filters at batch 8 is borderline on 4 GB; 64 filters does not train at batch 4.
- Mixed precision is about 2x slower than fp32 on this GPU (no tensor cores), so it is off by default.
- XLA (`jit_compile`) is disabled in `compile_model` for predictable memory.

## Things to decide

- No code licence file yet (dataset is CC BY-NC-SA 4.0).
- Model weights are git-ignored; decide whether to publish any (e.g. GitHub release).
- `notebooks/01_walkthrough.ipynb` has never been executed end to end.

## Handy commands

```bash
conda activate brain-mri-unet
make test-fast          # about 1 min
make train CONFIG=configs/default.yaml
make benchmark
```
