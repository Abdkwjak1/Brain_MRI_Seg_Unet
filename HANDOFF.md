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

Two runs exist (below), neither tuned. What I found:

1. **Per-slice Dice loss collapsed** to "predict no tumour" (val Dice stuck at 0.63, the all-background
   baseline). Fixed by switching to batch-level Dice in `losses.py`; regression test added.
2. **Baseline run: batch Dice, constant LR 1e-3, 16 filters, 30 epochs** (`runs/lgg_unet16`, ~40 min; `runs/` is git-ignored).
   Train Dice rose from 0.09 to about 0.69; validation Dice was very noisy (0.37-0.57, best 0.68) because the
   plateau scheduler never fired. Held-out test (16 patients, 549 slices): Dice 0.815 overall, 0.703 on tumour
   slices, detection recall 0.923, false-positive rate on empty slices 0.130, per-patient Dice 0.818
   (full numbers in `runs/lgg_unet16/test_metrics.json`, also in the README).
3. **Cosine LR run** (default config, `runs/lgg_unet16_cosine`, same split and seed): validation Dice was smooth
   (about 0.57 at the end) instead of noisy. Held-out test: Dice 0.869 overall, 0.701 on tumour slices,
   detection recall 0.906, false-positive rate on empty slices 0.049, per-patient Dice 0.867.
   Versus the baseline: overall Dice and false positives improved a lot, tumour-slice Dice did not change.

**Where things stand:** tumour-slice Dice is stuck at about 0.70 with the 1.95 M-parameter model. The literature
figure for patient-disjoint splits on this dataset is roughly 0.82-0.90, so there is headroom. Both runs are
single-seed, so small differences (e.g. 0.703 vs 0.701, recall 0.923 vs 0.906) are within noise.

Next steps, in order:

- Try `configs/large.yaml` (32 filters, 7.8 M params). It needs `batch_size: 4` on this 4 GB GPU and is roughly
  2x slower per epoch, so expect 1.5-2 hours for 40 epochs.
- Cheaper ideas to test too: more epochs, `loss: focal_tversky` (penalises missed tumour more), lower LR,
  oversampling tumour slices.
- Run each variant on at least two seeds before trusting a difference.
- Once a run is clearly better, update the README results table and set it as the default config.
- Optional: take a UI screenshot with the final model if wanted.

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
