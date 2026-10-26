# Usage

All commands assume the `brain-mri-unet` conda env is active. `make help` lists every target.

## Environment

```bash
conda env create -f environment.yml
conda activate brain-mri-unet
make env-hook
```

`make env-hook` installs a conda activate script that puts the pip-installed CUDA libraries on
`LD_LIBRARY_PATH`. Without it TensorFlow falls back to the CPU without an error. Skip it on a CPU-only machine.

## Data

```bash
make data
```

Downloads the TCGA-LGG dataset (3929 slices, 110 patients) to `Dataset/lgg-mri-segmentation/` from a public
Hugging Face mirror (CC BY-NC-SA 4.0). To use the original Kaggle release instead:

```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p Dataset --unzip
```

Any folder works as long as it contains `*.tif` images with matching `*_mask.tif` masks in per-patient folders
(`TCGA_XX_1234_19990101/...`); set `data.root` in the config.

## Train

```bash
make train                                  # configs/default.yaml -> runs/<timestamp>/
make train CONFIG=configs/large.yaml        # 32 filters, needs more GPU memory
python -m brainseg.train --config configs/default.yaml --epochs 5 --out runs/quick
```

Each run directory contains `best.keras`, `config.yaml`, `history.csv`, `training_curves.png`,
`test_metrics.json`, `test_per_slice.csv` and `test_predictions.png`. The best checkpoint is chosen on the
validation split; the test split is only used once at the end.

Config options are in [`src/brainseg/config.py`](../src/brainseg/config.py). The shipped configs:

| File | Filters | Params | Notes |
|---|---|---|---|
| `configs/default.yaml` | 16 | 1.95 M | fits a 4 GB GPU at batch 8 |
| `configs/large.yaml` | 32 | 7.77 M | use `batch_size: 4` on a 4 GB GPU |
| `configs/smoke.yaml` | 8 | small | 64 px images, used by the tests |

## Evaluate

```bash
make eval MODEL=runs/<run>/best.keras
```

Rebuilds the same patient-level split from the config and reports the metrics described in
[results.md](results.md#metrics).

## Predict

Command line:

```bash
python -m brainseg.predict slice.tif --model runs/<run>/best.keras --out predictions/
```

Python:

```python
from brainseg.predict import Segmenter, read_image

seg = Segmenter.from_path("runs/<run>/best.keras")
res = seg.predict(read_image("slice.tif"))   # any size, grayscale or RGB, 8 or 16 bit
res.mask            # uint8, original resolution
res.probability     # float32, original resolution
res.has_tumor, res.tumor_fraction, res.max_probability
```

## Demo app

```bash
make demo-model     # optional: tiny model trained on synthetic phantoms, no dataset needed
make app            # http://localhost:8501
```

The app lists models found under `runs/` and `models/`, or you can type a path. Inputs: synthetic phantom,
uploaded image, or a slice from the dataset. It shows the probability map, the thresholded overlay, tumour
area, and Dice/IoU when ground truth is available.

## Benchmark

```bash
make benchmark
```

Prints latency and throughput for 16/32/64 base filters and writes JSON with `--json`. 

## Tests

```bash
make test-fast      # about 1 minute: unit, data, inference, UI
make test           # adds end-to-end training and performance budgets
make coverage
```

Markers: `slow`, `perf`, `ui`, `realdata` (skipped unless `Dataset/` exists). See
[architecture.md](architecture.md#testing-strategy) for what each layer checks.
