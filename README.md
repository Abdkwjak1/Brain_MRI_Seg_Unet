# Brain MRI Tumour Segmentation (U-Net)

U-Net segmentation of lower-grade glioma on brain MRI slices (TCGA-LGG dataset), in TensorFlow/Keras.
Includes training and evaluation CLIs, an inference API and a Streamlit demo.

## Setup

```bash
conda env create -f environment.yml
conda activate brain-mri-unet
make env-hook        # lets TensorFlow find the pip-installed CUDA libs (skip on CPU)
```

## Usage

```bash
make data            # download the LGG dataset into Dataset/
make train           # configs/default.yaml -> runs/<timestamp>/
make eval MODEL=runs/<run>/best.keras
make app             # Streamlit demo
make demo-model      # tiny model trained on synthetic phantoms, so the app works without data
```

Segment images from the command line:

```bash
python -m brainseg.predict slice.tif --model runs/<run>/best.keras --out predictions/
```

or from Python:

```python
from brainseg.predict import Segmenter, read_image

seg = Segmenter.from_path("runs/<run>/best.keras")
res = seg.predict(read_image("slice.tif"))   # any size, grayscale or RGB
res.mask, res.probability, res.has_tumor, res.tumor_fraction
```

`make help` lists all targets.

## Layout

```
src/brainseg/   config, data, model, losses, metrics, train, evaluate, predict, viz, benchmark
app/            Streamlit app
configs/        default.yaml (4 GB GPU), large.yaml, smoke.yaml (tests)
tests/          unit, end-to-end, UI, performance, real-data
notebooks/      walkthrough + the archived original experiment
docs/           design notes, benchmark data
```

## Method

- Data is split by patient (not by slice), so train/val/test never share a scan.
- Loss is BCE + batch-level Dice. Augmentation is applied identically to image and mask.
- Reported metrics: Dice/IoU over all slices, Dice on tumour slices only, detection recall, and the
  false-positive rate on tumour-free slices. About 65% of slices have no tumour, so overall Dice alone is misleading.

Details and the list of fixes over the original notebooks: [docs/architecture.md](docs/architecture.md).

## Performance

GTX 1650 Ti (4 GB), 256x256 input, fp32, `make benchmark`:

| Base filters | Params | GPU latency (batch 1) | GPU inference (batch 8) | GPU training (batch 4) |
|---|---|---|---|---|
| 16 (default) | 1.95 M | 9.6 ms | 143 slices/s | 53 slices/s |
| 32 | 7.77 M | 22.2 ms | 61 slices/s | 21 slices/s |
| 64 | 31.05 M | 69.0 ms | 21 slices/s | out of memory |

Raw data: [docs/assets/](docs/assets/).

## Results

16 filters (1.95 M params), 30 epochs, patient-level split, evaluated once on the held-out test set
(16 patients, 549 slices). The two runs differ only in the learning-rate schedule.

| Metric | Constant LR | Cosine LR (default) |
|---|---|---|
| Dice, all slices | 0.815 | 0.869 |
| Dice, tumour slices only | 0.703 | 0.701 |
| IoU, tumour slices only | 0.609 | 0.605 |
| Tumour detection recall | 0.923 | 0.906 |
| False-positive rate on tumour-free slices | 0.130 | 0.049 |
| Dice, per-patient mean | 0.818 | 0.867 |

The cosine schedule mainly cuts false positives on tumour-free slices; segmentation quality on slices that
contain a tumour is unchanged at about 0.70 Dice, which is the main thing left to improve
(see [HANDOFF.md](HANDOFF.md)). The Dice 0.91 reported in the original notebook used random slice-level
splits and the test set for model selection, so it is not comparable.

## Tests

```bash
make test-fast       # unit, data, inference, UI
make test            # adds end-to-end training and performance budgets
```

## Data and licence

Dataset: Buda, Saha, Mazurowski, *Computers in Biology and Medicine*, 2019 (Kaggle `mateuszbuda/lgg-mri-segmentation`).
`make data` uses a Hugging Face mirror under CC BY-NC-SA 4.0 (non-commercial). Data and weights are git-ignored.
U-Net: Ronneberger et al., MICCAI 2015. No code licence has been chosen yet.

Research use only; not a medical device.
