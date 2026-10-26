# Brain MRI tumour segmentation with U-Net

Segments lower-grade glioma on brain MRI slices (TCGA-LGG dataset). TensorFlow/Keras, with training and
evaluation scripts, an inference API and a small Streamlit app.

On a patient-level held-out test set (16 patients, 549 slices) the default model reaches Dice 0.87 over all
slices and 0.70 on slices that contain a tumour. Inference takes about 10 ms per slice on a GTX 1650 Ti.
Details and caveats: [docs/results.md](docs/results.md).

## Quick start

```bash
conda env create -f environment.yml
conda activate brain-mri-unet
make env-hook            # lets TensorFlow find the CUDA libs; skip on CPU

make data                # download the dataset
make train               # trains the default config into runs/<timestamp>/
make app                 # Streamlit demo
```

Predict from Python:

```python
from brainseg.predict import Segmenter, read_image

seg = Segmenter.from_path("runs/<run>/best.keras")
res = seg.predict(read_image("slice.tif"))
res.mask, res.probability, res.has_tumor
```

## Docs

- [Usage](docs/usage.md): setup, data, training, evaluation, prediction, the app, tests
- [Results and performance](docs/results.md): metrics, test-set numbers, speed and memory
- [Architecture and design notes](docs/architecture.md): code layout, model, what was fixed from the original notebooks, test strategy
- [Handoff](HANDOFF.md): current status and next steps

## Layout

```
src/brainseg/   data, model, losses, metrics, train, evaluate, predict
app/            Streamlit app
configs/        default.yaml (4 GB GPU), large.yaml, smoke.yaml
tests/          unit, end-to-end, UI, performance
notebooks/      walkthrough and the archived original experiment
```

## Notes

- Splits are by patient, so no scan appears in both train and test.
- The 0.91 Dice in the original notebook is not comparable; its protocol leaked data between splits.
- Research use only. Not a medical device.
- Dataset: Buda, Saha, Mazurowski, *Computers in Biology and Medicine*, 2019. The downloaded mirror is CC BY-NC-SA 4.0.
  Data and trained weights are git-ignored. No code licence has been chosen yet.
