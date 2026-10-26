# Results and performance

## Setup

- Data: TCGA-LGG, 3929 slices from 110 patients. 35% of slices contain tumour; the median tumour covers about
  2.3% of the slice.
- Split by patient, seed 42: 78 / 16 / 16 patients (2753 / 627 / 549 slices) for train / validation / test.
- Model: U-Net, 16 base filters, depth 4, 1.95 M parameters. Loss: BCE + batch-level Dice. Adam, 30 epochs,
  batch 8, 256x256 input, flip/rotation/shift/zoom augmentation.
- The test split was used once, after training, on the best-validation checkpoint.

## Metrics

| Name | Meaning |
|---|---|
| Dice / IoU, all slices | mean over every slice; an empty slice predicted empty counts as 1 |
| Dice / IoU, tumour slices | mean over slices that really contain tumour; the number to watch |
| Detection recall | share of tumour slices where anything was predicted |
| False-positive rate | share of tumour-free slices where tumour was predicted |
| Dice, per-patient mean | per-patient average, so patients with many slices do not dominate |

Because about 65% of slices are empty, "all slices" Dice is flattered by easy empty slices.

## Test-set results

Two runs that differ only in the learning-rate schedule.

| Metric | Constant LR | Cosine LR (default) |
|---|---|---|
| Dice, all slices | 0.815 | 0.869 |
| IoU, all slices | 0.784 | 0.837 |
| Dice, tumour slices | 0.703 | 0.701 |
| IoU, tumour slices | 0.609 | 0.605 |
| Detection recall | 0.923 | 0.906 |
| False-positive rate | 0.130 | 0.049 |
| Dice, per-patient mean | 0.818 | 0.867 |
| Training time | 40 min | 40 min |

Single seed per run, so differences of about 0.02 are within noise. The schedule mostly reduces false
positives; Dice on tumour slices did not improve and is the main open problem.

The original notebook reported Dice about 0.91. It used random slice-level splits (slices of one patient in
both train and test), selected the model on the test set, and trained on un-normalised inputs, so that number
is not comparable. See [architecture.md](architecture.md#what-changed-versus-the-original-notebooks-and-why).

## Things that went wrong on the way

- A per-slice Dice loss made the model predict "no tumour" everywhere (validation Dice stuck at 0.63, the
  all-background baseline). The loss is now batch-level Dice; a test guards against it.
- With a constant learning rate, validation Dice swung between 0.37 and 0.57 from epoch to epoch.

## Speed and memory

Measured with `make benchmark` on a GTX 1650 Ti (4 GB), 12-core CPU, TensorFlow 2.21, 256x256 input, fp32.

| Base filters | Params | GPU latency, batch 1 | GPU inference, batch 8 | GPU training, batch 4 | CPU latency, batch 1 |
|---|---|---|---|---|---|
| 16 | 1.95 M | 9.6 ms | 143 slices/s | 53 slices/s | 29 ms |
| 32 | 7.77 M | 22.2 ms | 61 slices/s | 21 slices/s | 70 ms |
| 64 | 31.05 M | 69.0 ms | 21 slices/s | out of memory | not run |

- A 150-slice scan takes roughly one second at 16 filters on this GPU.
- 32 filters at batch 8 is borderline on 4 GB; use batch 4.
- Mixed precision was about 2x slower than fp32 here (no tensor cores on this GPU), so it is off by default.

Raw numbers: [assets/benchmark_gpu.json](assets/benchmark_gpu.json),
[assets/benchmark_cpu.json](assets/benchmark_cpu.json).
