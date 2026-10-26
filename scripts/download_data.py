"""Download the LGG MRI segmentation dataset (TCGA-LGG, Buda et al. 2019) into Dataset/.

Default source is a public Hugging Face mirror (no account needed, CC BY-NC-SA 4.0: research use,
do not redistribute commercially). To use the original Kaggle release instead:

    kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p Dataset --unzip
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Ehsan-rmz/lgg-mri-segmentation-research"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Dataset/lgg-mri-segmentation")
    args = ap.parse_args()
    path = snapshot_download(REPO, repo_type="dataset", local_dir=args.out, max_workers=8)
    n = len(list(Path(path).rglob("*_mask.tif")))
    print(f"Done: {n} image/mask pairs in {path}")


if __name__ == "__main__":
    main()
