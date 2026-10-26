"""Interactive demo: upload an MRI slice (or generate a phantom) and segment the tumour.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from brainseg.data import find_pairs, load_image, load_mask
from brainseg.metrics import slice_metrics
from brainseg.predict import Segmenter, to_rgb_uint8
from brainseg.synthetic import make_phantom
from brainseg.viz import overlay

ROOT = Path(__file__).resolve().parent.parent
DISCLAIMER = (
    "Research demo only. Not a medical device and not validated for clinical use; "
    "do not use it to make diagnostic or treatment decisions."
)

st.set_page_config(page_title="Brain MRI Tumour Segmentation", page_icon="🧠", layout="wide")


def discover_models() -> list[str]:
    roots = [
        Path(p)
        for p in os.environ.get("BRAINSEG_MODEL_DIRS", f"{ROOT / 'runs'}:{ROOT / 'models'}").split(":")
    ]
    found = [str(p) for r in roots if r.exists() for p in sorted(r.rglob("*.keras"))]
    return sorted(found, key=lambda p: Path(p).stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner="Loading model…")
def load_segmenter(path: str) -> Segmenter:
    return Segmenter.from_path(path)


@st.cache_data(show_spinner=False)
def dataset_index(root: str):
    try:
        return find_pairs(root)
    except FileNotFoundError:
        return None


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def heatmap(image: np.ndarray, prob: np.ndarray) -> np.ndarray:
    colored = cv2.cvtColor(
        cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    return cv2.addWeighted(image, 0.55, colored, 0.45, 0)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧠 BrainSeg")
    st.caption("U-Net tumour segmentation on LGG FLAIR MRI")

    models = discover_models()
    custom = st.text_input(
        "Model path (.keras)", value="", placeholder="runs/exp1/best.keras", key="model_path"
    )
    options = ([custom] if custom else []) + models
    model_path = st.selectbox("Trained model", options, key="model_select") if options else None
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="threshold")
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05, key="alpha")

    st.divider()
    ds_root = ROOT / "Dataset" / "lgg-mri-segmentation"
    pairs = dataset_index(str(ds_root))
    sources = ["Synthetic phantom", "Upload image"] + (["Dataset sample"] if pairs is not None else [])
    source = st.radio("Input", sources, key="source")
    st.warning(DISCLAIMER)

st.title("Brain MRI tumour segmentation")
st.write("Pick an input slice and the model highlights the predicted tumour region.")

# ------------------------------------------------------------------ input
image: np.ndarray | None = None
truth: np.ndarray | None = None

if source == "Synthetic phantom":
    seed = st.number_input("Phantom seed", 0, 10_000, 7, key="seed")
    image, truth = make_phantom(256, np.random.default_rng(int(seed)), with_tumor=True)
    truth = (truth > 127).astype(np.uint8)
elif source == "Upload image":
    up = st.file_uploader("MRI slice", type=["png", "jpg", "jpeg", "tif", "tiff"], key="upload")
    if up is not None:
        try:
            image = to_rgb_uint8(np.array(Image.open(up)))
        except Exception as exc:  # corrupt / unsupported upload
            st.error(f"Could not read that image: {exc}")
else:
    idx = st.slider("Slice index", 0, len(pairs) - 1, 0, key="slice_idx")
    row = pairs.iloc[idx]
    image = load_image(row["image"], 256)
    truth = load_mask(row["mask"], 256)[..., 0]
    st.caption(f"Patient {row['patient']}")

# ------------------------------------------------------------------ inference
if model_path is None:
    st.info(
        "No trained model found. Train one with `make train` (real data) or `make demo-model` "
        "(synthetic phantoms, ~1 min), or type a model path in the sidebar."
    )
elif image is None:
    st.info("Upload an image to begin.")
else:
    try:
        seg = load_segmenter(model_path)
    except Exception as exc:
        st.error(f"Could not load model `{model_path}`: {exc}")
        st.stop()

    res = seg.predict(image, threshold)
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Input", use_container_width=True)
    c2.image(heatmap(image, res.probability), caption="Tumour probability", use_container_width=True)
    pred_overlay = overlay(image, res.mask, alpha=alpha)
    c3.image(pred_overlay, caption="Predicted segmentation", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Tumour detected", "Yes" if res.has_tumor else "No")
    m2.metric(
        "Tumour area", f"{res.tumor_fraction:.2%} of slice", f"{res.tumor_pixels:,} px", delta_color="off"
    )
    m3.metric("Peak confidence", f"{res.max_probability:.2f}")

    if truth is not None:
        sm = slice_metrics(truth[None, ..., None], res.probability[None, ..., None], threshold).iloc[0]
        g1, g2, g3 = st.columns(3)
        g1.metric("Dice vs ground truth", f"{sm['dice']:.3f}")
        g2.metric("IoU vs ground truth", f"{sm['iou']:.3f}")
        g3.image(
            overlay(image, truth, (40, 220, 40), alpha), caption="Ground truth", use_container_width=True
        )

    d1, d2 = st.columns(2)
    d1.download_button("Download mask (PNG)", png_bytes(res.mask * 255), "mask.png", "image/png")
    d2.download_button("Download overlay (PNG)", png_bytes(pred_overlay), "overlay.png", "image/png")
