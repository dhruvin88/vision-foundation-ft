"""Dataset page — upload or point to a local dataset."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.state import init_state, get, set_

init_state()

st.header("Dataset Configuration")

task = get("task")

# ── Dataset source ───────────────────────────────────────────────────────────
source = st.radio("Dataset source", ["Local folder", "Upload ZIP"], horizontal=True)

dataset_path = None

if source == "Local folder":
    folder = st.text_input(
        "Path to dataset folder",
        value=str(get("dataset_path") or ""),
        placeholder="e.g. C:/data/my_dataset",
    )
    if folder and Path(folder).is_dir():
        dataset_path = folder
    elif folder:
        st.warning("Directory not found.")

else:
    uploaded = st.file_uploader("Upload dataset ZIP", type=["zip"])
    if uploaded is not None:
        extract_dir = Path(tempfile.mkdtemp(prefix="fft_dataset_"))
        with zipfile.ZipFile(uploaded) as zf:
            zf.extractall(extract_dir)
        # If ZIP contains a single top-level folder, use that
        children = list(extract_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            dataset_path = str(children[0])
        else:
            dataset_path = str(extract_dir)
        st.success(f"Extracted to `{dataset_path}`")

# ── Format help ──────────────────────────────────────────────────────────────
with st.expander("Expected folder structure"):
    if task == "classification":
        st.code("root/\n  class_a/\n    img1.jpg\n    img2.jpg\n  class_b/\n    img3.jpg")
    elif task == "detection":
        st.code("root/\n  images/\n    img1.jpg\n    img2.jpg\n  annotations.json   (COCO format)")
    else:
        st.code("root/\n  images/\n    img1.jpg\n  masks/\n    img1.png   (class index per pixel)")

# ── Load and preview ─────────────────────────────────────────────────────────
if dataset_path and st.button("Load Dataset"):
    with st.spinner("Loading dataset..."):
        try:
            import sys, os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from sdk import Dataset, Encoder

            encoder = get("encoder")
            transform = None
            if encoder is not None:
                transform = encoder.get_transform()
            else:
                # Create a temporary encoder just for the transform
                enc = Encoder(get("encoder_name"))
                transform = enc.get_transform()

            ds = Dataset.from_folder(dataset_path, task=task, transform=transform)
            stats = ds.get_stats()

            set_("dataset_path", dataset_path)
            set_("dataset", ds)
            set_("dataset_stats", stats)

            st.success(f"Loaded {stats['num_samples']} samples across {stats['num_classes']} classes.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

# ── Show stats & previews ───────────────────────────────────────────────────
stats = get("dataset_stats")
if stats:
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total samples", stats["num_samples"])
        st.metric("Number of classes", stats["num_classes"])
    with col2:
        st.write("**Class names:**", ", ".join(stats.get("class_names", [])))
        if "class_distribution" in stats:
            st.bar_chart(stats["class_distribution"])

    # Preview sample images
    ds = get("dataset")
    if ds is not None and len(ds) > 0:
        st.subheader("Sample Images")
        n_preview = min(8, len(ds))
        cols = st.columns(min(4, n_preview))
        for i in range(n_preview):
            sample = ds[i]
            img = Image.open(sample["image_path"])
            with cols[i % len(cols)]:
                st.image(img, caption=Path(sample["image_path"]).name, use_container_width=True)
