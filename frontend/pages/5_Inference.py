"""Inference page — run predictions on new images."""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.state import init_state, get
from utils.visualization import draw_detection_boxes, draw_segmentation_mask

init_state()

st.header("Inference")

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── Check prerequisites ─────────────────────────────────────────────────────
decoder = get("decoder")
if decoder is None:
    st.warning("Please build and train a model first.")
    st.stop()

task = get("task")
stats = get("dataset_stats")
class_names = stats.get("class_names", []) if stats else []

# ── Image upload ─────────────────────────────────────────────────────────────
st.subheader("Upload Image")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    # Save to temp file for run_inference (expects file paths)
    tmp = Path(tempfile.mktemp(suffix=".jpg", prefix="fft_infer_"))
    image.save(tmp)

    if st.button("Run Inference", type="primary"):
        with st.spinner("Running inference..."):
            try:
                from core.evaluation.inference import run_inference

                results = run_inference(decoder, [str(tmp)], device="cpu")
                result = results[0]

                st.subheader("Prediction")

                if task == "classification":
                    pred_class = result["predicted_class"]
                    confidence = result["confidence"]
                    probs = result["probabilities"]

                    name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
                    st.metric("Predicted Class", name)
                    st.metric("Confidence", f"{confidence:.2%}")

                    # Probability bar chart
                    if class_names and len(probs) == len(class_names):
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=probs, y=class_names, orientation="h",
                            marker_color=["#636EFA" if i == pred_class else "#C0C0C0"
                                          for i in range(len(class_names))],
                        ))
                        fig.update_layout(title="Class Probabilities", xaxis_title="Probability",
                                          height=max(200, len(class_names) * 30),
                                          template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

                elif task == "detection":
                    boxes = result["boxes"]
                    labels = result["labels"]
                    scores = result["scores"]

                    st.write(f"Detected **{len(boxes)}** objects")

                    if boxes:
                        annotated = draw_detection_boxes(image, boxes, scores, labels,
                                                         class_names=class_names or None)
                        st.image(annotated, caption="Detections", use_container_width=True)

                        # Details table
                        rows = []
                        for b, s, l in zip(boxes, scores, labels):
                            name = class_names[l] if l < len(class_names) else str(l)
                            rows.append({"Class": name, "Score": f"{s:.3f}",
                                         "Box (cxcywh)": f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"})
                        st.dataframe(rows, use_container_width=True)

                elif task == "segmentation":
                    import numpy as np

                    mask = result["mask"]
                    if hasattr(mask, "numpy"):
                        mask = mask.numpy()
                    mask = mask.astype(np.uint8)

                    annotated = draw_segmentation_mask(image, mask)
                    st.image(annotated, caption="Segmentation mask", use_container_width=True)

                    # Class area stats
                    class_areas = result.get("class_areas", {})
                    if class_areas:
                        st.write("**Class areas (pixels):**")
                        for cls_id, area in sorted(class_areas.items()):
                            name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                            st.write(f"- {name}: {area:,}")

            except Exception as e:
                st.error(f"Inference failed: {e}")

    # Clean up temp file
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
