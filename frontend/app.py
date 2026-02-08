"""FFT Platform — Streamlit Frontend.

Main entry point. Run with:
    streamlit run frontend/app.py
"""

import streamlit as st

from utils.state import init_state, get

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="FFT Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FFT Platform")
    st.caption("Foundation Model Fine-Tuning")

    st.divider()

    task = st.selectbox(
        "Task",
        ["classification", "detection", "segmentation"],
        index=["classification", "detection", "segmentation"].index(get("task")),
        key="task",
    )

    st.divider()
    st.markdown("### Pipeline")
    st.page_link("pages/1_Dataset.py", label="1. Dataset", icon="📁")
    st.page_link("pages/2_Model.py", label="2. Model", icon="🧠")
    st.page_link("pages/3_Training.py", label="3. Training", icon="⚡")
    st.page_link("pages/4_Results.py", label="4. Results", icon="📊")
    st.page_link("pages/5_Inference.py", label="5. Inference", icon="🔍")

# ── Home page ────────────────────────────────────────────────────────────────
st.title("Foundation Model Fine-Tuning Platform")

st.markdown("""
Fine-tune DINOv2 vision foundation models for **classification**, **detection**,
and **segmentation** tasks — all from your browser.

**Pipeline:**
1. **Dataset** — Upload or point to a local dataset
2. **Model** — Choose encoder + decoder configuration
3. **Training** — Train with live progress monitoring
4. **Results** — View loss curves and metrics
5. **Inference** — Run predictions on new images
""")

# Config summary
st.subheader("Current Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Task", get("task"))

with col2:
    st.metric("Encoder", get("encoder_name"))

with col3:
    status = get("training_status")
    st.metric("Training", status.capitalize())

if get("dataset_stats"):
    stats = get("dataset_stats")
    st.subheader("Dataset")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.metric("Samples", stats.get("num_samples", "?"))
    with dcol2:
        st.metric("Classes", stats.get("num_classes", "?"))
    with dcol3:
        st.metric("Path", str(get("dataset_path") or "—"))
