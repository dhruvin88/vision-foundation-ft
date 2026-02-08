"""Results page — view training metrics and export model."""

from pathlib import Path

import streamlit as st

from utils.state import init_state, get, set_
from utils.visualization import plot_loss_curve, format_metrics_table

init_state()

st.header("Training Results")

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── Check for results ────────────────────────────────────────────────────────
history = get("training_history")
status = get("training_status")

if not history:
    st.info("No training results yet. Run training first on the Training page.")
    st.stop()

# ── Loss curves ──────────────────────────────────────────────────────────────
st.subheader("Loss Curves")
st.plotly_chart(plot_loss_curve(history), use_container_width=True)

# ── Metrics table ────────────────────────────────────────────────────────────
st.subheader("Epoch Metrics")
st.dataframe(format_metrics_table(history), use_container_width=True)

# ── Summary metrics ──────────────────────────────────────────────────────────
st.subheader("Summary")
task = get("task")

if history:
    last = history[-1]
    best_val = min((h.get("val_loss", float("inf")) for h in history), default=None)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Train Loss", f"{last.get('train_loss', 0):.4f}")
    with col2:
        if best_val is not None and best_val < float("inf"):
            st.metric("Best Val Loss", f"{best_val:.4f}")
    with col3:
        st.metric("Epochs Trained", len(history))

    if task == "classification":
        best_acc = max((h.get("val_acc", 0) for h in history), default=None)
        if best_acc is not None:
            st.metric("Best Val Accuracy", f"{best_acc:.4f}")

# ── Training result info ─────────────────────────────────────────────────────
result = get("training_result")
if result:
    with st.expander("Training result details"):
        st.json(result)

# ── Export model ─────────────────────────────────────────────────────────────
st.subheader("Export Model")

decoder = get("decoder")
if decoder is None:
    st.warning("No trained model available.")
else:
    export_path = st.text_input("Save path", value="./trained_model.pt")
    if st.button("Save Decoder Weights", type="primary"):
        try:
            from core.export.weights import save_decoder_weights

            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_decoder_weights(decoder, path)
            st.success(f"Weights saved to `{path.resolve()}`")
        except Exception as e:
            st.error(f"Failed to save: {e}")
