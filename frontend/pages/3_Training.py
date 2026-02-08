"""Training page — configure hyperparameters and train with live progress."""

import threading
import time

import streamlit as st

from utils.state import init_state, get, set_
from utils.visualization import plot_loss_curve, format_metrics_table

init_state()

st.header("Training")

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── Prerequisites check ─────────────────────────────────────────────────────
decoder = get("decoder")
dataset = get("dataset")

if decoder is None:
    st.warning("Please build a model first (Model page).")
    st.stop()
if dataset is None:
    st.warning("Please load a dataset first (Dataset page).")
    st.stop()

# ── Hyperparameters ──────────────────────────────────────────────────────────
st.subheader("Hyperparameters")

col1, col2, col3 = st.columns(3)
with col1:
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=20)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=16)
with col2:
    lr = st.number_input("Learning rate", min_value=1e-6, max_value=1.0,
                         value=1e-3, format="%.1e")
    scheduler = st.selectbox("LR scheduler", ["cosine", "step", "constant"])
with col3:
    warmup_epochs = st.number_input("Warmup epochs", min_value=0, max_value=50, value=5)
    patience = st.number_input("Early stopping patience (0=off)", min_value=0, max_value=100, value=10)

num_workers = st.slider("Data loader workers", min_value=0, max_value=8, value=0,
                         help="Set to 0 on Windows to avoid multiprocessing issues")

# ── Training control ─────────────────────────────────────────────────────────
st.divider()

status = get("training_status")

if status == "running":
    st.info("Training is in progress...")
elif status == "finished":
    st.success("Training complete! View results on the Results page.")
elif status == "error":
    st.error(f"Training failed: {get('training_error')}")

col_start, col_stop = st.columns(2)

with col_start:
    start_disabled = status == "running"
    if st.button("Start Training", type="primary", disabled=start_disabled):
        from utils.training_worker import start_training_thread
        from core.training.trainer import Trainer

        try:
            trainer = Trainer(
                decoder=decoder,
                train_dataset=dataset,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                scheduler=scheduler,
                warmup_epochs=warmup_epochs,
                early_stopping_patience=patience,
                num_workers=num_workers,
                accelerator="auto",
            )

            progress = {}
            lock = threading.Lock()
            set_("progress", progress)
            set_("training_lock", lock)
            set_("training_status", "running")
            set_("training_history", [])
            set_("training_error", None)
            set_("trainer_obj", trainer)

            thread = start_training_thread(trainer, progress, lock)
            set_("training_thread", thread)

            st.rerun()
        except Exception as e:
            st.error(f"Failed to start training: {e}")

with col_stop:
    if st.button("Stop Training", disabled=status != "running"):
        # Signal PyTorch Lightning to stop
        trainer_obj = get("trainer_obj")
        if trainer_obj is not None:
            trainer_obj.pl_trainer.should_stop = True
        set_("training_status", "stopping")
        st.info("Stop signal sent. Training will finish the current epoch.")

# ── Live progress ────────────────────────────────────────────────────────────
if status in ("running", "stopping"):
    progress = get("progress")
    lock = get("training_lock")

    if progress and lock:
        with lock:
            p = dict(progress)

        total = p.get("total_epochs", epochs)
        current = p.get("current_epoch", 0)
        st.progress(current / total if total > 0 else 0,
                     text=f"Epoch {current}/{total}")

        history = p.get("history", [])
        if history:
            set_("training_history", history)
            st.plotly_chart(plot_loss_curve(history), use_container_width=True)
            st.dataframe(format_metrics_table(history), use_container_width=True)

        # Check if finished/errored
        p_status = p.get("status", "")
        if p_status == "finished":
            set_("training_status", "finished")
            set_("training_result", p.get("result"))
            set_("training_history", p.get("history", []))
            st.rerun()
        elif p_status == "error":
            set_("training_status", "error")
            set_("training_error", p.get("error", "Unknown error"))
            st.rerun()
        else:
            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()

# ── Show final results if finished ───────────────────────────────────────────
if status == "finished":
    history = get("training_history")
    if history:
        st.plotly_chart(plot_loss_curve(history), use_container_width=True)
        st.dataframe(format_metrics_table(history), use_container_width=True)

    result = get("training_result")
    if result:
        st.subheader("Training Result")
        st.json(result)
