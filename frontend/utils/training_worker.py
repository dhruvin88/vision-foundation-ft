"""Background training worker using threading."""

import threading
import traceback

import pytorch_lightning as pl

from core.training.callbacks import MetricsLoggerCallback


def run_training(trainer, progress: dict, lock: threading.Lock):
    """Run trainer.fit() in a background thread, updating progress dict.

    The MetricsLoggerCallback already exists in trainer.pl_trainer.callbacks.
    We poll it for history after each epoch via a custom lightweight callback.

    Args:
        trainer: core.training.trainer.Trainer instance (already configured)
        progress: shared dict for status updates
        lock: threading lock for thread-safe writes
    """

    class ProgressBridge(pl.Callback):
        """Bridges MetricsLoggerCallback history to the shared progress dict."""

        def on_train_start(self, pl_trainer, pl_module):
            with lock:
                progress["status"] = "running"
                progress["current_epoch"] = 0
                progress["total_epochs"] = pl_trainer.max_epochs

        def on_train_epoch_end(self, pl_trainer, pl_module):
            # Find the MetricsLoggerCallback and grab its history
            for cb in pl_trainer.callbacks:
                if isinstance(cb, MetricsLoggerCallback):
                    with lock:
                        progress["current_epoch"] = pl_trainer.current_epoch + 1
                        progress["history"] = list(cb.get_history())
                    break

        def on_train_end(self, pl_trainer, pl_module):
            with lock:
                progress["status"] = "finished"

    # Inject the bridge callback
    trainer.pl_trainer.callbacks.append(ProgressBridge())

    try:
        with lock:
            progress["status"] = "starting"
        result = trainer.fit()
        with lock:
            progress["status"] = "finished"
            progress["result"] = result
            # Final history grab
            for cb in trainer.pl_trainer.callbacks:
                if isinstance(cb, MetricsLoggerCallback):
                    progress["history"] = list(cb.get_history())
                    break
    except Exception as e:
        with lock:
            progress["status"] = "error"
            progress["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def start_training_thread(trainer, progress: dict, lock: threading.Lock) -> threading.Thread:
    """Start training in a daemon thread and return the thread object."""
    thread = threading.Thread(
        target=run_training, args=(trainer, progress, lock), daemon=True
    )
    thread.start()
    return thread
