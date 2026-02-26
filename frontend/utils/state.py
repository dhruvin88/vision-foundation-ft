"""Session state management for the Streamlit frontend."""

import streamlit as st


DEFAULTS = {
    "task": "classification",
    "encoder_name": "dinov3_vitb16",
    "encoder": None,
    "decoder": None,
    "decoder_config": {},
    "dataset_path": None,
    "dataset": None,
    "dataset_stats": None,
    "trained_model": None,
    "training_status": "idle",  # idle | running | finished | error
    "training_history": [],
    "training_result": None,
    "training_thread": None,
    "training_error": None,
    "progress": {},
}


def init_state():
    """Initialize session state with defaults (only sets keys not already present)."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state():
    """Reset all session state to defaults."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def get(key: str):
    """Get a session state value."""
    return st.session_state.get(key, DEFAULTS.get(key))


def set_(key: str, value):
    """Set a session state value."""
    st.session_state[key] = value
