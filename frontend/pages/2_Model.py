"""Model configuration page — encoder + decoder setup."""

import streamlit as st

from utils.state import init_state, get, set_

init_state()

st.header("Model Configuration")

task = get("task")

# ── Ensure SDK is importable ────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── Encoder ──────────────────────────────────────────────────────────────────
st.subheader("Encoder")

ENCODER_VARIANTS = {
    "dinov2_vits14": {"embed_dim": 384, "params": "21M", "num_blocks": 12},
    "dinov2_vitb14": {"embed_dim": 768, "params": "86M", "num_blocks": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "params": "300M", "num_blocks": 24},
    "dinov2_vitg14": {"embed_dim": 1536, "params": "1.1B", "num_blocks": 40},
}

encoder_name = st.selectbox(
    "DINOv2 variant",
    list(ENCODER_VARIANTS.keys()),
    index=list(ENCODER_VARIANTS.keys()).index(get("encoder_name")),
)

info = ENCODER_VARIANTS[encoder_name]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Embedding dim", info["embed_dim"])
with col2:
    st.metric("Patch size", 14)
with col3:
    st.metric("Parameters", info["params"])

set_("encoder_name", encoder_name)

# ── Decoder ──────────────────────────────────────────────────────────────────
st.subheader("Decoder")

decoder_config = dict(get("decoder_config") or {})

# Determine if the selected head type needs multi-scale features
needs_multiscale = False

if task == "classification":
    head_type = st.selectbox("Head type", ["linear", "mlp", "transformer"],
                             index=["linear", "mlp", "transformer"].index(
                                 decoder_config.get("head_type", "linear")))
    num_classes = st.number_input("Number of classes", min_value=2, max_value=10000,
                                  value=decoder_config.get("num_classes", 2))
    decoder_config = {"head_type": head_type, "num_classes": num_classes}

elif task == "detection":
    head_type = st.selectbox("Head type", ["detr_lite", "fpn"],
                             index=["detr_lite", "fpn"].index(
                                 decoder_config.get("head_type", "detr_lite")))
    num_classes = st.number_input("Number of classes", min_value=1, max_value=1000,
                                  value=decoder_config.get("num_classes", 1))
    if head_type == "detr_lite":
        col1, col2 = st.columns(2)
        with col1:
            num_queries = st.number_input("Num queries", min_value=10, max_value=500,
                                          value=decoder_config.get("num_queries", 100))
            num_decoder_layers = st.number_input("Decoder layers", min_value=1, max_value=12,
                                                  value=decoder_config.get("num_decoder_layers", 3))
        with col2:
            hidden_dim = st.selectbox("Hidden dim", [128, 256, 512],
                                      index=[128, 256, 512].index(
                                          decoder_config.get("hidden_dim", 256)))
            num_heads = st.number_input("Attention heads", min_value=1, max_value=16,
                                        value=decoder_config.get("num_heads", 8))
        decoder_config = {
            "head_type": head_type, "num_classes": num_classes,
            "num_queries": num_queries, "num_decoder_layers": num_decoder_layers,
            "hidden_dim": hidden_dim, "num_heads": num_heads,
        }
    else:
        needs_multiscale = True
        fpn_channels = st.selectbox("FPN channels", [128, 256, 512],
                                    index=[128, 256, 512].index(
                                        decoder_config.get("fpn_channels", 256)))
        decoder_config = {"head_type": head_type, "num_classes": num_classes,
                          "fpn_channels": fpn_channels}

else:  # segmentation
    head_type = st.selectbox("Head type", ["linear", "upernet", "mask_transformer"],
                             index=["linear", "upernet", "mask_transformer"].index(
                                 decoder_config.get("head_type", "linear")))
    num_classes = st.number_input("Number of classes", min_value=2, max_value=1000,
                                  value=decoder_config.get("num_classes", 2))
    decoder_config = {"head_type": head_type, "num_classes": num_classes}
    if head_type == "upernet":
        needs_multiscale = True

# ── Multi-scale feature extraction ──────────────────────────────────────────
if needs_multiscale:
    st.divider()
    st.subheader("Multi-Scale Features")
    st.info(
        f"**{head_type.upper()}** uses intermediate encoder layers for multi-scale "
        f"feature extraction. This produces richer features at multiple resolutions, "
        f"improving performance for {task}."
    )

    n_blocks = info["num_blocks"]
    # Default: 4 evenly-spaced layers
    default_layers = [n_blocks // 4 - 1, n_blocks // 2 - 1, 3 * n_blocks // 4 - 1, n_blocks - 1]

    use_custom = st.checkbox("Customize intermediate layers",
                             value=False,
                             help=f"Default: {default_layers} (4 evenly-spaced from {n_blocks} blocks)")
    if use_custom:
        layers_str = st.text_input(
            f"Layer indices (0-{n_blocks - 1}, comma-separated)",
            value=", ".join(str(l) for l in default_layers),
        )
        try:
            custom_layers = [int(x.strip()) for x in layers_str.split(",")]
            for idx in custom_layers:
                if idx < 0 or idx >= n_blocks:
                    st.error(f"Layer {idx} out of range (0-{n_blocks - 1})")
                    custom_layers = None
                    break
        except ValueError:
            st.error("Enter comma-separated integers")
            custom_layers = None
        decoder_config["intermediate_layers"] = custom_layers or default_layers
    else:
        decoder_config["intermediate_layers"] = default_layers

    st.caption(f"Using layers: **{decoder_config.get('intermediate_layers', default_layers)}**")

set_("decoder_config", decoder_config)

# ── Build model ──────────────────────────────────────────────────────────────
st.divider()
if st.button("Build Model", type="primary"):
    with st.spinner("Loading encoder and building decoder..."):
        try:
            from sdk import Encoder, ClassificationHead, DetectionHead, SegmentationHead

            # Pass intermediate_layers to encoder if the head needs them
            intermediate_layers = decoder_config.get("intermediate_layers")
            encoder = Encoder(encoder_name, intermediate_layers=intermediate_layers)
            set_("encoder", encoder)

            cfg = dict(decoder_config)
            head_type = cfg.pop("head_type")
            num_classes = cfg.pop("num_classes")
            cfg.pop("intermediate_layers", None)  # Already configured on encoder

            if task == "classification":
                decoder = ClassificationHead(encoder, num_classes, head_type=head_type)
            elif task == "detection":
                decoder = DetectionHead(encoder, num_classes, head_type=head_type, **cfg)
            else:
                decoder = SegmentationHead(encoder, num_classes, head_type=head_type, **cfg)

            set_("decoder", decoder)
            trainable = decoder.num_trainable_params()
            multiscale_msg = ""
            if intermediate_layers:
                multiscale_msg = f" Multi-scale layers: {intermediate_layers}."
            st.success(f"Model built: {type(decoder).__name__} with {trainable:,} trainable parameters.{multiscale_msg}")
        except Exception as e:
            st.error(f"Failed to build model: {e}")

# Show current model status
decoder = get("decoder")
if decoder is not None:
    st.info(f"Current model: **{type(decoder).__name__}** ({decoder.task}), "
            f"{decoder.num_trainable_params():,} trainable params")
