"""Plotting and visualization helpers for the Streamlit frontend."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont


# Color palette for bounding boxes / segmentation classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def plot_loss_curve(history: list[dict]) -> go.Figure:
    """Plot training and validation loss curves from training history."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No training data yet", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font_size=16)
        return fig

    epochs = [h["epoch"] + 1 for h in history]
    fig = go.Figure()

    if any("train_loss" in h for h in history):
        train_loss = [h.get("train_loss") for h in history]
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers",
                                 name="Train Loss", line=dict(color="#636EFA")))

    if any("val_loss" in h for h in history):
        val_loss = [h.get("val_loss") for h in history]
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers",
                                 name="Val Loss", line=dict(color="#EF553B")))

    if any("val_acc" in h for h in history):
        val_acc = [h.get("val_acc") for h in history]
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers",
                                 name="Val Accuracy", yaxis="y2",
                                 line=dict(color="#00CC96", dash="dot")))
        fig.update_layout(yaxis2=dict(title="Accuracy", overlaying="y",
                                       side="right", range=[0, 1]))

    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    return fig


def draw_detection_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    class_names: list[str] | None = None,
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image.
        boxes: List of [cx, cy, w, h] normalized to [0,1].
        scores: Confidence scores per box.
        labels: Class index per box.
        class_names: Optional list of class names.

    Returns:
        PIL Image with boxes drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    for box, score, label in zip(boxes, scores, labels):
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        color = COLORS[label % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        name = class_names[label] if class_names else str(label)
        text = f"{name} {score:.2f}"
        draw.text((x1, max(0, y1 - 14)), text, fill=color)

    return img


def draw_segmentation_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a colored segmentation mask on an image.

    Args:
        image: PIL Image.
        mask: 2D numpy array of class indices (H, W).
        alpha: Overlay transparency.

    Returns:
        PIL Image with mask overlay.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pixels = overlay.load()

    # Resize mask to image size if needed
    if mask.shape != (img.size[1], img.size[0]):
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize(img.size, PILImage.NEAREST)
        mask = np.array(mask_img)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            cls = int(mask[y, x])
            if cls > 0:  # skip background (class 0)
                r, g, b = COLORS[cls % len(COLORS)]
                pixels[x, y] = (r, g, b, int(255 * alpha))

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def format_metrics_table(history: list[dict]) -> list[dict]:
    """Format training history into a table-friendly list of dicts."""
    rows = []
    for h in history:
        row = {"Epoch": h.get("epoch", 0) + 1}
        if "train_loss" in h:
            row["Train Loss"] = f"{h['train_loss']:.4f}"
        if "val_loss" in h:
            row["Val Loss"] = f"{h['val_loss']:.4f}"
        if "val_acc" in h:
            row["Val Acc"] = f"{h['val_acc']:.4f}"
        if "epoch_time_s" in h:
            row["Time (s)"] = f"{h['epoch_time_s']:.1f}"
        rows.append(row)
    return rows
