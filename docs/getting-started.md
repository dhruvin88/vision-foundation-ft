# Getting Started

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, not required)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/foundation-model-any-ft.git
cd foundation-model-any-ft

# Install core library
pip install -e .

# Install with backend support
pip install -e ".[backend]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

## Your First Classification Model

### 1. Prepare Your Data

Organize images into folders by class:

```
my_dataset/
  cats/
    img_001.jpg
    img_002.jpg
  dogs/
    img_001.jpg
    img_002.jpg
```

### 2. Train with the SDK

```python
import sdk as fft

# Load encoder (downloads weights on first use)
encoder = fft.Encoder("dinov2_vitb14")

# Create a classification head
head = fft.ClassificationHead(encoder, num_classes=2, head_type="linear")

# Load dataset with encoder's transforms
dataset = fft.Dataset.from_folder(
    "./my_dataset/",
    task="classification",
    transform=encoder.get_transform(),
)

# Train (auto-splits into train/val)
trainer = fft.Trainer(head, dataset, lr=1e-3, epochs=20, batch_size=32)
results = trainer.fit()
print(f"Training complete: {results}")

# Save weights
trainer.save("./classifier.pt")
```

### 3. Or Train with the CLI

```bash
fft train \
    --data ./my_dataset \
    --task classification \
    --num-classes 2 \
    --encoder dinov2_vitb14 \
    --decoder linear_probe \
    --epochs 20 \
    --output ./output
```

### 4. Run Inference

```python
from core.export.weights import load_decoder_weights
from core.evaluation.inference import run_inference

# Recreate and load decoder
encoder = fft.Encoder("dinov2_vitb14")
head = fft.ClassificationHead(encoder, num_classes=2)
load_decoder_weights(head, "./classifier.pt")

# Predict
results = run_inference(head, ["./test_image.jpg"])
print(results)
```

## Object Detection

### Data Format

Expects COCO JSON format:

```
detection_dataset/
  images/
    img_001.jpg
    img_002.jpg
  annotations.json
```

### Train

```python
encoder = fft.Encoder("dinov2_vitb14")
head = fft.DetectionHead(encoder, num_classes=10, head_type="detr_lite")
dataset = fft.Dataset.from_folder("./detection_dataset/", task="detection")
trainer = fft.Trainer(head, dataset, lr=1e-4, epochs=50)
trainer.fit()
```

## Semantic Segmentation

### Data Format

```
seg_dataset/
  images/
    img_001.jpg
    img_002.jpg
  masks/
    img_001.png   # Pixel values = class IDs
    img_002.png
```

### Train

```python
encoder = fft.Encoder("dinov2_vitb14")
head = fft.SegmentationHead(encoder, num_classes=5, head_type="upernet")
dataset = fft.Dataset.from_folder("./seg_dataset/", task="segmentation")
trainer = fft.Trainer(head, dataset, lr=1e-4, epochs=50)
trainer.fit()
```

## Choosing an Encoder

| Variant | Embed Dim | Speed | Quality |
|---------|-----------|-------|---------|
| `dinov2_vits14` | 384 | Fastest | Good |
| `dinov2_vitb14` | 768 | Fast | Better |
| `dinov2_vitl14` | 1024 | Moderate | Best |
| `dinov2_vitg14` | 1536 | Slowest | Highest |

Add `_reg` suffix (e.g., `dinov2_vitb14_reg`) for register variants with improved attention maps.

## Choosing a Decoder

### Classification
- **LinearProbe**: Fastest to train, strong baseline
- **MLPHead**: Slightly more capacity, minimal overhead
- **TransformerHead**: Most expressive, uses cross-attention

### Detection
- **DETRLite**: End-to-end, no anchors or NMS needed
- **FPN**: Multi-scale with anchor-based predictions

### Segmentation
- **LinearSeg**: Simple per-patch classifier
- **UPerNet**: Multi-scale with pyramid pooling (strongest)
- **MaskTransformer**: Dot-product mask prediction

## Next Steps

- [SDK Reference](sdk-reference.md) for the full API
- [Architecture](architecture.md) for design details
