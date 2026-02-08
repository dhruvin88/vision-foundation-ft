# Foundation Model Fine-Tuning (FFT)

Fine-tune lightweight decoder heads on frozen vision foundation models for classification, object detection, and semantic segmentation.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Input Image                    │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Frozen Encoder (DINOv2)             │
│  ViT-S/14 · ViT-B/14 · ViT-L/14 · ViT-G/14    │
│  384D       768D       1024D       1536D         │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌─────────────┐ ┌───────────┐ ┌──────────────┐
│Classification│ │ Detection │ │ Segmentation │
├─────────────┤ ├───────────┤ ├──────────────┤
│ LinearProbe │ │ DETRLite  │ │ LinearSeg    │
│ MLPHead     │ │ FPN       │ │ UPerNet      │
│ Transformer │ │           │ │ MaskTransf.  │
└─────────────┘ └───────────┘ └──────────────┘
```

Only the lightweight decoder heads are trained. The encoder stays frozen, making fine-tuning fast and memory-efficient.

## Quickstart

### Installation

```bash
pip install -e ".[all]"
```

### Python SDK

```python
import sdk as fft

# Load a frozen encoder
encoder = fft.Encoder("dinov2_vitb14")

# Create a classification head
head = fft.ClassificationHead(encoder, num_classes=5, head_type="mlp")

# Load dataset
dataset = fft.Dataset.from_folder("./my_images/", task="classification")

# Train
trainer = fft.Trainer(head, dataset, lr=1e-3, epochs=20)
results = trainer.fit()

# Save
trainer.save("./model.pt")
```

### CLI

```bash
# Train a classifier
fft train --data ./images --task classification --num-classes 5 --epochs 20

# Run inference
fft predict --image ./test.jpg --weights ./model.pt --decoder linear_probe \
    --task classification --num-classes 5

# List available encoders and decoders
fft info
```

### REST API

```bash
# Start the backend
uvicorn backend.main:app --reload

# Health check
curl http://localhost:8000/api/health

# List encoders
curl http://localhost:8000/api/encoders
```

## Supported Tasks

| Task | Decoders | Input Format |
|------|----------|-------------|
| Classification | LinearProbe, MLPHead, TransformerHead | Folder-per-class |
| Detection | DETRLite, FPN | COCO JSON, VOC XML, YOLO TXT |
| Segmentation | LinearSeg, UPerNet, MaskTransformer | images/ + masks/ |

## Project Structure

```
core/
  encoders/      # Frozen foundation model encoders (DINOv2)
  decoders/      # Task-specific lightweight heads
  data/          # Dataset loading, augmentations, format converters
  training/      # PyTorch Lightning trainer, schedulers, callbacks
  evaluation/    # Metrics and inference utilities
  export/        # Weight save/load and inference script generation
  cli.py         # Command-line interface
sdk/             # High-level Python SDK
backend/         # FastAPI REST API with project/dataset/training management
examples/        # End-to-end example scripts
tests/           # Unit tests
docs/            # Documentation
```

## Data Formats

FFT supports multiple annotation formats:

- **COCO JSON** -- `load_coco(annotations_path, images_dir)`
- **Pascal VOC XML** -- `load_voc(annotations_dir, images_dir)`
- **YOLO TXT** -- `load_yolo(labels_dir, images_dir, class_names)`
- **CSV** -- `load_csv(csv_path, images_dir)`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check .
```

## License

Apache-2.0
