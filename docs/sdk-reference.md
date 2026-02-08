# SDK Reference

The SDK (`sdk` module) provides a high-level API for fine-tuning vision foundation models.

## Encoder

```python
import sdk as fft

encoder = fft.Encoder(model_name="dinov2_vitb14", input_size=518)
```

**Parameters:**
- `model_name` (str): DINOv2 variant. One of `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`, or their `_reg` variants. Default: `"dinov2_vitb14"`.
- `input_size` (int): Input image size. Must be divisible by patch size (14). Default: `518`.

**Properties:**
- `embed_dim` (int): Output embedding dimension.
- `model`: The underlying `DINOv2Encoder` instance.

**Methods:**
- `get_transform()`: Returns a `torchvision.transforms.Compose` pipeline that resizes and normalizes images for this encoder.

## Classification Heads

```python
head = fft.ClassificationHead(encoder, num_classes=10, head_type="linear")
```

**Parameters:**
- `encoder`: An `Encoder` instance or `DINOv2Encoder`.
- `num_classes` (int): Number of output classes.
- `head_type` (str): One of `"linear"`, `"mlp"`, `"transformer"`. Default: `"linear"`.

**Head types:**

| Type | Class | Description |
|------|-------|-------------|
| `"linear"` | `LinearProbe` | Single linear layer on CLS token |
| `"mlp"` | `MLPHead` | 2-layer MLP with ReLU and dropout |
| `"transformer"` | `TransformerHead` | Cross-attention decoder with learnable queries |

## Detection Heads

```python
head = fft.DetectionHead(encoder, num_classes=10, head_type="detr_lite")
```

**Parameters:**
- `encoder`: An `Encoder` instance or `DINOv2Encoder`.
- `num_classes` (int): Number of object classes.
- `head_type` (str): One of `"detr_lite"`, `"fpn"`. Default: `"detr_lite"`.
- `**kwargs`: Additional arguments passed to the decoder constructor (e.g., `num_queries=50` for DETRLite).

**Head types:**

| Type | Class | Description |
|------|-------|-------------|
| `"detr_lite"` | `DETRLiteDecoder` | Lightweight DETR with learnable object queries |
| `"fpn"` | `FPNHead` | Feature Pyramid Network with anchor-based detection |

**DETRLite output:** `{"pred_logits": (B, Q, C+1), "pred_boxes": (B, Q, 4)}`

**FPNHead output:** `{"cls_preds": [...], "reg_preds": [...]}`

## Segmentation Heads

```python
head = fft.SegmentationHead(encoder, num_classes=5, head_type="upernet")
```

**Parameters:**
- `encoder`: An `Encoder` instance or `DINOv2Encoder`.
- `num_classes` (int): Number of segmentation classes.
- `head_type` (str): One of `"linear"`, `"upernet"`, `"mask_transformer"`. Default: `"linear"`.
- `**kwargs`: Additional arguments (e.g., `output_size=256`).

**Head types:**

| Type | Class | Description |
|------|-------|-------------|
| `"linear"` | `LinearSegHead` | Per-patch linear classifier + bilinear upsampling |
| `"upernet"` | `UPerNetHead` | UPerNet with pyramid pooling module |
| `"mask_transformer"` | `MaskTransformerHead` | Dot-product mask prediction |

**Output:** Segmentation logits of shape `(B, num_classes, H, W)`.

## Dataset

```python
dataset = fft.Dataset.from_folder("./data/", task="classification", transform=encoder.get_transform())
```

`fft.Dataset` is an alias for `core.data.dataset.FFTDataset`.

**Class methods:**
- `from_folder(root, task, transform=None)`: Load from directory structure.

**Instance methods:**
- `split(val_ratio=0.2, seed=42) -> (train_ds, val_ds)`: Split into train/val.
- `get_stats() -> dict`: Return dataset statistics.

**Properties:**
- `task` (str): Task type.
- `class_names` (list[str]): Class name list.
- `samples` (list[dict]): Raw sample list.

## Trainer

```python
trainer = fft.Trainer(
    decoder=head,
    train_dataset=dataset,
    val_dataset=None,      # auto-split if None
    lr=1e-3,
    epochs=50,
    batch_size=32,
    scheduler="cosine",    # "cosine", "step", "constant"
    warmup_epochs=5,
    augmentation="light",  # "none", "light", "heavy"
    early_stopping_patience=10,
    val_ratio=0.2,
)
```

**Methods:**
- `fit() -> dict`: Run training. Returns results dict with `best_model_path`, `best_val_loss`, `epochs_trained`.
- `save(path)`: Save decoder weights.
- `load(path)`: Load decoder weights.

## Inference

```python
results = fft.run_inference(decoder, image_paths)
```

Runs batch inference on a list of image paths. Returns task-specific predictions.

## Metrics

```python
metrics = fft.compute_metrics(predictions, targets, task="classification")
```

Computes task-specific metrics:
- **Classification**: accuracy, per-class accuracy
- **Detection**: precision, recall, F1, mAP@50
- **Segmentation**: mIoU, pixel accuracy, per-class IoU

## Weights

```python
fft.save_decoder_weights(decoder, "model.pt")
fft.load_decoder_weights(decoder, "model.pt")
```

Saves only the decoder parameters (encoder is frozen, not saved). Includes metadata about the decoder class, task, and architecture.

## Script Generation

```python
fft.generate_inference_script(
    decoder_class="LinearProbe",
    task="classification",
    encoder_name="dinov2_vitb14",
    num_classes=5,
    weights_path="model.pt",
    output_path="inference.py",
)
```

Generates a standalone Python script for inference that can be distributed without the full library.
