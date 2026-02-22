"""Re-run vits14 detection experiments with proper mAP metrics."""

import json
import time
from pathlib import Path
from datetime import datetime


def main():
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "experiments" / "datasets"

    # Import here to avoid slow startup message before print
    from run_dinov2_benchmarks import run_detection_experiment

    coco_path = datasets_dir / "coco_detection_subset"
    if not (coco_path / "annotations.json").exists():
        print(f"COCO dataset not found at {coco_path}")
        return

    with open(coco_path / "annotations.json") as f:
        coco_data = json.load(f)
        num_classes = len(coco_data["categories"])

    print("\n" + "=" * 60)
    print("RE-RUNNING: vits14 detection with proper mAP metrics")
    print("=" * 60)
    print(f"Dataset: {coco_path}")
    print(f"Classes: {num_classes}")

    for decoder in ["rtdetr", "detr_lite", "detr_multiscale"]:
        try:
            run_detection_experiment(
                coco_path,
                "coco_subset",
                "dinov2_vits14",
                decoder,
                num_classes=num_classes,
                epochs=20,
                batch_size=8,
                lr=1e-4,
            )
        except Exception as e:
            print(f"ERROR: dinov2_vits14 + {decoder} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
