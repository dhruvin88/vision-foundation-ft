"""Prepare a retail/grocery detection dataset for benchmarking.

Two modes:

  1. COCO-retail (default, no downloads needed if val2017 already present):
     Filters COCO val2017 to food, drink, and shoppable product categories
     (~20 categories, ~3000 images).  Same infrastructure as coco_15cls.

  2. Roboflow (recommended for authentic shelf/SKU detection):
     Downloads a public retail dataset from Roboflow Universe.
     Requires a free API key from https://roboflow.com

     pip install roboflow
     python experiments/prepare_retail_dataset.py --source roboflow --api-key YOUR_KEY

USAGE
-----
# COCO-retail (fastest, no extra downloads):
    python experiments/prepare_retail_dataset.py

# Roboflow grocery dataset (more authentic, needs free API key):
    python experiments/prepare_retail_dataset.py \\
        --source roboflow \\
        --api-key YOUR_ROBOFLOW_KEY \\
        --workspace roboflow-universe-projects \\
        --project grocery-dataset-q9fj2 \\
        --version 3

OUTPUT
------
    experiments/datasets/retail_coco/       (COCO-retail mode)
    experiments/datasets/retail_roboflow/   (Roboflow mode)
        images/
        annotations.json                    (COCO JSON)
"""

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Retail categories from COCO
# ---------------------------------------------------------------------------
# Chosen to represent items you'd find in a supermarket, drug store or gift
# shop — edible products, beverages, personal-care items and giftware.
# We deliberately exclude large-scene classes (person, car, chair …).

RETAIL_COCO_CATEGORIES = {
    # Beverages / kitchen
    "bottle", "wine glass", "cup", "bowl",
    # Fresh produce
    "banana", "apple", "orange", "broccoli", "carrot",
    # Prepared / packaged food
    "sandwich", "hot dog", "pizza", "donut", "cake",
    # Personal care / stationery / toys
    "scissors", "toothbrush", "book", "teddy bear",
    # Accessories / bags (impulse buy area)
    "backpack", "handbag",
}


# ---------------------------------------------------------------------------
# Mode 1 — COCO retail subset
# ---------------------------------------------------------------------------

def prepare_coco_retail(data_dir: Path) -> Path:
    """Filter COCO val2017 to retail categories."""
    out_dir = data_dir / "retail_coco"

    if (out_dir / "annotations.json").exists():
        with open(out_dir / "annotations.json") as f:
            d = json.load(f)
        print(f"retail_coco already at {out_dir} "
              f"({len(d['images'])} images, {len(d['categories'])} classes)")
        return out_dir

    print("\n" + "=" * 60)
    print("Preparing COCO-retail dataset")
    print(f"Categories: {sorted(RETAIL_COCO_CATEGORIES)}")
    print("=" * 60)

    # Download annotations if needed
    from urllib.request import urlretrieve

    ann_path = data_dir / "instances_val2017.json"
    if not ann_path.exists():
        ann_zip = data_dir / "annotations_trainval2017.zip"
        if not ann_zip.exists():
            print("Downloading COCO 2017 annotations (~241 MB)...")
            urlretrieve(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                ann_zip,
            )
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_val2017.json", data_dir)
        shutil.move(str(data_dir / "annotations" / "instances_val2017.json"), ann_path)
        ann_sub = data_dir / "annotations"
        if ann_sub.exists() and not any(ann_sub.iterdir()):
            ann_sub.rmdir()

    print("Loading COCO annotations...")
    with open(ann_path) as f:
        coco = json.load(f)

    # --- Select retail categories ---
    retail_cats = [
        c for c in coco["categories"]
        if c["name"] in RETAIL_COCO_CATEGORIES
    ]
    if not retail_cats:
        raise RuntimeError("No retail categories found in annotations.json — check COCO format.")

    retail_cat_ids = {c["id"] for c in retail_cats}
    print(f"\nFound {len(retail_cats)} matching categories:")
    for c in sorted(retail_cats, key=lambda x: x["name"]):
        cnt = sum(1 for a in coco["annotations"] if a["category_id"] == c["id"])
        print(f"  {c['name']:<20} {cnt:>5} annotations")

    # Re-index categories 1…N
    sorted_cats = sorted(retail_cats, key=lambda x: x["name"])
    old_to_new = {c["id"]: i + 1 for i, c in enumerate(sorted_cats)}
    new_categories = [{"id": i + 1, "name": c["name"]} for i, c in enumerate(sorted_cats)]

    # Filter annotations
    filtered_anns = [
        {**a, "category_id": old_to_new[a["category_id"]]}
        for a in coco["annotations"]
        if a["category_id"] in retail_cat_ids
    ]

    # Only keep images that have at least one retail annotation
    img_ids_with_retail = {a["image_id"] for a in filtered_anns}
    filtered_images = [img for img in coco["images"] if img["id"] in img_ids_with_retail]

    print(f"\nFiltered to {len(filtered_images)} images, {len(filtered_anns)} annotations")

    # --- Extract images from val2017.zip ---
    val_zip = data_dir / "val2017.zip"
    if not val_zip.exists():
        print("Downloading COCO val2017 images (~816 MB)...")
        urlretrieve(
            "http://images.cocodataset.org/zips/val2017.zip",
            val_zip,
        )

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    needed = {img["file_name"] for img in filtered_images}
    already = {f.name for f in images_dir.glob("*.jpg")}
    to_extract = needed - already

    if to_extract:
        print(f"Extracting {len(to_extract)} images from val2017.zip...")
        with zipfile.ZipFile(val_zip) as z:
            members = [m for m in z.namelist() if Path(m).name in to_extract]
            for member in tqdm(members, desc="Extracting"):
                fname = Path(member).name
                dst = images_dir / fname
                if not dst.exists():
                    with z.open(member) as src, open(dst, "wb") as out_f:
                        out_f.write(src.read())
    else:
        print("All images already extracted.")

    # Drop any images whose file is missing
    present = {f.name for f in images_dir.glob("*.jpg")}
    filtered_images = [img for img in filtered_images if img["file_name"] in present]
    valid_ids = {img["id"] for img in filtered_images}
    filtered_anns = [a for a in filtered_anns if a["image_id"] in valid_ids]

    coco_out = {
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": new_categories,
    }
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(coco_out, f)

    print(f"\nretail_coco ready:")
    print(f"  Images:      {len(filtered_images)}")
    print(f"  Annotations: {len(filtered_anns)}")
    print(f"  Classes:     {len(new_categories)}")
    print(f"  Avg ann/img: {len(filtered_anns)/max(len(filtered_images),1):.1f}")
    print(f"  Path:        {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Mode 2 — Roboflow Universe
# ---------------------------------------------------------------------------

def prepare_roboflow(
    data_dir: Path,
    api_key: str,
    workspace: str,
    project: str,
    version: int,
) -> Path:
    """Download a Roboflow Universe dataset in COCO format.

    Free API keys: https://roboflow.com  (sign up → account → API key)

    Recommended retail datasets on Roboflow Universe:
      - roboflow-universe-projects / grocery-dataset-q9fj2        (~600 images, 25 classes)
      - danielkorthals / supermarket-products                       (~500 images, 10 classes)
      - roboflow / pistachio-nuts                                   (packaged goods example)
      - search https://universe.roboflow.com for "retail", "grocery", "shelf"
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "roboflow package not installed.\n"
            "Install with: pip install roboflow\n"
            "Get a free API key at: https://roboflow.com"
        )

    out_dir = data_dir / "retail_roboflow"

    if (out_dir / "annotations.json").exists():
        with open(out_dir / "annotations.json") as f:
            d = json.load(f)
        print(f"retail_roboflow already at {out_dir} "
              f"({len(d['images'])} images, {len(d['categories'])} classes)")
        return out_dir

    print("\n" + "=" * 60)
    print(f"Downloading Roboflow dataset: {workspace}/{project} v{version}")
    print("=" * 60)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    # Download into a temp subdirectory then move to standard location
    tmp_dir = data_dir / "_rf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataset = ver.download("coco", location=str(tmp_dir))

    # Roboflow COCO exports have train/valid/test splits with separate jsons.
    # Merge them into one flat dataset for our pipeline.
    merged = _merge_roboflow_coco_splits(Path(dataset.location))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Move images
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    src_images = list(Path(dataset.location).rglob("*.jpg")) + \
                 list(Path(dataset.location).rglob("*.png")) + \
                 list(Path(dataset.location).rglob("*.jpeg"))
    print(f"Copying {len(src_images)} images...")
    for img_path in tqdm(src_images):
        shutil.copy2(img_path, images_dir / img_path.name)

    # Fix file_name to just basename (no split prefix)
    for img in merged["images"]:
        img["file_name"] = Path(img["file_name"]).name

    with open(out_dir / "annotations.json", "w") as f:
        json.dump(merged, f)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nretail_roboflow ready:")
    print(f"  Images:      {len(merged['images'])}")
    print(f"  Annotations: {len(merged['annotations'])}")
    print(f"  Classes:     {len(merged['categories'])}")
    print(f"  Path:        {out_dir}")
    return out_dir


def _merge_roboflow_coco_splits(rf_dir: Path) -> dict:
    """Merge train/valid/test COCO JSONs from a Roboflow download into one."""
    split_jsons = sorted(rf_dir.rglob("_annotations.coco.json"))
    if not split_jsons:
        # Some Roboflow exports use a different name
        split_jsons = sorted(rf_dir.rglob("*.json"))

    merged_images, merged_anns, categories = [], [], None
    img_id_offset, ann_id_offset = 0, 0

    for jpath in split_jsons:
        with open(jpath) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        old_to_new_img = {}
        for img in data.get("images", []):
            new_id = img_id_offset + img["id"]
            old_to_new_img[img["id"]] = new_id
            merged_images.append({**img, "id": new_id})

        for ann in data.get("annotations", []):
            merged_anns.append({
                **ann,
                "id": ann_id_offset + ann["id"],
                "image_id": old_to_new_img.get(ann["image_id"], ann["image_id"]),
            })

        img_id_offset += len(data.get("images", []))
        ann_id_offset += len(data.get("annotations", []))

    return {
        "images": merged_images,
        "annotations": merged_anns,
        "categories": categories or [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare retail detection dataset")
    parser.add_argument("--source", choices=["coco", "roboflow"], default="coco",
                        help="Dataset source: 'coco' (default) or 'roboflow'")

    # Roboflow options
    parser.add_argument("--api-key", default="",
                        help="Roboflow API key (required for --source roboflow)")
    parser.add_argument("--workspace", default="roboflow-universe-projects",
                        help="Roboflow workspace name")
    parser.add_argument("--project", default="grocery-dataset-q9fj2",
                        help="Roboflow project name")
    parser.add_argument("--version", type=int, default=3,
                        help="Roboflow dataset version number")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    data_dir = Path(__file__).parent / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "roboflow":
        if not args.api_key:
            parser.error(
                "--api-key is required for --source roboflow.\n"
                "Get a free key at https://roboflow.com (account → API key)"
            )
        out = prepare_roboflow(
            data_dir,
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
        )
    else:
        out = prepare_coco_retail(data_dir)

    print("\n" + "=" * 60)
    print("RETAIL DATASET READY")
    print("=" * 60)
    print(f"  Path: {out}")
    print()
    print("Run locally:")
    print(f"  python -m core.cli train --task detection --data {out} --decoder rtdetr")
    print()
    print("Run on Modal:")
    print("  # First add dataset to Modal volume:")
    print("  modal run experiments/modal_train.py::upload_datasets")
    print("  # Then train:")
    print("  modal run experiments/modal_train.py::main")


if __name__ == "__main__":
    main()
