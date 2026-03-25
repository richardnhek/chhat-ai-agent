#!/usr/bin/env python3
"""
Local RF-DETR Training Script
==============================
Merges all training datasets → trains RF-DETR on Apple M4 MPS.

Usage:
    python train_local.py                    # Full pipeline: merge + train
    python train_local.py --merge-only       # Just merge datasets
    python train_local.py --train-only       # Just train (assumes merged data exists)
    python train_local.py --epochs 50        # Custom epochs
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict


def merge_datasets(output_dir: str = "training_data/merged") -> dict:
    """
    Merge all COCO-format datasets into one unified dataset.
    Remaps all brand classes to a single "cigarette_pack" class for initial training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dir = output_path / "train"
    val_dir = output_path / "valid"
    test_dir = output_path / "test"
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(exist_ok=True)

    # Single class for initial training (brand-agnostic detection)
    categories = [{"id": 1, "name": "cigarette_pack", "supercategory": "tobacco"}]

    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    img_id = 1
    ann_id = 1

    # Source datasets to merge
    dataset_dirs = [
        "training_data/cigarette-packs-apcpr",
        "training_data/cigarette-box",
        "training_data/cigarette-pack-detection",
        "training_data/synthetic",
        "training_data/roboflow_annotated",
        "training_data/gemini_boxes",
        "training_data/auto_labeled",
        "training_data/video_frames",
    ]

    total_images = 0
    total_annotations = 0

    for dataset_dir in dataset_dirs:
        ds_path = Path(dataset_dir)
        if not ds_path.exists():
            print(f"  Skipping {dataset_dir} (not found)")
            continue

        print(f"  Processing {dataset_dir}...")

        # Find annotation files and image directories
        for split in ["train", "valid", "test", ""]:
            split_dir = ds_path / split if split else ds_path
            ann_file = split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                ann_file = split_dir / "annotations.json"
            if not ann_file.exists():
                continue

            is_val = (split == "valid" or split == "test")

            with open(ann_file) as f:
                coco_data = json.load(f)

            # Build old_id → new_id mapping for images
            old_img_id_map = {}

            for img_info in coco_data.get("images", []):
                old_id = img_info["id"]
                new_id = img_id
                old_img_id_map[old_id] = new_id

                # Find and copy the image file
                img_filename = img_info["file_name"]
                src_img = split_dir / img_filename
                if not src_img.exists():
                    # Try without subdirectory
                    src_img = ds_path / img_filename
                if not src_img.exists():
                    continue

                # Copy to merged directory
                new_filename = f"img_{img_id:06d}{src_img.suffix}"
                dest_dir = val_dir if is_val else train_dir
                shutil.copy2(src_img, dest_dir / new_filename)

                img_entry = {
                    "id": new_id,
                    "file_name": new_filename,
                    "width": img_info.get("width", 0),
                    "height": img_info.get("height", 0),
                }

                if is_val:
                    val_images.append(img_entry)
                else:
                    train_images.append(img_entry)

                img_id += 1
                total_images += 1

            # Remap annotations — all classes → cigarette_pack (id=1)
            for ann in coco_data.get("annotations", []):
                old_img = ann["image_id"]
                if old_img not in old_img_id_map:
                    continue

                new_ann = {
                    "id": ann_id,
                    "image_id": old_img_id_map[old_img],
                    "category_id": 1,  # All → cigarette_pack
                    "bbox": ann["bbox"],
                    "area": ann.get("area", float(ann["bbox"][2]) * float(ann["bbox"][3])),
                    "iscrowd": ann.get("iscrowd", 0),
                }

                if old_img_id_map[old_img] in [i["id"] for i in val_images]:
                    val_annotations.append(new_ann)
                else:
                    train_annotations.append(new_ann)

                ann_id += 1
                total_annotations += 1

    # If no validation set, split 10% from training
    if not val_images and train_images:
        split_idx = max(1, len(train_images) // 10)
        val_images = train_images[:split_idx]
        train_images = train_images[split_idx:]

        val_img_ids = {img["id"] for img in val_images}
        val_annotations = [a for a in train_annotations if a["image_id"] in val_img_ids]
        train_annotations = [a for a in train_annotations if a["image_id"] not in val_img_ids]

        # Move validation images to val directory
        for img in val_images:
            src = train_dir / img["file_name"]
            dst = val_dir / img["file_name"]
            if src.exists():
                shutil.move(str(src), str(dst))

    # Save COCO JSON files
    train_coco = {"images": train_images, "annotations": train_annotations, "categories": categories}
    val_coco = {"images": val_images, "annotations": val_annotations, "categories": categories}

    with open(train_dir / "_annotations.coco.json", "w") as f:
        json.dump(train_coco, f)
    with open(val_dir / "_annotations.coco.json", "w") as f:
        json.dump(val_coco, f)

    # Also copy to test (use val as test)
    shutil.copy2(val_dir / "_annotations.coco.json", test_dir / "_annotations.coco.json")
    for img in val_images:
        src = val_dir / img["file_name"]
        if src.exists():
            shutil.copy2(src, test_dir / img["file_name"])

    print(f"\n  Merged dataset:")
    print(f"    Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"    Val:   {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"    Total: {total_images} images, {total_annotations} annotations")
    print(f"    Classes: 1 (cigarette_pack)")
    print(f"    Output: {output_dir}/")

    return {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "total_annotations": total_annotations,
        "output_dir": output_dir,
    }


def train_rfdetr(
    dataset_dir: str = "training_data/merged",
    epochs: int = 30,
    batch_size: int = 2,
    lr: float = 1e-4,
    model_size: str = "base",
    output_dir: str = "models",
):
    """
    Train RF-DETR locally on Apple M4 with MPS fallback.
    """
    # Set MPS fallback for unsupported operations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Model: RF-DETR {model_size}")
    print(f"  Dataset: {dataset_dir}")

    from rfdetr import RFDETRBase, RFDETRLarge

    if model_size == "large":
        model = RFDETRLarge()
    else:
        model = RFDETRBase()

    print(f"\n  Starting training... (this may take several hours on M4)")
    print(f"  Tip: monitor with Activity Monitor → GPU usage")
    print(f"  Training logs will appear below:\n")

    start_time = time.time()

    try:
        # Fix: RF-DETR uses 640px default which isn't divisible by 56 (DINOv2 block size).
        # Patch the backbone to resize inputs to nearest valid size before forward pass.
        import rfdetr.models.backbone.dinov2 as _dinov2_mod
        import torch.nn.functional as F

        _orig_forward = _dinov2_mod.DinoV2.forward

        def _patched_forward(self, x, *args, **kwargs):
            block_size = 56
            h, w = x.shape[2], x.shape[3]
            new_h = (h // block_size) * block_size
            new_w = (w // block_size) * block_size
            if new_h != h or new_w != w:
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            return _orig_forward(self, x, *args, **kwargs)

        _dinov2_mod.DinoV2.forward = _patched_forward
        print(f"  Patched DINOv2 backbone to auto-resize inputs to nearest block_size multiple")

        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        elapsed = time.time() - start_time
        hours = elapsed / 3600

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = output_path / f"rfdetr_{model_size}_{timestamp}.pth"

        # RF-DETR saves checkpoints automatically, find the best one
        checkpoint_dir = Path("output")  # Default RF-DETR output directory
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                shutil.copy2(best_checkpoint, model_path)
                print(f"\n  Model saved: {model_path}")

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE!")
        print(f"  Duration: {hours:.1f} hours")
        print(f"  Model: {model_path}")
        print(f"  Next: update detector.py model_path to use this model")
        print(f"{'='*60}")

        return {"model_path": str(model_path), "duration_hours": round(hours, 1)}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  Training failed after {elapsed/60:.1f} minutes: {e}")
        print(f"  Try: reduce batch_size to 1, or use --epochs 10 for a quick test")
        raise


def main():
    parser = argparse.ArgumentParser(description="Local RF-DETR training on Apple M4")
    parser.add_argument("--merge-only", action="store_true", help="Only merge datasets, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train, assume data is merged")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB RAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model-size", choices=["base", "large"], default="base", help="RF-DETR model size")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RF-DETR Local Training Pipeline")
    print(f"{'='*60}")

    if not args.train_only:
        print(f"\n  Step 1: Merging datasets...")
        merge_result = merge_datasets()

        if merge_result["total_annotations"] == 0:
            print("  ERROR: No training data! Generate synthetic data first:")
            print("    python synthetic_generator.py --count 500")
            sys.exit(1)

    if not args.merge_only:
        print(f"\n  Step 2: Training RF-DETR...")
        train_rfdetr(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_size=args.model_size,
        )


if __name__ == "__main__":
    main()
