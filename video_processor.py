#!/usr/bin/env python3
"""
Video Processor — Extracts frames from cigarette pack videos for training data.

Usage:
    # Process a single video
    python video_processor.py video.mp4 --brand "MEVIUS" --sku "MEVIUS ORIGINAL"

    # Process all videos in a folder
    python video_processor.py --dir videos/ --mapping video_mapping.json

    # Process with auto-brand detection (uses classifier)
    python video_processor.py video.mp4 --auto-detect

The mapping JSON format:
{
    "mevius_video.mp4": {"brand": "MEVIUS", "sku": "MEVIUS ORIGINAL"},
    "ara_red.mp4": {"brand": "ARA", "sku": "ARA RED"}
}
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from brands import BRANDS_AND_SKUS


def extract_frames(
    video_path: str,
    fps: float = 2.0,
    max_frames: int = 50,
    min_blur_score: float = 30.0,
) -> list[dict]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        fps: Frames per second to extract (default 2 = 1 frame every 0.5s)
        max_frames: Maximum frames to extract
        min_blur_score: Minimum Laplacian variance (skip blurry frames)

    Returns:
        List of {"frame_idx": int, "image": PIL.Image, "blur_score": float}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    frame_interval = max(1, frame_interval)

    print(f"    Video: {video_fps:.0f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"    Extracting every {frame_interval} frames ({fps} fps target)")

    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Check blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score >= min_blur_score:
                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                frames.append({
                    "frame_idx": frame_idx,
                    "image": pil_img,
                    "blur_score": blur_score,
                })

        frame_idx += 1

    cap.release()
    print(f"    Extracted {len(frames)} frames (skipped blurry ones)")
    return frames


def apply_augmentations(image: Image.Image) -> list[Image.Image]:
    """
    Generate augmented versions of a frame to simulate store conditions.
    Returns the original + augmented versions.
    """
    results = [image]  # Original

    # Rotation variants (packs appear upside down, sideways)
    results.append(image.rotate(180))
    results.append(image.rotate(90, expand=True))
    results.append(image.rotate(270, expand=True))

    # Brightness variations
    from PIL import ImageEnhance
    results.append(ImageEnhance.Brightness(image).enhance(0.7))  # Darker
    results.append(ImageEnhance.Brightness(image).enhance(1.3))  # Brighter

    # Slight blur (simulates dirty glass)
    from PIL import ImageFilter
    results.append(image.filter(ImageFilter.GaussianBlur(radius=1.5)))

    return results


def process_video(
    video_path: str,
    brand: str,
    sku: str | None = None,
    output_dir: str = "training_data/video_frames",
    fps: float = 2.0,
    augment: bool = True,
) -> dict:
    """
    Process a single video: extract frames, augment, save for training.
    """
    video_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing: {video_path.name}")
    print(f"  Brand: {brand}, SKU: {sku or 'auto'}")

    # Extract frames
    frames = extract_frames(str(video_path), fps=fps)

    if not frames:
        print(f"    No valid frames extracted!")
        return {"frames_saved": 0}

    # Save frames (with optional augmentation)
    saved_count = 0
    video_stem = video_path.stem.replace(" ", "_").lower()

    for i, frame_data in enumerate(frames):
        img = frame_data["image"]

        if augment:
            variants = apply_augmentations(img)
        else:
            variants = [img]

        for v_idx, variant in enumerate(variants):
            filename = f"{video_stem}_{brand}_{i:03d}_v{v_idx}.jpg"
            variant.save(out_dir / filename, format="JPEG", quality=90)
            saved_count += 1

    print(f"    Saved {saved_count} images ({len(frames)} frames × {len(variants)} variants)")

    # Save metadata
    meta = {
        "video": video_path.name,
        "brand": brand,
        "sku": sku,
        "frames_extracted": len(frames),
        "augmented_images": saved_count,
        "output_dir": str(out_dir),
    }

    return meta


def process_directory(
    video_dir: str,
    mapping_file: str | None = None,
    output_dir: str = "training_data/video_frames",
    fps: float = 2.0,
    augment: bool = True,
) -> list[dict]:
    """
    Process all videos in a directory.

    If mapping_file is provided, use it to map video filenames to brand/SKU.
    Otherwise, try to infer brand from filename.
    """
    video_dir = Path(video_dir)
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

    videos = [f for f in video_dir.iterdir()
              if f.suffix.lower() in video_extensions]

    if not videos:
        print(f"No video files found in {video_dir}")
        return []

    print(f"Found {len(videos)} videos in {video_dir}")

    # Load mapping if provided
    mapping = {}
    if mapping_file and Path(mapping_file).exists():
        with open(mapping_file) as f:
            mapping = json.load(f)
        print(f"Loaded mapping for {len(mapping)} videos")

    results = []
    all_brands = list(BRANDS_AND_SKUS.keys())

    for video in sorted(videos):
        # Get brand/SKU from mapping or filename
        if video.name in mapping:
            brand = mapping[video.name].get("brand", "UNKNOWN")
            sku = mapping[video.name].get("sku")
        else:
            # Try to infer from filename
            brand = _infer_brand_from_filename(video.name, all_brands)
            sku = None

        if brand == "UNKNOWN":
            print(f"\n  Skipping {video.name} — cannot determine brand.")
            print(f"  Create a mapping file or name videos like: mevius_original.mp4")
            continue

        meta = process_video(
            str(video), brand=brand, sku=sku,
            output_dir=output_dir, fps=fps, augment=augment,
        )
        results.append(meta)

    # Summary
    total_images = sum(r.get("augmented_images", 0) for r in results)
    print(f"\n{'='*60}")
    print(f"  DONE: {len(results)} videos processed")
    print(f"  Total training images generated: {total_images}")
    print(f"  Output: {output_dir}/")

    return results


def _infer_brand_from_filename(filename: str, brands: list[str]) -> str:
    """Try to match a brand name from the video filename."""
    name_upper = filename.upper().replace("-", " ").replace("_", " ")
    for brand in sorted(brands, key=len, reverse=True):  # Longest match first
        if brand.upper() in name_upper:
            return brand
    return "UNKNOWN"


def rebuild_classifier_with_video_frames(
    video_frames_dir: str = "training_data/video_frames",
    reference_dir: str = "reference_images",
):
    """
    After processing videos, retrain the brand classifier
    using both reference images AND video frames.
    """
    from brand_classifier import train_classifier

    # For now, the classifier trains on reference_images.
    # TODO: merge video frames into the training pipeline.
    # For a quick win, copy video frames into reference_images
    # with proper naming, then retrain.

    vf_dir = Path(video_frames_dir)
    ref_dir = Path(reference_dir)

    if not vf_dir.exists():
        print("No video frames found.")
        return

    # Copy frames to reference_images with brand-based naming
    frame_count = 0
    for img_path in vf_dir.glob("*.jpg"):
        # Filename format: videoname_BRAND_001_v0.jpg
        parts = img_path.stem.split("_")
        # Find brand in parts
        for brand in BRANDS_AND_SKUS.keys():
            if brand.replace(" ", "_").upper() in "_".join(parts).upper():
                dest = ref_dir / f"video_{img_path.name}"
                if not dest.exists():
                    import shutil
                    shutil.copy2(img_path, dest)

                    # Update mapping.json
                    mapping_path = ref_dir / "mapping.json"
                    with open(mapping_path) as f:
                        mapping = json.load(f)
                    mapping[f"video_{img_path.name}"] = {
                        "brand": brand,
                        "sku": None,
                        "source": "video",
                    }
                    with open(mapping_path, "w") as f:
                        json.dump(mapping, f, indent=2)
                    frame_count += 1
                break

    print(f"\nAdded {frame_count} video frames to reference catalog")
    print("Retraining brand classifier...")
    train_classifier(reference_dir=reference_dir, epochs=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cigarette pack videos for training")
    parser.add_argument("video", nargs="?", help="Video file to process")
    parser.add_argument("--dir", help="Directory of videos to process")
    parser.add_argument("--brand", help="Brand name (e.g., MEVIUS)")
    parser.add_argument("--sku", help="SKU name (e.g., MEVIUS ORIGINAL)")
    parser.add_argument("--mapping", help="JSON mapping file (video_name → brand/sku)")
    parser.add_argument("--output", default="training_data/video_frames", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--retrain", action="store_true", help="Retrain classifier after processing")
    args = parser.parse_args()

    if args.video:
        if not args.brand:
            brand = _infer_brand_from_filename(args.video, list(BRANDS_AND_SKUS.keys()))
            if brand == "UNKNOWN":
                print("Cannot infer brand from filename. Use --brand MEVIUS")
                sys.exit(1)
        else:
            brand = args.brand

        process_video(args.video, brand=brand, sku=args.sku,
                     output_dir=args.output, fps=args.fps, augment=not args.no_augment)

    elif args.dir:
        process_directory(args.dir, mapping_file=args.mapping,
                         output_dir=args.output, fps=args.fps, augment=not args.no_augment)

    else:
        print("Provide a video file or --dir folder")
        print("  python video_processor.py mevius.mp4 --brand MEVIUS")
        print("  python video_processor.py --dir videos/ --mapping mapping.json")
        sys.exit(1)

    if args.retrain:
        rebuild_classifier_with_video_frames(args.output)
