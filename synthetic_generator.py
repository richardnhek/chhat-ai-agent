"""
Synthetic Training Data Generator for CHHAT Cigarette Brand Detection.

Composites reference pack images onto shelf/display backgrounds with realistic
augmentations and outputs COCO-format annotations for RF-DETR training.

Usage:
    python synthetic_generator.py --count 1000 --output training_data/synthetic/
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "reference_images"
MAPPING_FILE = REF_DIR / "mapping.json"

IMG_W, IMG_H = 1280, 960

# Shelf background colours (BGR for OpenCV, but we work in RGB via PIL)
SHELF_COLORS = [
    (139, 90, 43),    # wood brown
    (200, 170, 130),  # light wood
    (240, 240, 240),  # white shelf
    (200, 200, 200),  # gray shelf
    (180, 195, 210),  # blue-gray (glass cabinet)
    (60, 60, 70),     # dark cabinet
    (160, 140, 120),  # beige
    (100, 110, 130),  # steel blue-gray
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_reference_images(mapping_path: Path, ref_dir: Path):
    """Load all reference images and their metadata from mapping.json."""
    with open(mapping_path) as f:
        mapping = json.load(f)

    refs = []
    for filename, meta in mapping.items():
        img_path = ref_dir / filename
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGBA")
            refs.append({
                "image": img,
                "filename": filename,
                "brand": meta["brand"],
                "sku": meta["sku"],
            })
        except Exception:
            continue

    return refs


def build_category_map(refs: list[dict]) -> tuple[dict, list[dict]]:
    """Build COCO category list from reference metadata.

    Returns:
        name_to_id: dict mapping sku_name -> category_id
        categories: list of COCO category dicts
    """
    skus = sorted(set(r["sku"] for r in refs))
    name_to_id = {}
    categories = []
    for i, sku in enumerate(skus, start=1):
        name_to_id[sku] = i
        # find brand for this sku
        brand = next((r["brand"] for r in refs if r["sku"] == sku), "UNKNOWN")
        categories.append({
            "id": i,
            "name": sku,
            "supercategory": brand,
        })
    return name_to_id, categories


# ── Background generators ────────────────────────────────────────────────────

def gen_solid_bg(w: int, h: int) -> Image.Image:
    """Create a solid-colored shelf background."""
    color = random.choice(SHELF_COLORS)
    img = Image.new("RGB", (w, h), color)
    # Add subtle noise
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-8, 9, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gen_gradient_bg(w: int, h: int) -> Image.Image:
    """Create a vertical gradient background (simulates lighting variation)."""
    c1 = random.choice(SHELF_COLORS)
    c2 = random.choice(SHELF_COLORS)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        arr[y, :] = [int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)]
    # Add noise
    noise = np.random.randint(-6, 7, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gen_dirty_glass_bg(w: int, h: int) -> Image.Image:
    """Create a noisy blurred background simulating dirty glass."""
    base_color = random.choice(SHELF_COLORS)
    arr = np.full((h, w, 3), base_color, dtype=np.uint8)
    # Heavy noise
    noise = np.random.randint(-30, 31, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img


def gen_shelf_lines_bg(w: int, h: int) -> Image.Image:
    """Create a background with horizontal shelf lines."""
    img = gen_solid_bg(w, h)
    draw = ImageDraw.Draw(img)
    n_shelves = random.randint(2, 4)
    for i in range(1, n_shelves + 1):
        y = int(h * i / (n_shelves + 1))
        y += random.randint(-15, 15)
        shade = random.randint(30, 80)
        line_color = tuple(max(0, c - shade) for c in random.choice(SHELF_COLORS))
        draw.line([(0, y), (w, y)], fill=line_color, width=random.randint(2, 5))
    return img


def random_background(w: int, h: int) -> Image.Image:
    """Pick a random background generator."""
    gen = random.choice([gen_solid_bg, gen_gradient_bg, gen_dirty_glass_bg, gen_shelf_lines_bg])
    return gen(w, h)


# ── Pack augmentation ────────────────────────────────────────────────────────

def augment_pack(pack_img: Image.Image, target_h: int) -> Image.Image:
    """Apply random augmentations to a single pack image."""
    img = pack_img.copy()

    # 1. Random rotation (cardinal directions mostly, with slight tilt)
    cardinal = random.choice([0, 0, 0, 90, 180, 270])  # bias toward upright
    tilt = random.uniform(-5, 5)
    angle = cardinal + tilt
    if angle != 0:
        img = img.rotate(-angle, resample=Image.BICUBIC, expand=True)

    # 2. Random resize (scale relative to target height)
    scale = random.uniform(0.3, 0.8)
    new_h = int(target_h * scale)
    aspect = img.width / img.height if img.height > 0 else 1
    new_w = int(new_h * aspect)
    new_w = max(new_w, 20)
    new_h = max(new_h, 20)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 3. Brightness/contrast variation
    if random.random() < 0.7:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
    if random.random() < 0.7:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)

    # 4. Slight color jitter
    if random.random() < 0.4:
        factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)

    # 5. Gaussian blur (dirty glass simulation)
    if random.random() < 0.5:
        sigma = random.uniform(0.0, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 6. Perspective warp (slight)
    if random.random() < 0.3:
        img = _perspective_warp(img)

    return img


def _perspective_warp(img: Image.Image) -> Image.Image:
    """Apply a slight random perspective transform using OpenCV."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return img

    # Small random perturbation of corners
    d = int(min(w, h) * 0.08)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


# ── Placement ────────────────────────────────────────────────────────────────

def place_packs_grid(bg: Image.Image, packs: list[Image.Image]) -> list[tuple[int, int, int, int]]:
    """Place pack images on background in a grid-like shelf layout.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    n = len(packs)
    bw, bh = bg.size

    # Determine grid layout
    cols = min(n, random.randint(3, 6))
    rows = math.ceil(n / cols)

    cell_w = bw // cols
    cell_h = bh // rows

    boxes = []
    for idx, pack in enumerate(packs):
        r = idx // cols
        c = idx % cols

        pw, ph = pack.size

        # Center in cell with random offset
        cx = c * cell_w + cell_w // 2
        cy = r * cell_h + cell_h // 2

        # Random offset (up to 15% of cell size)
        ox = random.randint(-int(cell_w * 0.15), int(cell_w * 0.15))
        oy = random.randint(-int(cell_h * 0.15), int(cell_h * 0.15))

        x = cx - pw // 2 + ox
        y = cy - ph // 2 + oy

        # Clamp to image bounds
        x = max(0, min(x, bw - pw))
        y = max(0, min(y, bh - ph))

        # Paste (handle RGBA transparency)
        if pack.mode == "RGBA":
            bg.paste(pack, (x, y), pack)
        else:
            bg.paste(pack, (x, y))

        boxes.append((x, y, x + pw, y + ph))

    return boxes


# ── Main generation ──────────────────────────────────────────────────────────

def generate_dataset(
    count: int,
    output_dir: str,
    packs_per_image: tuple[int, int] = (3, 8),
    progress_callback=None,
):
    """Generate synthetic training images with COCO annotations.

    Args:
        count: number of images to generate
        output_dir: output directory path
        packs_per_image: (min, max) packs per image
        progress_callback: optional callable(current, total) for progress reporting

    Returns:
        Path to the annotations.json file
    """
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load references
    refs = load_reference_images(MAPPING_FILE, REF_DIR)
    if not refs:
        raise RuntimeError(f"No reference images found in {REF_DIR} with mapping {MAPPING_FILE}")

    name_to_id, categories = build_category_map(refs)

    # COCO structure
    coco = {
        "info": {
            "description": "CHHAT Synthetic Cigarette Pack Training Data",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    ann_id = 1

    for img_idx in range(count):
        # Pick random number of packs
        n_packs = random.randint(*packs_per_image)
        chosen = random.choices(refs, k=n_packs)

        # Generate background
        bg = random_background(IMG_W, IMG_H)

        # Augment each pack
        augmented = []
        for ref in chosen:
            aug = augment_pack(ref["image"], IMG_H)
            augmented.append(aug)

        # Place packs on background
        boxes = place_packs_grid(bg, augmented)

        # Save image
        fname = f"syn_{img_idx:06d}.jpg"
        fpath = img_dir / fname
        bg.save(fpath, "JPEG", quality=92)

        # Record COCO image entry
        coco["images"].append({
            "id": img_idx + 1,
            "file_name": fname,
            "width": IMG_W,
            "height": IMG_H,
        })

        # Record annotations
        for box, ref in zip(boxes, chosen):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w < 5 or h < 5:
                continue

            cat_id = name_to_id.get(ref["sku"], 1)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_idx + 1,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],  # COCO format: [x, y, width, height]
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

        if progress_callback:
            progress_callback(img_idx + 1, count)

    # Save annotations
    ann_path = out / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    return str(ann_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for CHHAT cigarette pack detection."
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Number of synthetic images to generate (default: 1000)",
    )
    parser.add_argument(
        "--output", type=str, default="training_data/synthetic/",
        help="Output directory (default: training_data/synthetic/)",
    )
    parser.add_argument(
        "--min-packs", type=int, default=3,
        help="Minimum packs per image (default: 3)",
    )
    parser.add_argument(
        "--max-packs", type=int, default=8,
        help="Maximum packs per image (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Generating {args.count} synthetic images...")
    print(f"Output: {args.output}")

    t0 = time.time()

    def progress(current, total):
        pct = current / total * 100
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}% ({current}/{total})")
        sys.stdout.flush()

    ann_path = generate_dataset(
        count=args.count,
        output_dir=args.output,
        packs_per_image=(args.min_packs, args.max_packs),
        progress_callback=progress,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Annotations: {ann_path}")

    # Print summary
    with open(ann_path) as f:
        coco = json.load(f)
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")


if __name__ == "__main__":
    main()
