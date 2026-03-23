"""
Embedding Search — Visual similarity matching for cigarette pack SKU identification.

Signal 2 in the hybrid pipeline:
  1. Build a catalog of embeddings from 238 reference pack images
  2. For each detected crop, compute its embedding
  3. Compare against the catalog using cosine similarity
  4. Return top-k matches with scores

Handles rotated packs by testing 4 orientations (0, 90, 180, 270 degrees).
"""

import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Lazy-loaded model and catalog
_model = None
_transform = None
_catalog = None  # {"embeddings": np.array, "metadata": list[dict]}

CATALOG_PATH = "reference_images/embeddings.npz"
MAPPING_PATH = "reference_images/mapping.json"
REFERENCE_DIR = "reference_images"
EMBEDDING_DIM = 2048  # ResNet50 penultimate layer


def _get_model():
    """Lazy-load the ResNet50 feature extractor."""
    global _model, _transform
    if _model is not None:
        return _model, _transform

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet.eval()

    # Remove the final classification layer to get feature vectors
    # ResNet50 avgpool outputs (batch, 2048)
    import torch.nn as nn
    _model = nn.Sequential(*list(resnet.children())[:-1])  # everything except fc
    _model.eval()

    # Standard ImageNet preprocessing
    _transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    return _model, _transform


def _compute_embedding(pil_image: Image.Image) -> np.ndarray:
    """Compute a normalized embedding vector for a single PIL image."""
    import torch

    model, transform = _get_model()
    img = pil_image.convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = model(tensor)  # (1, 2048, 1, 1)

    embedding = features.squeeze().numpy()  # (2048,)
    # L2-normalize for cosine similarity via dot product
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def _compute_embedding_with_rotations(pil_image: Image.Image) -> np.ndarray:
    """
    Compute embeddings for 4 rotations of the image.
    Returns shape (4, EMBEDDING_DIM).
    """
    embeddings = []
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = pil_image
        else:
            rotated = pil_image.rotate(angle, expand=True)
        embeddings.append(_compute_embedding(rotated))
    return np.array(embeddings)


def build_catalog(reference_dir: str = REFERENCE_DIR, force: bool = False):
    """
    Build/rebuild the embedding catalog from reference images.

    Loads all images from reference_dir, computes embeddings using ResNet50,
    and saves to disk as embeddings.npz.
    """
    global _catalog

    catalog_path = os.path.join(reference_dir, "embeddings.npz")
    mapping_path = os.path.join(reference_dir, "mapping.json")

    # Check if cached catalog exists and is up-to-date
    if not force and os.path.exists(catalog_path):
        print(f"Loading cached catalog from {catalog_path}")
        _catalog = _load_catalog(catalog_path)
        print(f"Loaded {len(_catalog['metadata'])} reference embeddings")
        return _catalog

    # Load mapping
    with open(mapping_path) as f:
        mapping = json.load(f)

    print(f"Building embedding catalog from {len(mapping)} reference images...")
    start = time.time()

    embeddings = []
    metadata = []
    skipped = 0

    for filename, info in mapping.items():
        img_path = os.path.join(reference_dir, filename)
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {filename}")
            skipped += 1
            continue

        # Skip WDP (HD Photo) format — PIL can't open these
        if filename.lower().endswith(".wdp"):
            print(f"  SKIP (WDP format): {filename}")
            skipped += 1
            continue

        try:
            img = Image.open(img_path)
            img = img.convert("RGB")
            emb = _compute_embedding(img)
            embeddings.append(emb)
            metadata.append({
                "brand": info.get("brand", ""),
                "sku": info.get("sku", ""),
                "filename": filename,
            })
        except Exception as e:
            print(f"  SKIP ({e}): {filename}")
            skipped += 1
            continue

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Save to disk
    # We serialize metadata as JSON string inside the npz
    meta_json = json.dumps(metadata)
    np.savez_compressed(
        catalog_path,
        embeddings=embeddings_array,
        metadata=np.array([meta_json]),
    )

    elapsed = time.time() - start
    print(f"Built catalog: {len(metadata)} images, {skipped} skipped, {elapsed:.1f}s")
    print(f"Saved to {catalog_path}")

    _catalog = {"embeddings": embeddings_array, "metadata": metadata}
    return _catalog


def _load_catalog(catalog_path: str = CATALOG_PATH) -> dict:
    """Load a previously-built catalog from disk."""
    data = np.load(catalog_path, allow_pickle=False)
    embeddings = data["embeddings"]
    meta_json = str(data["metadata"][0])
    metadata = json.loads(meta_json)
    return {"embeddings": embeddings, "metadata": metadata}


def _ensure_catalog():
    """Make sure the catalog is loaded (from cache or by building it)."""
    global _catalog
    if _catalog is not None:
        return _catalog

    if os.path.exists(CATALOG_PATH):
        _catalog = _load_catalog(CATALOG_PATH)
    else:
        _catalog = build_catalog()

    return _catalog


def find_matching_sku(crop_image_data: bytes, top_k: int = 5) -> list[dict]:
    """
    Match a cropped cigarette pack against the reference catalog.

    Takes raw image bytes (JPEG/PNG), computes embeddings for 4 rotations,
    and returns the top-k most similar reference images.

    Returns:
        [{"brand": "ARA", "sku": "ARA RED", "similarity": 0.92, "reference_image": "image55.jpeg"}, ...]
    """
    catalog = _ensure_catalog()
    ref_embeddings = catalog["embeddings"]  # (N, 2048)
    ref_metadata = catalog["metadata"]

    # Load the query image
    img = Image.open(io.BytesIO(crop_image_data)).convert("RGB")

    # Compute embeddings for 4 rotations
    query_embeddings = _compute_embedding_with_rotations(img)  # (4, 2048)

    # Compute similarities: (4, N) = (4, 2048) @ (2048, N)
    similarities = query_embeddings @ ref_embeddings.T  # cosine sim (unit vectors)

    # For each reference image, take the BEST similarity across rotations
    best_sims = similarities.max(axis=0)  # (N,)

    # Get top-k indices
    if top_k >= len(best_sims):
        top_indices = np.argsort(best_sims)[::-1]
    else:
        # Use argpartition for efficiency (not that it matters for 238 items)
        top_indices = np.argpartition(best_sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(best_sims[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        meta = ref_metadata[idx]
        results.append({
            "brand": meta["brand"],
            "sku": meta["sku"],
            "similarity": float(best_sims[idx]),
            "reference_image": meta["filename"],
        })

    return results


def find_matching_sku_single(crop_image_data: bytes, top_k: int = 5) -> list[dict]:
    """
    Like find_matching_sku but without rotation handling (faster).
    Use when you know the crop is roughly upright.
    """
    catalog = _ensure_catalog()
    ref_embeddings = catalog["embeddings"]
    ref_metadata = catalog["metadata"]

    img = Image.open(io.BytesIO(crop_image_data)).convert("RGB")
    query_emb = _compute_embedding(img)  # (2048,)

    # Cosine similarity via dot product (already normalized)
    sims = ref_embeddings @ query_emb  # (N,)

    if top_k >= len(sims):
        top_indices = np.argsort(sims)[::-1]
    else:
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        meta = ref_metadata[idx]
        results.append({
            "brand": meta["brand"],
            "sku": meta["sku"],
            "similarity": float(sims[idx]),
            "reference_image": meta["filename"],
        })

    return results


# ── CLI for building/testing ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        force = "--force" in sys.argv
        build_catalog(force=force)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with a reference image — should match itself with ~1.0 similarity
        test_image = sys.argv[2] if len(sys.argv) > 2 else "reference_images/image2.jpeg"
        print(f"\nTesting with: {test_image}")
        with open(test_image, "rb") as f:
            crop_data = f.read()
        results = find_matching_sku(crop_data, top_k=5)
        print("\nTop 5 matches:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['sku']:40s} sim={r['similarity']:.4f}  ({r['reference_image']})")
    else:
        print("Usage:")
        print("  python embedding_search.py build [--force]   Build the reference catalog")
        print("  python embedding_search.py test [image_path] Test matching against catalog")
