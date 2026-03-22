"""
Disk-based image cache — avoids re-fetching the same Ipsos URLs.

Cache key = SHA256 hash of the URL.
Cache directory: image_cache/
Max cache size: 1GB (LRU eviction).
"""

import hashlib
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path("image_cache")
MAX_CACHE_SIZE_BYTES = 1_073_741_824  # 1 GB


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def _url_to_key(url: str) -> str:
    """Convert a URL to a SHA256 cache key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _data_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.img"


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta"


def get_cached_image(url: str) -> tuple[bytes, str] | None:
    """
    Return (image_data, media_type) from cache, or None if not cached.
    Updates access time on hit (for LRU).
    """
    key = _url_to_key(url)
    dp = _data_path(key)
    mp = _meta_path(key)

    if not dp.exists() or not mp.exists():
        return None

    try:
        with open(mp, "r") as f:
            meta = json.load(f)
        media_type = meta.get("media_type", "image/jpeg")

        with open(dp, "rb") as f:
            image_data = f.read()

        # Touch files to update access time (LRU)
        now = time.time()
        os.utime(dp, (now, now))
        os.utime(mp, (now, now))

        return image_data, media_type
    except (IOError, json.JSONDecodeError, OSError):
        return None


def cache_image(url: str, image_data: bytes, media_type: str):
    """Save image data and metadata to the cache. Evicts old entries if over size limit."""
    _ensure_cache_dir()
    key = _url_to_key(url)

    # Write data
    with open(_data_path(key), "wb") as f:
        f.write(image_data)

    # Write metadata
    meta = {
        "url": url,
        "media_type": media_type,
        "size": len(image_data),
        "cached_at": time.time(),
    }
    with open(_meta_path(key), "w") as f:
        json.dump(meta, f)

    # Enforce max cache size
    _evict_if_needed()


def _evict_if_needed():
    """Delete oldest files (by modification time) until cache is under the size limit."""
    if not CACHE_DIR.exists():
        return

    # Gather all .img files with their sizes and mtimes
    entries = []
    total_size = 0
    for p in CACHE_DIR.iterdir():
        if p.suffix == ".img":
            try:
                stat = p.stat()
                entries.append((stat.st_mtime, stat.st_size, p))
                total_size += stat.st_size
            except OSError:
                continue

    if total_size <= MAX_CACHE_SIZE_BYTES:
        return

    # Sort by mtime ascending (oldest first)
    entries.sort(key=lambda e: e[0])

    for mtime, size, path in entries:
        if total_size <= MAX_CACHE_SIZE_BYTES:
            break
        key = path.stem
        try:
            path.unlink(missing_ok=True)
            _meta_path(key).unlink(missing_ok=True)
            total_size -= size
        except OSError:
            continue


def get_cache_stats() -> dict:
    """Return cache statistics: total_images and cache_size_mb."""
    if not CACHE_DIR.exists():
        return {"total_images": 0, "cache_size_mb": 0.0}

    total_images = 0
    total_bytes = 0
    for p in CACHE_DIR.iterdir():
        if p.suffix == ".img":
            total_images += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                continue

    return {
        "total_images": total_images,
        "cache_size_mb": round(total_bytes / (1024 * 1024), 2),
    }


def clear_cache():
    """Delete all cached images."""
    if not CACHE_DIR.exists():
        return

    for p in CACHE_DIR.iterdir():
        try:
            p.unlink()
        except OSError:
            continue
