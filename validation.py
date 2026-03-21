"""
Input validation & API key verification.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def validate_excel_file(file_path: str) -> dict:
    """
    Validate an uploaded Excel file.
    Returns {"valid": True} or {"valid": False, "error": "..."}.
    """
    p = Path(file_path)

    if not p.exists():
        return {"valid": False, "error": f"File not found: {file_path}"}

    if p.suffix.lower() not in (".xlsx", ".xlsm"):
        return {"valid": False, "error": f"Invalid file type: {p.suffix}. Expected .xlsx or .xlsm"}

    if p.stat().st_size == 0:
        return {"valid": False, "error": "File is empty"}

    if p.stat().st_size > 50 * 1024 * 1024:  # 50MB
        return {"valid": False, "error": "File is too large (max 50MB)"}

    try:
        from openpyxl import load_workbook
        wb = load_workbook(file_path, read_only=True)
        sheet_names = wb.sheetnames

        # Check for Raw Data sheet
        raw_data_sheets = [s for s in sheet_names if "raw data" in s.lower()]
        if not raw_data_sheets:
            return {
                "valid": False,
                "error": f"No 'Raw Data' sheet found. Available sheets: {', '.join(sheet_names)}",
            }

        wb.close()
        return {"valid": True, "sheets": sheet_names}

    except Exception as e:
        return {"valid": False, "error": f"Cannot read Excel file: {e}"}


def validate_api_keys(model: str = "gemini-2.5-pro") -> dict:
    """
    Check if the required API key is set for the given model.
    Returns {"valid": True, "provider": "..."} or {"valid": False, "error": "..."}.
    """
    from image_analyzer import MODEL_REGISTRY, get_provider

    if model in MODEL_REGISTRY:
        provider = MODEL_REGISTRY[model][0]
    else:
        provider = get_provider(model)

    key_env_map = {
        "claude": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/"),
        "gemini": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
        "fireworks": ("FIREWORKS_API_KEY", "https://fireworks.ai/"),
    }

    env_var, url = key_env_map.get(provider, ("UNKNOWN", ""))
    api_key = os.getenv(env_var, "")

    if not api_key:
        return {
            "valid": False,
            "provider": provider,
            "error": f"{env_var} is not set. Get your key at {url}",
        }

    # Basic format check
    if provider == "claude" and not api_key.startswith("sk-ant-"):
        return {
            "valid": False,
            "provider": provider,
            "error": f"{env_var} doesn't look like a valid Anthropic key (should start with 'sk-ant-')",
        }

    if provider == "gemini" and not api_key.startswith("AIza"):
        return {
            "valid": False,
            "provider": provider,
            "error": f"{env_var} doesn't look like a valid Google AI key (should start with 'AIza')",
        }

    return {"valid": True, "provider": provider, "env_var": env_var}


def validate_image_url(url: str) -> bool:
    """Quick check if a string looks like a valid image URL."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith("http://") or url.startswith("https://")


def estimate_processing_time(
    num_images: int,
    model: str = "gemini-2.5-pro",
    enhancements_enabled: bool = True,
) -> dict:
    """
    Estimate processing time based on model RPM and image count.
    """
    from rate_limiter import RateLimiter

    limiter = RateLimiter(model)
    safe_workers = limiter.get_safe_workers()

    # Base time per image (API call)
    base_time_per_image = limiter.interval  # seconds between requests

    # Enhancement overhead per image
    enhancement_overhead = 0
    if enhancements_enabled:
        enhancement_overhead = 8  # ~8s extra for OCR + SKU refinement (2 extra API calls)

    # Total time with parallelism
    effective_time_per_image = base_time_per_image + enhancement_overhead
    total_sequential = num_images * effective_time_per_image
    total_parallel = total_sequential / safe_workers

    # Add overhead for fetching images
    fetch_overhead = num_images * 1.5  # ~1.5s per image fetch
    total_with_fetch = total_parallel + (fetch_overhead / safe_workers)

    minutes = total_with_fetch / 60

    return {
        "estimated_seconds": int(total_with_fetch),
        "estimated_minutes": round(minutes, 1),
        "display": f"~{int(minutes)}min" if minutes >= 1 else f"~{int(total_with_fetch)}sec",
        "workers": safe_workers,
        "rpm": limiter.rpm,
        "per_image_seconds": round(effective_time_per_image, 1),
    }
