"""
Image analyzer — fetches images from URLs and analyzes them using multiple AI providers.
Supports: Claude (Anthropic), Gemini (Google), Qwen2.5-VL (via Fireworks AI).
"""

import base64
import io
import re
import json
from urllib.parse import urlparse

import requests
from PIL import Image

from brands import get_brand_list_for_prompt


# Supported image MIME types
SUPPORTED_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Max image size before base64 encoding (3.5MB raw → ~4.7MB base64, under 5MB API limit)
MAX_IMAGE_SIZE = 3_500_000

_BRAND_LIST = get_brand_list_for_prompt()

ANALYSIS_PROMPT = f"""You are an expert tobacco product auditor for the Cambodian market. You analyze real-world photos of store display cases to identify which cigarette BRANDS are physically present as actual cigarette boxes/packs.

IMPORTANT RULES:
- ONLY identify brands where you can see an actual physical cigarette BOX or PACK.
- Do NOT count brand names printed on shelf labels, price tags, signs, posters, or advertisements — only actual product boxes.
- Glass reflections, glare, and dirty glass are common — do not let reflections trick you into identifying brands that aren't physically there.
- Some packs may be upside-down or sideways — still count them if you can identify the brand.
- Ignore all non-cigarette products (lighters, candy, razors, drinks, etc.)

Here is the OFFICIAL brand list. You MUST match brands to this list. Use EXACTLY these brand names:
{_BRAND_LIST}

For each brand you identify, also try to identify the specific SKU variant if possible (e.g. "ARA RED" vs just "ARA").

Respond in this EXACT JSON format (no markdown, just raw JSON):
{{
    "brands_found": [
        {{"brand": "<EXACT brand name from list>", "skus": ["<SKU1>", "<SKU2>"], "notes": "<brief description of what you see>"}},
    ],
    "brand_count": <number of distinct brands found>,
    "confidence": "<high|medium|low>",
    "notes": "<any visibility issues or observations>"
}}

If NO cigarette boxes are visible, respond with:
{{
    "brands_found": [],
    "brand_count": 0,
    "confidence": "high",
    "notes": "<what the image shows>"
}}

Be methodical: scan left-to-right, top-to-bottom. Only report brands with physical boxes present.
"""

# ── Provider registry ────────────────────────────────────────────────────────

# Maps user-friendly names to (provider, model_id)
MODEL_REGISTRY = {
    # Claude models
    "claude-sonnet-4-6": ("claude", "claude-sonnet-4-6"),
    "claude-haiku-4-5": ("claude", "claude-haiku-4-5-20251001"),
    "claude-opus-4-6": ("claude", "claude-opus-4-6"),
    # Gemini models
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-2.5-pro": ("gemini", "gemini-2.5-pro"),
    "gemini-2.0-flash": ("gemini", "gemini-2.0-flash-001"),
    # Open-source via Fireworks AI
    "qwen2.5-vl-72b": ("fireworks", "accounts/fireworks/models/qwen2p5-vl-72b-instruct"),
    "qwen2.5-vl-7b": ("fireworks", "accounts/fireworks/models/qwen2p5-vl-7b-instruct"),
}


def get_available_models() -> list[str]:
    """Return list of model names."""
    return list(MODEL_REGISTRY.keys())


def get_provider(model_name: str) -> str:
    """Return the provider for a given model name."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name][0]
    # Fallback: guess from name
    if "gemini" in model_name:
        return "gemini"
    if "qwen" in model_name:
        return "fireworks"
    return "claude"


# ── Image fetching ───────────────────────────────────────────────────────────

def fetch_image(url: str, timeout: int = 30) -> tuple[bytes, str]:
    """Fetch an image from a URL and return (image_bytes, media_type)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": urlparse(url).scheme + "://" + urlparse(url).netloc + "/",
    }

    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "jpeg" in content_type or "jpg" in content_type:
        media_type = "image/jpeg"
    elif "png" in content_type:
        media_type = "image/png"
    elif "gif" in content_type:
        media_type = "image/gif"
    elif "webp" in content_type:
        media_type = "image/webp"
    else:
        ext = _get_extension(url)
        media_type = SUPPORTED_TYPES.get(ext, "image/jpeg")

    image_data = response.content
    image_data, media_type = _resize_image(image_data, media_type)
    return image_data, media_type


def _get_extension(url: str) -> str:
    path = urlparse(url).path
    ext = ""
    for known_ext in SUPPORTED_TYPES:
        if known_ext in path.lower():
            ext = known_ext
            break
    return ext


def _resize_image(image_data: bytes, media_type: str) -> tuple[bytes, str]:
    """Resize image to fit within API size limits."""
    img = Image.open(io.BytesIO(image_data))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    max_dim = 1600
    quality = 80

    while max_dim >= 400:
        resized = img.copy()
        resized.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=quality, optimize=True)
        result_data = buffer.getvalue()
        if len(result_data) <= MAX_IMAGE_SIZE:
            return result_data, "image/jpeg"
        max_dim -= 400
        quality = max(50, quality - 10)

    img.thumbnail((400, 400), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=50, optimize=True)
    return buffer.getvalue(), "image/jpeg"


# ── Response parsing ─────────────────────────────────────────────────────────

def _parse_response(response_text: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    try:
        # Strip markdown code fences
        cleaned = response_text
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting the outermost JSON object
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            return json.loads(json_match.group())

        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "brands_found": [],
            "brand_count": 0,
            "confidence": "low",
            "notes": f"Could not parse response: {response_text[:300]}",
        }


# ── Claude (Anthropic) ──────────────────────────────────────────────────────

def _analyze_claude(image_data: bytes, media_type: str, model: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    b64_image = base64.b64encode(image_data).decode("utf-8")

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_image}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
    )
    return _parse_response(message.content[0].text.strip())


# ── Gemini (Google) ──────────────────────────────────────────────────────────

def _analyze_gemini(image_data: bytes, media_type: str, model: str, api_key: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type=media_type),
            ANALYSIS_PROMPT,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.1,
        ),
    )
    return _parse_response(response.text.strip())


# ── Qwen2.5-VL via Fireworks AI (OpenAI-compatible) ─────────────────────────

def _analyze_fireworks(image_data: bytes, media_type: str, model: str, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    b64_image = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{media_type};base64,{b64_image}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
    )
    return _parse_response(response.choices[0].message.content.strip())


# ── Unified interface ────────────────────────────────────────────────────────

_PROVIDER_FNS = {
    "claude": _analyze_claude,
    "gemini": _analyze_gemini,
    "fireworks": _analyze_fireworks,
}


def analyze_image(
    image_data: bytes,
    media_type: str,
    model: str = "claude-sonnet-4-6",
    api_keys: dict[str, str] | None = None,
    # Legacy support: accept a client object and ignore it
    client=None,
) -> dict:
    """
    Analyze an image using any supported provider.

    Args:
        image_data: Raw image bytes
        media_type: MIME type
        model: Model name from MODEL_REGISTRY or a raw model ID
        api_keys: Dict of provider -> API key, e.g. {"claude": "sk-...", "gemini": "AIza..."}
        client: (Legacy) Ignored, kept for backward compatibility
    """
    # Resolve provider and model ID
    if model in MODEL_REGISTRY:
        provider, model_id = MODEL_REGISTRY[model]
    else:
        provider = get_provider(model)
        model_id = model

    # Get API key
    if api_keys and provider in api_keys:
        api_key = api_keys[provider]
    else:
        import os
        key_env_map = {
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
        }
        api_key = os.getenv(key_env_map.get(provider, ""))
        if not api_key:
            return {"error": f"No API key for provider '{provider}'. Set {key_env_map.get(provider, 'UNKNOWN')} in .env"}

    # Call provider
    fn = _PROVIDER_FNS.get(provider)
    if not fn:
        return {"error": f"Unknown provider: {provider}"}

    try:
        return fn(image_data, media_type, model_id, api_key)
    except Exception as e:
        return {"error": f"{provider} API error: {e}"}


def analyze_url(url: str, model: str = "claude-sonnet-4-6", api_keys: dict | None = None) -> dict:
    """Full pipeline: fetch image from URL → analyze."""
    try:
        image_data, media_type = fetch_image(url)
        return analyze_image(image_data, media_type, model=model, api_keys=api_keys)
    except requests.exceptions.Timeout:
        return {"error": f"Timeout fetching image from {url}"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error fetching image: {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching image: {e}"}
