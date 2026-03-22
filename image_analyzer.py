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
from retry import retry_with_backoff
from logger import get_logger
from image_cache import get_cached_image, cache_image


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

ANALYSIS_PROMPT = f"""You are an expert tobacco product auditor for the Cambodian market. You analyze real-world photos of store display cases to identify which cigarette BRANDS and SKUs are physically present.

RULES FOR BRAND IDENTIFICATION:
- ONLY identify brands where you can see an actual physical cigarette BOX or PACK.
- Do NOT count brand names printed on shelf labels, price tags, signs, posters, or advertisements — only actual product boxes.
- Glass reflections, glare, and dirty glass are common — do not let reflections trick you into identifying brands that aren't physically there.
- Some packs may be upside-down or sideways — still count them if you can identify the brand.
- Ignore all non-cigarette products (lighters, candy, razors, drinks, etc.)

RULES FOR SKU IDENTIFICATION — THIS IS CRITICAL:
- For EACH brand you identify, you MUST determine the specific SKU variant. Do NOT just say the brand name — identify the exact SKU.
- SKU clues: pack COLOR (red, blue, gold, green, black, purple), text on the pack (e.g. "MENTHOL", "LIGHTS", "CHANGE", "ORIGINAL"), pack SIZE (slim, compact, king size, hard pack), and any capsule/option indicators.
- Common SKU patterns in Cambodia:
  - Red pack = usually "RED" or "HARD PACK" or "FF" (full flavor)
  - Blue pack = usually "LIGHTS" or "BLUE" or "MENTHOL"
  - Gold pack = usually "GOLD"
  - Green pack = usually "MENTHOL"
  - Black/dark pack = usually "ORIGINAL" or premium variant
  - Slim/thin pack = "SLIMS" or "SUPER SLIMS"
- If you can see the pack but cannot determine the exact SKU, use the brand name + "OTHERS" (e.g. "ESSE OTHERS")

RULES FOR BLURRY / PARTIALLY VISIBLE BOXES:
- If you see shapes that look like cigarette boxes but are TOO BLURRY to identify the brand, report them in "unidentified_packs".
- If you can partially identify a blurry box (e.g. you can tell it's red but can't read the text), note it as your best guess with low confidence.
- The presence of blurry/unidentifiable boxes should LOWER your overall confidence level.
- Do NOT simply ignore boxes you can't read — acknowledge them.

CONFIDENCE RULES:
- "high" = all boxes are clearly visible, text is readable, you are very sure of all identifications
- "medium" = most boxes are clear but some are partially obscured, angled, or behind dirty glass
- "low" = significant portion of boxes are blurry, obscured, or hard to identify; there are unidentified packs

Here is the OFFICIAL brand and SKU list. Match brands and SKUs EXACTLY to this list:
{_BRAND_LIST}

Respond in this EXACT JSON format (no markdown, just raw JSON):
{{
    "brands_found": [
        {{"brand": "<EXACT brand name from list>", "skus": ["<EXACT SKU name from list>"], "notes": "<describe what you see: pack color, text, position>"}},
    ],
    "brand_count": <number of distinct brands found>,
    "unidentified_packs": <number of cigarette-shaped boxes you can see but cannot identify>,
    "confidence": "<high|medium|low>",
    "notes": "<visibility issues, blurry areas, anything that affected your analysis>"
}}

If NO cigarette boxes are visible, respond with:
{{
    "brands_found": [],
    "brand_count": 0,
    "unidentified_packs": 0,
    "confidence": "high",
    "notes": "<what the image shows>"
}}

Be methodical: scan left-to-right, top-to-bottom. Report ALL cigarette boxes you see, even blurry ones.
"""

# ── Provider registry ────────────────────────────────────────────────────────

# Maps user-friendly names to (provider, model_id)
MODEL_REGISTRY = {
    # Gemini models (recommended — best accuracy-to-throughput ratio)
    "gemini-2.5-pro": ("gemini", "gemini-2.5-pro"),       # Default: best balance of accuracy + heavy load
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),    # Fastest + cheapest, slightly less accurate
    "gemini-3.1-pro": ("gemini", "gemini-3.1-pro-preview"),# Highest accuracy but preview/unstable
    "gemini-3-pro": ("gemini", "gemini-3-pro-preview"),
    "gemini-3-flash": ("gemini", "gemini-3-flash-preview"),
    # Claude models
    "claude-sonnet-4-6": ("claude", "claude-sonnet-4-6"),
    "claude-haiku-4-5": ("claude", "claude-haiku-4-5-20251001"),
    "claude-opus-4-6": ("claude", "claude-opus-4-6"),
    # Open-source via Fireworks AI (fine-tunable)
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

def fetch_image(url: str, timeout: int = 30, use_cache: bool = True) -> tuple[bytes, str]:
    """Fetch an image from a URL and return (image_bytes, media_type).

    Args:
        url: Image URL to fetch.
        timeout: HTTP request timeout in seconds.
        use_cache: If True (default), check disk cache before fetching and cache after fetch.
    """
    # Check cache first
    if use_cache:
        cached = get_cached_image(url)
        if cached is not None:
            return cached

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

    # Cache after successful fetch
    if use_cache:
        cache_image(url, image_data, media_type)

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

@retry_with_backoff(max_retries=3, base_delay=2.0)
def _analyze_claude(image_data: bytes, media_type: str, model: str, api_key: str, prompt: str = "") -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    b64_image = base64.b64encode(image_data).decode("utf-8")

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_image}},
                {"type": "text", "text": prompt or ANALYSIS_PROMPT},
            ],
        }],
    )
    return _parse_response(message.content[0].text.strip())


# ── Gemini (Google) ──────────────────────────────────────────────────────────

@retry_with_backoff(max_retries=3, base_delay=2.0)
def _analyze_gemini(image_data: bytes, media_type: str, model: str, api_key: str, prompt: str = "") -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type=media_type),
            prompt or ANALYSIS_PROMPT,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.1,
        ),
    )
    return _parse_response(response.text.strip())


# ── Qwen2.5-VL via Fireworks AI (OpenAI-compatible) ─────────────────────────

@retry_with_backoff(max_retries=3, base_delay=2.0)
def _analyze_fireworks(image_data: bytes, media_type: str, model: str, api_key: str, prompt: str = "") -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    b64_image = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{media_type};base64,{b64_image}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt or ANALYSIS_PROMPT},
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
    model: str = "gemini-2.5-pro",
    api_keys: dict[str, str] | None = None,
    correction_context: str = "",
    # Legacy support: accept a client object and ignore it
    client=None,
) -> dict:
    """
    Analyze an image using any supported provider.

    Args:
        image_data: Raw image bytes
        media_type: MIME type
        model: Model name from MODEL_REGISTRY or a raw model ID
        api_keys: Dict of provider -> API key
        correction_context: Past corrections formatted as few-shot examples
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

    # Build prompt with correction context
    prompt = ANALYSIS_PROMPT
    if correction_context:
        prompt = prompt + "\n\n" + correction_context

    # Call provider
    fn = _PROVIDER_FNS.get(provider)
    if not fn:
        return {"error": f"Unknown provider: {provider}"}

    try:
        return fn(image_data, media_type, model_id, api_key, prompt=prompt)
    except Exception as e:
        return {"error": f"{provider} API error: {e}"}


def analyze_url(url: str, model: str = "gemini-2.5-pro", api_keys: dict | None = None) -> dict:
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
