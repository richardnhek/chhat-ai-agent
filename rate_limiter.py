"""
Rate limiter — prevents API rate limit errors across all providers.
Uses a token bucket algorithm per provider.
"""

import time
import threading


# Requests per minute limits per provider/model
# Set GEMINI_PAID_TIER=true in .env to use paid tier limits
import os

_gemini_paid = os.getenv("GEMINI_PAID_TIER", "").lower() in ("true", "1", "yes")

RATE_LIMITS = {
    # Gemini — paid tier (Tier 1) vs free tier
    "gemini-2.5-pro": 360 if _gemini_paid else 5,
    "gemini-2.5-flash": 1000 if _gemini_paid else 10,
    "gemini-3.1-pro": 50 if _gemini_paid else 5,
    "gemini-3-pro": 50 if _gemini_paid else 5,
    "gemini-3-flash": 200 if _gemini_paid else 10,
    # Claude (Tier 1)
    "claude-sonnet-4-6": 50,
    "claude-haiku-4-5": 50,
    "claude-opus-4-6": 50,
    # Fireworks
    "qwen2.5-vl-72b": 30,
    "qwen2.5-vl-7b": 60,
}

# Default if model not in the list
DEFAULT_RPM = 10


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, model: str):
        self.rpm = RATE_LIMITS.get(model, DEFAULT_RPM)
        self.interval = 60.0 / self.rpm  # Seconds between requests
        self.lock = threading.Lock()
        self.last_request_time = 0.0
        self.model = model

    def wait(self):
        """Block until it's safe to make the next request."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def get_safe_workers(self) -> int:
        """Return the max number of parallel workers that won't exceed rate limits."""
        # With N workers and interval I, effective RPM = N * (60/I) = N * RPM
        # We want N * RPM_per_worker <= RPM_limit
        # But each worker waits on the shared limiter, so workers is bounded by RPM
        # Practically: allow up to RPM/3 workers (leaving headroom)
        return max(1, self.rpm // 3)

    def __repr__(self):
        return f"RateLimiter(model={self.model}, rpm={self.rpm}, interval={self.interval:.1f}s)"
