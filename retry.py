"""
Retry logic with exponential backoff for API calls.
"""

import time
import logging
from functools import wraps

logger = logging.getLogger("chhat")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retryable_errors: tuple = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        retryable_errors: Tuple of exception types to retry on
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Don't retry on auth errors or invalid requests
                    if any(s in error_str for s in ["invalid_api_key", "unauthorized", "403", "401"]):
                        logger.error(f"{func.__name__} auth error (not retrying): {e}")
                        raise

                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        # Add jitter to avoid thundering herd
                        import random
                        delay += random.uniform(0, delay * 0.1)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")

            raise last_exception

        return wrapper
    return decorator
