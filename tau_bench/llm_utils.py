# Copyright Sierra

import time

from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

TRANSIENT_ERRORS = (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
    Timeout,
)


def completion_with_backoff(**kwargs):
    """litellm completion that survives Azure per-minute rate limits and
    transient connection drops.

    Azure GlobalStandard limits reset on a 60s window, so litellm's default
    short retry backoff can exhaust its attempts inside a single window.
    """
    kwargs.setdefault("timeout", 300)
    max_attempts = 14
    for attempt in range(max_attempts):
        try:
            return completion(**kwargs)
        except TRANSIENT_ERRORS:
            if attempt == max_attempts - 1:
                raise
            # ramp to 2 min between tries: sustained rate limiting under high
            # concurrency can last many minutes, and giving up mid-loop silently
            # discards a whole task's intervention
            time.sleep(min(120, 10 * (attempt + 1)))
