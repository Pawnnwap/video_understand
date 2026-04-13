"""utils/retry.py
Shared retry logic and frame compression utilities used across all stages.

Retry strategy:
  - Exponential backoff with jitter
  - Separate budgets for transient errors vs hard failures
  - Every retryable call logs attempt number and wait time
  - Sync variants for pipeline phases

Frame compression:
  - Caps image dimensions before encoding to base64
  - Estimates token cost and downsizes further if needed
  - Keeps aspect ratio; never upscales
"""

from __future__ import annotations

import functools
import io
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────────────────────
#  Retry configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetryConfig:
    """Centralised retry knobs — override per call site if needed."""

    max_attempts: int = 4
    base_delay_s: float = 2.0      # first back-off wait
    max_delay_s: float = 30.0     # ceiling on any single wait
    jitter_factor: float = 0.25     # ± fraction of computed delay
    # Exceptions that warrant a retry (everything else is re-raised immediately)
    retryable: tuple = field(default_factory=lambda: (
        ConnectionError,
        TimeoutError,
        OSError,
    ))


DEFAULT_RETRY = RetryConfig()


def _wait_time(attempt: int, cfg: RetryConfig) -> float:
    """Exponential backoff with bounded jitter."""
    delay = min(cfg.base_delay_s * (2 ** attempt), cfg.max_delay_s)
    jitter = delay * cfg.jitter_factor * (random.random() * 2 - 1)
    return max(0.1, delay + jitter)


# ─────────────────────────────────────────────────────────────────────────────
#  Synchronous retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def with_retry(
    func: Callable | None = None,
    *,
    cfg: RetryConfig = DEFAULT_RETRY,
    label: str = "",
):
    """Decorator for synchronous functions.

    Usage:
        @with_retry
        def call_vlm(...): ...

        @with_retry(cfg=RetryConfig(max_attempts=6))
        def fragile_call(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        name = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(cfg.max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    # Check if it's a known retryable type OR an HTTP 5xx / rate-limit
                    if not _is_retryable(exc, cfg):
                        log.error(f"[{name}] Non-retryable error: {exc}")
                        raise
                    wait = _wait_time(attempt, cfg)
                    log.warning(
                        f"[{name}] Attempt {attempt + 1}/{cfg.max_attempts} failed: "
                        f"{type(exc).__name__}: {exc}  — retrying in {wait:.1f}s",
                    )
                    time.sleep(wait)
            log.error(f"[{name}] All {cfg.max_attempts} attempts exhausted.")
            raise last_exc  # type: ignore[misc]

        return wrapper

    if func is not None:          # called as @with_retry (no parentheses)
        return decorator(func)
    return decorator              # called as @with_retry(...)


# ─────────────────────────────────────────────────────────────────────────────
#  Retry helper for inline use (no decorator needed)
# ─────────────────────────────────────────────────────────────────────────────


def retry_sync(fn: Callable, *args, cfg: RetryConfig = DEFAULT_RETRY, label="", **kwargs):
    """Inline sync retry without decorator."""
    last_exc = None
    name = label or getattr(fn, "__qualname__", str(fn))
    for attempt in range(cfg.max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, cfg):
                raise
            wait = _wait_time(attempt, cfg)
            log.warning(
                f"[{name}] Attempt {attempt + 1}/{cfg.max_attempts}: "
                f"{type(exc).__name__} — retry in {wait:.1f}s",
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
#  Retryable error classification
# ─────────────────────────────────────────────────────────────────────────────

def _is_retryable(exc: Exception, cfg: RetryConfig) -> bool:
    """Returns True if the exception is worth retrying.
    Handles:
      - Standard Python exceptions (ConnectionError, TimeoutError, OSError)
      - OpenAI SDK errors (APIConnectionError, RateLimitError, APIStatusError 5xx)
      - Any exception whose message contains "connection" or "timeout"
    """
    if isinstance(exc, cfg.retryable):
        return True

    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    # OpenAI SDK error class names
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }
    if exc_type in retryable_names:
        return True

    # HTTP status codes in the exception message
    for code in ("500", "502", "503", "504", "429"):
        if code in exc_msg:
            return True

    # Generic message patterns
    for pattern in ("connection", "timeout", "timed out", "reset", "refused", "unavailable"):
        if pattern in exc_msg:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Frame compression  (WebP — principled quality-based, no arbitrary byte cap)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Why WebP?
#    • Built into Pillow 9+ — no extra install.
#    • ~25-35% smaller than JPEG at equivalent perceptual quality.
#    • Lossless mode available; "quality=85" gives near-transparent loss.
#    • Widely supported as image/webp data-URIs by modern VLMs.
#
#  Strategy: single-pass encode at configurable quality (default 85) with a
#  hard resolution cap.  No ladder, no arbitrary byte budget — quality drives
#  the decision, not file size.
#
#  Fallback: if WebP is unavailable (Pillow built without libwebp), fall back
#  to optimised progressive JPEG at the same quality setting.

_WEBP_SUPPORTED: bool | None = None   # lazily detected


def _webp_available() -> bool:
    global _WEBP_SUPPORTED
    if _WEBP_SUPPORTED is None:
        try:
            import io as _io

            from PIL import Image
            _buf = _io.BytesIO()
            Image.new("RGB", (4, 4)).save(_buf, format="WEBP")
            _WEBP_SUPPORTED = True
        except Exception:
            _WEBP_SUPPORTED = False
    return _WEBP_SUPPORTED


def compress_frame_safe(image_path, cfg=None) -> tuple[bytes, int, int]:
    """Compress a video frame for VLM ingestion.

    Uses WebP at `quality` (default 85) for best perceptual quality per byte.
    Falls back to progressive JPEG if libwebp is unavailable.
    Resizes to `max_dim` on the longest side — never upscales.

    Returns (raw_bytes, final_width, final_height).
    """
    from PIL import Image

    max_dim = getattr(cfg, "FRAME_MAX_DIM", 1280) if cfg else 1280
    quality = getattr(cfg, "FRAME_QUALITY", 85) if cfg else 85

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    buf = io.BytesIO()
    if _webp_available():
        # method=4 balances encode speed vs compression ratio
        img.save(buf, format="WEBP", quality=quality, method=4)
    else:
        img.save(buf, format="JPEG", quality=quality,
                 optimize=True, progressive=True)

    raw = buf.getvalue()
    log.debug(
        f"Frame compressed → {w}×{h}  fmt={'webp' if _webp_available() else 'jpeg'}"
        f"  q={quality}  size={len(raw) // 1024}KB"
        f"  src={getattr(image_path, 'name', str(image_path))}",
    )
    return raw, w, h


def compress_frame_for_vlm(image_path, cfg=None) -> tuple[str, str, int, int]:
    """Compress a frame and return everything the VLM call needs:
      (base64_string, mime_type, width, height)

    Always produces JPEG — LM Studio and most local VLMs only accept JPEG/PNG
    as base64 data URIs.  WebP is used by compress_frame_safe for storage but
    is not suitable here.
    """
    import base64

    from PIL import Image

    max_dim = getattr(cfg, "FRAME_MAX_DIM", 1280) if cfg else 1280
    quality = getattr(cfg, "FRAME_QUALITY", 85) if cfg else 85

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    log.debug(
        f"VLM frame → {w}×{h}  fmt=jpeg  q={quality}  size={len(raw) // 1024}KB"
        f"  src={getattr(image_path, 'name', str(image_path))}",
    )
    return b64, "image/jpeg", w, h


# ─────────────────────────────────────────────────────────────────────────────
#  Frame similarity  (rule-based, no ML needed)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Algorithm: downsample both frames to a small thumbnail (default 16×16),
#  convert to grayscale, compute normalised MSE.
#
#    similarity = 1 - MSE / 255²
#
#  This is equivalent to a PSNR-derived score and needs only numpy + Pillow.
#  At 16×16 the operation takes < 1 ms per pair — negligible overhead.
#
#  Interpretation examples (empirically calibrated on lecture video):
#    ≥ 0.99  → virtually identical  (same slide, compression rounding)
#    0.92-0.99 → slight diff        (cursor move, clock tick, small animation)
#    0.80-0.92 → moderate diff      (bullet revealed, text appended)
#    < 0.80  → clearly different    (slide change, scene cut)
#
#  Recommended threshold for "skip OCR/VLM":  0.90  (configurable)
_SIM_THUMB_SIZE = 16   # px — fast and still discriminative


def compute_frame_similarity(path_a, path_b, thumb_size: int = _SIM_THUMB_SIZE) -> float:
    """Return perceptual similarity between two frame images in [0.0, 1.0].
    1.0 = pixel-perfect identical, 0.0 = maximally different.

    Fast: loads two tiny thumbnails, no GPU needed.
    """
    import numpy as np
    from PIL import Image

    def _thumb(p):
        return np.array(
            Image.open(p).convert("L").resize((thumb_size, thumb_size), Image.LANCZOS),
            dtype=float,
        )

    try:
        a, b = _thumb(path_a), _thumb(path_b)
        mse = float(np.mean((a - b) ** 2))
        return max(0.0, 1.0 - mse / (255.0 ** 2))
    except Exception as e:
        log.warning(f"Frame similarity check failed ({e}) — treating as different")
        return 0.0
