from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


def get_video_duration(video_path: str, timeout_s: int = 300) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout_s,
    )
    if result.returncode != 0:
        log.warning(f"ffprobe returned {result.returncode}: {result.stderr.strip()[:200]}")
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        log.warning(f"ffprobe invalid output: {result.stdout.strip()[:100]}")
        return 0.0
