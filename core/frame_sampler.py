"""core/frame_sampler.py — Phase 2a
Builds an STT-driven frame schedule, then extracts only those frames via ffmpeg.
Retry logic wraps every ffmpeg subprocess call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from utils.retry import RetryConfig, retry_sync

log = logging.getLogger(__name__)

_FFMPEG_RETRY = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=10.0)


@dataclass
class FrameRequest:
    timestamp_ms: int
    reason: str   # "sentence_end" | "long_pause" | "floor"
    sentence_id: int


# ─────────────────────────────────────────────────────────────────────────────
#  Schedule builder (pure logic, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_schedule(sentences, cfg) -> list[FrameRequest]:
    requests: list[FrameRequest] = []
    seen_buckets = set()

    def add(ts_ms: int, reason: str, sid: int):
        ts_ms = max(0, ts_ms)
        bucket = ts_ms // 300          # 300 ms de-dup window
        if bucket not in seen_buckets:
            seen_buckets.add(bucket)
            requests.append(FrameRequest(ts_ms, reason, sid))

    for seg in sentences:
        # Trigger 1: sentence-end capture (settled visual state)
        add(seg.end_ms - cfg.SENTENCE_END_OFFSET_MS, "sentence_end", seg.id)

        # Trigger 2: long-pause mid-point (slide transitions live here)
        if seg.pause_after_ms >= cfg.LONG_PAUSE_THRESHOLD_MS:
            add(seg.end_ms + seg.pause_after_ms // 2, "long_pause", seg.id)

    # Trigger 3: sparse floor for demo / silent stretches
    if sentences:
        video_end_ms = sentences[-1].end_ms + sentences[-1].pause_after_ms
        interval_ms = int(1000 / cfg.FALLBACK_FPS_FLOOR)
        t = 0
        while t <= video_end_ms:
            add(t, "floor", _nearest_sentence_id(sentences, t))
            t += interval_ms

    requests.sort(key=lambda r: r.timestamp_ms)
    log.info(
        f"Frame schedule: {len(requests)} frames  "
        f"({sum(1 for r in requests if r.reason == 'sentence_end')} sentence_end  "
        f"{sum(1 for r in requests if r.reason == 'long_pause')} long_pause  "
        f"{sum(1 for r in requests if r.reason == 'floor')} floor)",
    )
    return requests


def _nearest_sentence_id(sentences, ts_ms: int) -> int:
    best, best_dist = 0, float("inf")
    for s in sentences:
        dist = abs(s.end_ms - ts_ms)
        if dist < best_dist:
            best_dist, best = dist, s.id
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    schedule: list[FrameRequest],
    out_dir: Path,
    cfg,
) -> list[tuple[FrameRequest, Path]]:
    """Decode only scheduled frames via ffmpeg.
    Skips already-extracted frames (resume-safe).
    Each ffmpeg call is retried up to 4 times.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    to_extract = []
    for req in schedule:
        path = _frame_path(frames_dir, req.timestamp_ms)
        if not path.exists():
            to_extract.append(req)

    if to_extract:
        log.info(f"Extracting {len(to_extract)} new frames …")
        for req in to_extract:
            _extract_one_frame(video_path, req, frames_dir, cfg)
    else:
        log.info("All frames already on disk.")

    # Collect final list (only frames that actually exist)
    results = []
    for req in schedule:
        path = _frame_path(frames_dir, req.timestamp_ms)
        if path.exists():
            results.append((req, path))
        else:
            log.warning(f"Frame missing after extraction: ts={req.timestamp_ms}ms")

    log.info(f"Frames available: {len(results)}/{len(schedule)}")
    return results


def _extract_one_frame(video_path: str, req: FrameRequest, frames_dir: Path, cfg):
    """Extract a single frame at req.timestamp_ms.  Retried on failure."""
    out_path = _frame_path(frames_dir, req.timestamp_ms)
    ts_sec = req.timestamp_ms / 1000.0

    def _run():
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts_sec:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-vf", f"scale='min({cfg.FRAME_MAX_DIM},iw)':-2",
            str(out_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=getattr(cfg, "FFMPEG_TIMEOUT_S", 300),
            )
        except subprocess.TimeoutExpired:
            raise OSError(f"ffmpeg frame extract timed out (ts={req.timestamp_ms}ms)")
        if result.returncode != 0:
            raise OSError(
                f"ffmpeg frame extract failed (ts={req.timestamp_ms}ms): "
                f"{result.stderr[-300:]}",
            )

    try:
        retry_sync(_run, cfg=_FFMPEG_RETRY, label=f"ffmpeg_frame_{req.timestamp_ms}")
    except Exception as e:
        log.error(f"Giving up on frame ts={req.timestamp_ms}ms after retries: {e}")


def _frame_path(frames_dir: Path, ts_ms: int) -> Path:
    return frames_dir / f"frame_{ts_ms:010d}.jpg"


def frame_hash(frame_path: Path) -> str:
    """MD5 of first 8 KB — fast near-duplicate detection."""
    with open(frame_path, "rb") as f:
        return hashlib.md5(f.read(8192)).hexdigest()


def save_schedule(schedule: list[FrameRequest], out_dir: Path):
    path = out_dir / "frame_schedule.json"
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in schedule], f, indent=2)
    log.info(f"Frame schedule saved → {path}")
