"""core/vision/frame_sampler.py — Phase 2a
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

from utils.video import get_video_duration

log = logging.getLogger(__name__)


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
    seen_buckets: set[int] = set()

    def add(ts_ms: int, reason: str, sid: int) -> None:
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
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    to_extract = [req for req in schedule if not _frame_path(frames_dir, req.timestamp_ms).exists()]

    if to_extract:
        log.info(f"Extracting {len(to_extract)} new frames ...")
        duration_s = get_video_duration(video_path, getattr(cfg, "FFMPEG_TIMEOUT_S", 300))
        failed = 0
        for req in to_extract:
            success = _extract_one_frame(video_path, req, frames_dir, cfg, duration_s)
            if not success:
                failed += 1
        if failed:
            log.warning(f"Failed to extract {failed}/{len(to_extract)} frames")
    else:
        log.info("All frames already on disk.")

    results = []
    for req in schedule:
        path = _frame_path(frames_dir, req.timestamp_ms)
        if path.exists():
            results.append((req, path))
        else:
            log.warning(f"Frame missing after extraction: ts={req.timestamp_ms}ms")

    log.info(f"Frames available: {len(results)}/{len(schedule)}")
    return results


def _ffmpeg_commands(video_path: str, ts_sec: float, out_path: Path, cfg) -> list[list[str]]:
    cmds = []
    cmds.append([
        "ffmpeg", "-y",
        "-ss", f"{ts_sec:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        "-vf", f"scale='min({cfg.FRAME_MAX_DIM},iw)':-2",
        str(out_path),
    ])
    cmds.append([
        "ffmpeg", "-y",
        "-ss", f"{ts_sec:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ])
    cmds.append([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", f"{ts_sec:.3f}",
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ])
    cmds.append([
        "ffmpeg", "-y",
        "-noaccurate_seek",
        "-ss", f"{ts_sec:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ])
    return cmds


def _extract_one_frame(video_path: str, req: FrameRequest, frames_dir: Path, cfg, duration_s: float) -> bool:
    out_path = _frame_path(frames_dir, req.timestamp_ms)
    ts_sec = req.timestamp_ms / 1000.0

    if duration_s > 0 and ts_sec > duration_s:
        log.warning(f"Timestamp {ts_sec:.3f}s exceeds duration {duration_s:.1f}s — skipping")
        return False

    commands = _ffmpeg_commands(video_path, ts_sec, out_path, cfg)
    timeout_s = getattr(cfg, "FFMPEG_EXTRACTION_TIMEOUT_S", 60)
    last_error: str | None = None

    for attempt, cmd in enumerate(commands):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout_s,
            )
            if result.returncode == 0 and out_path.exists():
                if attempt > 0:
                    log.info(f"Frame ts={req.timestamp_ms}ms extracted via fallback #{attempt}")
                return True

            log.error(
                f"ffmpeg attempt {attempt + 1}/{len(commands)} failed (ret={result.returncode}) ts={req.timestamp_ms}ms\n"
                f"  video: {video_path}\n"
                f"  output: {out_path}\n"
                f"  command: {' '.join(cmd)}\n"
                f"  stderr: {result.stderr}",
            )
            last_error = f"ret={result.returncode}: {result.stderr[-500:] if result.stderr else 'no stderr'}"
        except subprocess.TimeoutExpired:
            log.error(f"ffmpeg attempt {attempt + 1} timed out for ts={req.timestamp_ms}ms")
            last_error = "timeout"

    log.error(f"All {len(commands)} extraction attempts failed for ts={req.timestamp_ms}ms: {last_error}")
    return False


def _frame_path(frames_dir: Path, ts_ms: int) -> Path:
    return frames_dir / f"frame_{ts_ms:010d}.jpg"


def frame_hash(frame_path: Path) -> str:
    """MD5 of first 8 KB — fast near-duplicate detection."""
    with open(frame_path, "rb") as f:
        return hashlib.md5(f.read(8192)).hexdigest()


def save_schedule(schedule: list[FrameRequest], out_dir: Path) -> None:
    path = out_dir / "frame_schedule.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in schedule], f, indent=2)
    log.info(f"Frame schedule saved -> {path}")
