"""
pipeline.py — orchestrator
The single entry point. Given a video file, runs all phases and
produces a queryable VideoDatabase.

Usage:
    python pipeline.py my_lecture.mp4

Resume-safe: if the process is interrupted, re-running picks up where it left off.
"""

from __future__ import annotations
# Block all HuggingFace network calls — everything must run fully locally.
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import config as cfg

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("pipeline")


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=cfg.FFMPEG_TIMEOUT_S,
        )
        if result.returncode != 0:
            log.warning(f"ffprobe returned {result.returncode}: {result.stderr.strip()[:200]}")
            return 0.0
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        log.error(f"ffprobe timed out after {cfg.FFMPEG_TIMEOUT_S}s for {video_path}")
        return 0.0
    except (ValueError, FileNotFoundError) as e:
        log.warning(f"ffprobe failed: {e}")
        return 0.0


def make_db_dir(video_path: str, db_root: str) -> Path:
    """Create a per-video output directory under db_root."""
    stem   = Path(video_path).stem
    db_dir = Path(db_root) / stem
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def build_client():
    """Instantiate the OpenAI-compatible client pointing at LM Studio."""
    from openai import OpenAI
    return OpenAI(base_url=cfg.LM_STUDIO_BASE_URL, api_key=cfg.LM_STUDIO_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, force_reprocess: bool = False):

    log.info(f"{'='*60}")
    log.info(f"  Video Understanding Pipeline")
    log.info(f"  Input : {video_path}")
    log.info(f"{'='*60}")

    # ── Resolve source: local file or YouTube/Bilibili URL ───────────────
    from downloader import resolve_source, is_url, _expand_short_code
    needs_download = is_url(video_path) or (_expand_short_code(video_path) is not None)
    if needs_download:
        log.info("Remote source detected — downloading video …")
        try:
            local = resolve_source(
                video_path,
                download_dir=cfg.DB_DIR + "/_downloads",
                max_duration_sec=getattr(cfg, "DOWNLOAD_MAX_DURATION_SEC", 0),
            )
            video_path = str(local)
            log.info(f"Using downloaded file: {video_path}")
        except Exception as e:
            log.error(f"Download failed: {e}")
            sys.exit(1)
    elif not Path(video_path).exists():
        log.error(f"File not found: {video_path}")
        sys.exit(1)

    db_dir   = make_db_dir(video_path, cfg.DB_DIR)
    client   = build_client()
    duration = get_video_duration(video_path)
    log.info(f"Video duration: {duration:.1f}s  |  Output dir: {db_dir}")

    # ── PHASE 1 : STT ────────────────────────────────────────────────────
    log.info("\n── Phase 1: Speech-to-Text ──────────────────────────────")
    from core.stt import extract_audio, transcribe, save_transcript, load_transcript

    sentences = load_transcript(db_dir) if not force_reprocess else None
    if sentences:
        log.info(f"Loaded cached transcript ({len(sentences)} sentences).")
    else:
        audio_path = extract_audio(video_path, db_dir)
        sentences  = transcribe(audio_path, cfg)
        save_transcript(sentences, db_dir)

    # ── PHASE 2a : Frame schedule ────────────────────────────────────────
    log.info("\n── Phase 2a: Adaptive Frame Sampling ───────────────────")
    from core.frame_sampler import build_frame_schedule, extract_frames, save_schedule

    schedule     = build_frame_schedule(sentences, cfg)
    save_schedule(schedule, db_dir)
    frame_results = extract_frames(video_path, schedule, db_dir, cfg)
    log.info(f"Frames ready: {len(frame_results)}")

    # ── PHASE 2b : VLM analysis ──────────────────────────────────────────
    log.info("\n── Phase 2b: VLM Frame Analysis ─────────────────────────")
    from core.vlm_analyser import analyse_all_frames

    analyses = analyse_all_frames(frame_results, client, cfg, db_dir)

    # ── PHASE 3 : Temporal fusion ────────────────────────────────────────
    log.info("\n── Phase 3: Temporal Fusion ─────────────────────────────")
    from core.fusion import fuse, save_fused, load_fused

    fused = load_fused(db_dir) if not force_reprocess else None
    if fused:
        log.info(f"Loaded cached fusion ({len(fused)} segments).")
    else:
        fused = fuse(sentences, analyses, client, cfg)
        save_fused(fused, db_dir)

    # ── PHASE 4 : Database ───────────────────────────────────────────────
    log.info("\n── Phase 4: Building Database ───────────────────────────")
    from core.database import VideoDatabase

    db = VideoDatabase(db_dir, cfg)
    db.ingest(fused, video_path, duration)

    # ── Summary stats ────────────────────────────────────────────────────
    slides = db.get_slide_index()
    log.info(f"\n{'='*60}")
    log.info(f"  Pipeline complete!")
    log.info(f"  Segments : {db.count()}")
    log.info(f"  Slides   : {len(slides)}")
    log.info(f"  DB dir   : {db_dir}")
    log.info(f"{'='*60}\n")

    return db


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video Understanding Pipeline")
    parser.add_argument("video",   help="Local video path OR YouTube/Bilibili URL")
    parser.add_argument("--force", action="store_true", help="Force full reprocessing")
    parser.add_argument("--query", action="store_true", help="Launch interactive query REPL after processing")
    args = parser.parse_args()

    db = run_pipeline(args.video, force_reprocess=args.force)

    if args.query:
        from query.query_engine import QueryEngine
        client = build_client()
        engine = QueryEngine(db, client, cfg)
        engine.repl()


if __name__ == "__main__":
    main()
