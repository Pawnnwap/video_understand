"""pipeline.py — orchestrator
The single entry point. Given a video file, runs all phases and
produces a queryable VideoDatabase.

Usage:
    python pipeline.py my_lecture.mp4

Resume-safe: if the process is interrupted, re-running picks up where it left off.
"""

from __future__ import annotations

import argparse
import logging

# Local tracks (STT, OCR, embeddings) run fully offline — block HF network
# calls so they never try to fetch models on demand.  The VLM track is REMOTE
# (OpenCode Zen) and is unaffected by these flags.
import os
import sys
from pathlib import Path

import config as cfg
from utils.logging_setup import setup_logging
from utils.video import get_video_duration

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# ModelScope (FunASR weights) — use the local cache only, never hit the network.
os.environ["MODELSCOPE_OFFLINE"] = "1"
os.environ["MODELSCOPE_DOMAIN_NAME"] = "www.modelscope.cn"


def _log_file_for(video_path: str, db_root: str) -> str | None:
    try:
        stem = Path(video_path).stem
        d = Path(db_root) / stem
        d.mkdir(parents=True, exist_ok=True)
        return str(d / "pipeline.log")
    except Exception:
        return None


log = logging.getLogger("pipeline")


def make_db_dir(video_path: str, db_root: str) -> Path:
    """Create a per-video output directory under db_root."""
    stem = Path(video_path).stem
    db_dir = Path(db_root) / stem
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def build_vlm():
    """Instantiate the opencode-backed VLM used for the vision track AND
    the Phase-3 text-LLM ("pure mode" — text-only, no image).

    Spawns a persistent ``opencode serve`` subprocess.  Vision calls reuse
    one session; text-LLM calls create a fresh session per message.  No
    base_url / API key needed — opencode's built-in provider hosts the free
    models.
    """
    from core.vision.opencode_vlm import OpencodeVLM
    return OpencodeVLM(
        model=cfg.VLM_MODEL,
        port=getattr(cfg, "OPENCODE_SERVER_PORT", 0) or 0,
        variant=getattr(cfg, "VLM_VARIANT", None),
        text_model=getattr(cfg, "LLM_MODEL", cfg.VLM_MODEL),
        text_variant=getattr(cfg, "LLM_VARIANT", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, force_reprocess: bool = False):

    setup_logging(level=logging.INFO, log_file=_log_file_for(video_path, cfg.DB_DIR))
    log.info(f"{'=' * 60}")
    log.info("  Video Understanding Pipeline")
    log.info(f"  Input : {video_path}")
    log.info(f"{'=' * 60}")

    # ── Resolve source: local file or YouTube/Bilibili URL ───────────────
    from downloader import _expand_short_code, is_url, resolve_source
    needs_download = is_url(video_path) or (_expand_short_code(video_path) is not None)
    if needs_download:
        log.info("Remote source detected — downloading video …")
        try:
            from pathlib import Path as _P
            local = resolve_source(
                video_path,
                download_dir=str(_P(cfg.DB_DIR) / "_downloads"),
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

    db_dir = make_db_dir(video_path, cfg.DB_DIR)
    vlm = build_vlm()
    log.info(f"VLM  (opencode): model={cfg.VLM_MODEL}  variant={getattr(cfg, 'VLM_VARIANT', None)}")
    log.info(f"LLM  (opencode pure mode for Phase-3 fusion): model={getattr(cfg, 'LLM_MODEL', cfg.VLM_MODEL)}  variant={getattr(cfg, 'LLM_VARIANT', None)}")
    duration = get_video_duration(video_path)
    log.info(f"Video duration: {duration:.1f}s  |  Output dir: {db_dir}")

    # ── PHASE 1 : STT ────────────────────────────────────────────────────
    log.info("\n── Phase 1: Speech-to-Text ──────────────────────────────")
    from core.stt import extract_audio, load_transcript, save_transcript, transcribe

    sentences = load_transcript(db_dir) if not force_reprocess else None
    if sentences:
        log.info(f"Loaded cached transcript ({len(sentences)} sentences).")
    else:
        audio_path = extract_audio(video_path, db_dir, cfg)
        sentences = transcribe(audio_path, cfg)
        save_transcript(sentences, db_dir)

    from core.lang import detect_language
    lang = detect_language(sentences)
    log.info(f"Detected language: {lang}")

    # ── PHASE 2a : Frame schedule ────────────────────────────────────────
    log.info("\n── Phase 2a: Adaptive Frame Sampling ───────────────────")
    from core.vision.frame_sampler import (
        build_frame_schedule,
        extract_frames,
        save_schedule,
    )

    schedule = build_frame_schedule(sentences, cfg)
    save_schedule(schedule, db_dir)
    frame_results = extract_frames(video_path, schedule, db_dir, cfg)
    log.info(f"Frames ready: {len(frame_results)}")

# ── PHASE 2b : VLM analysis ──────────────────────────────────────────
    log.info("\n── Phase 2b: VLM Frame Analysis ─────────────────────────────────")
    from core.vision.vlm_analyser import analyse_all_frames

    # vlm stays alive through Phase 3 (also used for the fusion text-LLM).
    try:

        analyses = analyse_all_frames(frame_results, vlm, cfg, db_dir, lang=lang)

        # ── PHASE 3 : Temporal fusion ────────────────────────────────────
        log.info("\n── Phase 3: Temporal Fusion (opencode pure mode) ───────────────")
        from core.fusion import fuse, load_fused, save_fused

        fused = load_fused(db_dir) if not force_reprocess else None
        if fused:
            log.info(f"Loaded cached fusion ({len(fused)} segments).")
        else:
            fused = fuse(sentences, analyses, vlm, cfg, lang=lang)
            save_fused(fused, db_dir)
    finally:
        vlm.close()

    # ── PHASE 4 : Database ───────────────────────────────────────────────
    log.info("\n── Phase 4: Building Database ───────────────────────────")
    from core.database import VideoDatabase

    db = VideoDatabase(db_dir, cfg)
    db.ingest(fused, video_path, duration)

    # ── Summary stats ────────────────────────────────────────────────────
    slides = db.get_slide_index()
    log.info(f"\n{'=' * 60}")
    log.info("  Pipeline complete!")
    log.info(f"  Segments : {db.count()}")
    log.info(f"  Slides   : {len(slides)}")
    log.info(f"  DB dir   : {db_dir}")
    log.info(f"{'=' * 60}\n")

    return db


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video Understanding Pipeline")
    parser.add_argument("video", help="Local video path OR YouTube/Bilibili URL")
    parser.add_argument("--force", action="store_true", help="Force full reprocessing")
    parser.add_argument("--query", action="store_true", help="Launch interactive query REPL after processing")
    parser.add_argument("--vlm-model", help="opencode vision model id (default opencode/mimo-v2.5-free)")
    parser.add_argument("--vlm-variant", help="opencode model variant (low|medium|high)")
    parser.add_argument(
        "--text-model", "--llm-model", dest="text_model",
        help="opencode text model for fusion and queries",
    )
    parser.add_argument(
        "--text-variant", "--llm-variant", dest="text_variant",
        help="opencode text-model reasoning variant (low|medium|high)",
    )
    parser.add_argument("--opencode-port", type=int, help="Fixed port for the opencode server (0 = random)")
    args = parser.parse_args()

    if args.vlm_model:
        cfg.VLM_MODEL = args.vlm_model
    if args.vlm_variant is not None:
        cfg.VLM_VARIANT = cfg.normalize_variant(args.vlm_variant)
    if args.text_model:
        cfg.LLM_MODEL = args.text_model
    if args.text_variant is not None:
        cfg.LLM_VARIANT = cfg.normalize_variant(args.text_variant)
    if args.opencode_port is not None:
        cfg.OPENCODE_SERVER_PORT = args.opencode_port

    db = run_pipeline(args.video, force_reprocess=args.force)

    if args.query:
        # Reuse the complete project session from cli.py instead of keeping a
        # second, divergent REPL here.
        from cli import _workspace_repl
        _workspace_repl(Path(cfg.DB_DIR), open_immediately=db.db_dir)


if __name__ == "__main__":
    main()
