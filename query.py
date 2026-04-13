"""query.py — standalone query interface
Load an existing video database and query it interactively
without rerunning the pipeline.

Usage:
    python query.py ./video_db/my_lecture
    python query.py ./video_db/my_lecture --ask "What is the main topic?"
    python query.py ./video_db/my_lecture --summary
    python query.py ./video_db/my_lecture --outline
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging

# Block all HuggingFace network calls — everything must run fully locally.
import os
import sys
from pathlib import Path

import config as cfg
from core.database import VideoDatabase
from openai import OpenAI

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# query.py shadows the query/ package — load query_engine directly by path
_qe_path = Path(__file__).parent / "query" / "query_engine.py"
_spec = importlib.util.spec_from_file_location("query.query_engine", _qe_path)
_qe_mod = importlib.util.module_from_spec(_spec)
sys.modules["query.query_engine"] = _qe_mod
_spec.loader.exec_module(_qe_mod)
QueryEngine = _qe_mod.QueryEngine
_parse_timestamp = _qe_mod._parse_timestamp

logging.basicConfig(level=logging.WARNING)   # quiet for interactive use


def main():
    parser = argparse.ArgumentParser(description="Query a processed video database")
    parser.add_argument("db_dir", help="Path to the video database directory")
    parser.add_argument("--ask", help="Ask a single question and exit")
    parser.add_argument("--summary", action="store_true", help="Print video summary")
    parser.add_argument("--outline", action="store_true", help="Print topic outline")
    parser.add_argument("--slides", action="store_true", help="List all slide changes")
    parser.add_argument("--transcript", action="store_true", help="Print full transcript")
    parser.add_argument("--knowledge", help="Deep knowledge extraction on a topic")
    parser.add_argument("--at", help="Query at timestamp MM:SS", metavar="MM:SS")
    parser.add_argument("--question", help="Question for --at (optional)", default="What is happening here?")
    parser.add_argument("--base-url", help="LM Studio base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--model", help="Model name for queries")
    args = parser.parse_args()

    if args.base_url:
        cfg.LM_STUDIO_BASE_URL = args.base_url
    if args.api_key:
        cfg.LM_STUDIO_API_KEY = args.api_key
    if args.model:
        cfg.LLM_MODEL = args.model

    try:
        db = VideoDatabase.load(args.db_dir, cfg)
    except Exception as e:
        print(f"Error loading database: {e}", file=sys.stderr)
        sys.exit(1)

    if db.count() == 0:
        print("Database is empty. Run pipeline.py first.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=cfg.LM_STUDIO_BASE_URL, api_key=cfg.LM_STUDIO_API_KEY)
    engine = QueryEngine(db, client, cfg)

    # Non-interactive modes
    if args.summary:
        print(engine.summarize("comprehensive"))

    elif args.outline:
        print(engine.get_topic_outline())

    elif args.slides:
        slides = db.get_slide_index()
        print(f"\n{len(slides)} slide changes:\n")
        for s in slides:
            print(f"  {s['timestamp']}  {s['slide_title'] or '(no title)'}")

    elif args.transcript:
        print(db.get_full_transcript())

    elif args.ask:
        print(engine.ask(args.ask))

    elif args.knowledge:
        print(engine.extract_knowledge(args.knowledge))

    elif args.at:
        ts_ms = _parse_timestamp(args.at)
        print(engine.query_at_time(ts_ms, args.question))

    else:
        # Default: interactive REPL
        engine.repl()


if __name__ == "__main__":
    main()
