"""One-shot queries and compatibility launcher for a processed video project.

Interactive use is delegated to the canonical project session in ``cli.py`` so
there is only one set of interactive commands to maintain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config as cfg
from cli import _load_project, _workspace_repl
from query.crosscheck import crosscheck
from query.query_engine import _parse_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a processed video database")
    parser.add_argument("db_dir", help="Path to the video database directory")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--ask", help="Ask a question or request a topic deep dive, then exit")
    action.add_argument(
        "--summary", nargs="?", const="comprehensive",
        choices=["comprehensive", "brief", "headline"],
        help="Print a whole-video summary (style: comprehensive [default], brief, headline)",
    )
    action.add_argument("--outline", action="store_true", help="Print the topic outline")
    action.add_argument("--slides", action="store_true", help="List all slide changes")
    action.add_argument("--transcript", action="store_true", help="Print the full transcript")
    action.add_argument("--at", help="Query timestamp MM:SS", metavar="MM:SS")
    action.add_argument("--crosscheck", type=int, metavar="N", help="Fact-check the top N claims")
    parser.add_argument("--question", help="Question for --at", default="What is happening here?")
    parser.add_argument(
        "--text-model", "--llm-model", dest="text_model",
        help="opencode text model for the query",
    )
    parser.add_argument(
        "--text-variant", "--llm-variant", dest="text_variant",
        help="opencode text-model reasoning variant (low|medium|high)",
    )
    args = parser.parse_args()

    if args.text_model:
        cfg.VLM_LLM_MODEL = args.text_model
    if args.text_variant:
        cfg.VLM_LLM_VARIANT = args.text_variant

    db_path = Path(args.db_dir)
    if not db_path.is_dir() or not (db_path / "transcript.json").exists():
        print(f"Not a processed video project: {db_path}", file=sys.stderr)
        raise SystemExit(1)

    has_action = any(
        (
            args.ask,
            args.summary,
            args.outline,
            args.slides,
            args.transcript,
            args.at,
            args.crosscheck is not None,
        )
    )
    if not has_action:
        _workspace_repl(Path(cfg.DB_DIR), open_immediately=db_path)
        return

    loaded = _load_project(db_path)
    if loaded is None:
        raise SystemExit(1)
    db, engine, llm = loaded
    try:
        if args.summary:
            print(engine.summarize(args.summary))
        elif args.outline:
            print(engine.get_topic_outline())
        elif args.slides:
            slides = db.get_slide_index()
            print(f"\n{len(slides)} slide changes:\n")
            for slide in slides:
                print(f"  {slide['timestamp']}  {slide['slide_title'] or '(no title)'}")
        elif args.transcript:
            print(db.get_full_transcript())
        elif args.ask:
            print(engine.ask(args.ask))
        elif args.at:
            print(engine.query_at_time(_parse_timestamp(args.at), args.question))
        elif args.crosscheck is not None:
            print(crosscheck(engine, max(1, min(args.crosscheck, 10))))
    finally:
        llm.close()


if __name__ == "__main__":
    main()
