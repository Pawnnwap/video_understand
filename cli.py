"""cli.py -- workspace shell for the Video Understanding System

Usage:
    python cli.py                    # enter workspace, pick a project
    python cli.py BV1iatTeGENk       # process + open that video directly
    python cli.py ./video_db/foo     # open existing db directly

Workspace commands:
    list / ls               list all processed projects
    open <name|#>           enter a project
    process <path|URL|BV>   run pipeline on a new video
    BV1iatTeGENk            bare BV/YouTube code -- download, analyse, open
    dQw4w9WgXcQ             (YouTube ID works the same way)
    https://...             full URL also works
    help                    show help
    quit / exit             exit

Project commands (once inside a project):
    /summary                comprehensive summary
    /headline               one-line headline
    /brief                  3-5 sentence overview
    /outline                topic outline from slides
    /slides                 list all slide changes with timestamps
    /transcript             full spoken transcript
    /at MM:SS [question]    what was on screen at a specific moment
    /knowledge <topic>      deep extraction on a topic
    /open <name|#>          switch to a different project
    /help                   show this help
    /back                   return to workspace prompt
    /quit                   exit entirely
    <anything else>         semantic search + RAG answer
"""

from __future__ import annotations

import io
import argparse
import sys

import config as cfg
from core.database import VideoDatabase
from downloader import _expand_short_code, is_url
from openai import OpenAI

# Force UTF-8 on Windows consoles so CJK / box-drawing chars render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import importlib.util
import json
import logging
import os
import subprocess
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


logging.basicConfig(level=logging.WARNING)   # quiet in interactive mode

# -- resolve query_engine by path to avoid query.py / query/ package shadow --
_qe_path = Path(__file__).parent / "query" / "query_engine.py"
_spec = importlib.util.spec_from_file_location("query.query_engine", _qe_path)
_qe_mod = importlib.util.module_from_spec(_spec)
sys.modules["query.query_engine"] = _qe_mod
_spec.loader.exec_module(_qe_mod)
QueryEngine = _qe_mod.QueryEngine
_parse_timestamp = _qe_mod._parse_timestamp


def _is_video_source(s: str) -> bool:
    """Return True if s looks like something the pipeline can process."""
    return is_url(s) or (_expand_short_code(s) is not None)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_client(base_url=None, api_key=None):
    return OpenAI(
        base_url=base_url or cfg.LM_STUDIO_BASE_URL,
        api_key=api_key or cfg.LM_STUDIO_API_KEY,
    )

def _list_projects(db_root: Path) -> list:
    """Return processed project dirs, newest first."""
    if not db_root.exists():
        return []
    return sorted(
        [d for d in db_root.iterdir()
         if d.is_dir()
         and not d.name.startswith("_")
         and (d / "transcript.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def _print_project_list(projects: list):
    if not projects:
        print("  (no processed projects yet)")
        return
    for i, p in enumerate(projects, 1):
        name = p.name
        segs = ""
        tl = p / "timeline.json"
        if tl.exists():
            try:
                with open(tl, encoding="utf-8") as f:
                    data = json.load(f)
                segs = f"  [{len(data.get('segments', []))} segments]"
            except Exception:
                pass
        print(f"  {i:2}.  {name}{segs}")


def _resolve_project(token: str, projects: list):
    """Resolve token to a Path by list number, exact name, or substring."""
    # Strip surrounding quotes (single or double) that shells or users may add
    token = token.strip().strip("\"'")
    try:
        idx = int(token) - 1
        if 0 <= idx < len(projects):
            return projects[idx]
        print(f"  No project #{int(token)}.")
        return None
    except ValueError:
        pass

    for p in projects:
        if p.name == token:
            return p

    matches = [p for p in projects if token.lower() in p.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  Ambiguous -- matches {len(matches)} projects:")
        for m in matches:
            print(f"    {m.name}")
        return None

    print(f"  No project matching '{token}'.")
    return None


def _load_project(db_path: Path):
    """Load DB + engine. Returns (db, engine) or None on failure."""
    try:
        db = VideoDatabase.load(str(db_path), cfg)
    except Exception as e:
        print(f"  Error loading database: {e}")
        return None
    if db.count() == 0:
        print("  Database is empty -- run 'process' first.")
        return None
    engine = QueryEngine(db, _build_client(), cfg)
    return db, engine


def _run_pipeline(source: str, base_url=None, api_key=None, vlm_model=None, llm_model=None) -> bool:
    """Run pipeline.py on source, inheriting stdio for live output."""
    print(f"\n  Launching pipeline for: {source}")
    print('  (runs in the foreground -- please wait)\n')
    cmd = [sys.executable, str(Path(__file__).parent / 'pipeline.py'), source]
    if base_url:
        cmd.extend(['--base-url', base_url])
    if api_key:
        cmd.extend(['--api-key', api_key])
    if vlm_model:
        cmd.extend(['--vlm-model', vlm_model])
    if llm_model:
        cmd.extend(['--llm-model', llm_model])
    r = subprocess.run(cmd)
    return r.returncode == 0


# ---------------------------------------------------------------------------
#  Project-level REPL
# ---------------------------------------------------------------------------

_HELP_PROJECT = """
  /summary              comprehensive summary
  /headline             one-line headline
  /brief                3-5 sentence overview
  /outline              topic outline from slides
  /slides               all slide changes with timestamps
  /transcript           full spoken transcript
  /at MM:SS [question]  what was on screen at this moment
  /knowledge <topic>    deep extraction on a topic
  /open <name|#>        switch to another project
  /help                 show this help
  /back                 return to workspace
  /quit                 exit
  <anything else>       semantic search + RAG answer"""


def _project_repl(db_path: Path, db_root: Path) -> bool:
    """Inner REPL for one project.
    Returns True  -> caller should return to workspace.
    Returns False -> caller should exit entirely.
    """
    result = _load_project(db_path)
    if result is None:
        return True

    db, engine = result
    name = db_path.name
    short = name[:52] + "..." if len(name) > 55 else name

    print("\n" + "=" * 60)
    print(f"  Project : {short}")
    print(f"  Segments: {db.count()}")
    print("=" * 60)
    print(_HELP_PROJECT)
    print()

    while True:
        try:
            raw = input(f"[{short[:35]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        # -- navigation --
        if cmd in ("/quit", "/exit", "quit", "exit"):
            return False

        if cmd in ("/back", "/workspace", "back"):
            return True

        if cmd == "/help":
            print(_HELP_PROJECT)

        elif cmd == "/open":
            if not rest:
                print("  Usage: /open <name or number>")
                continue
            projects = _list_projects(db_root)
            target = _resolve_project(rest, projects)
            if target:
                keep_going = _project_repl(target, db_root)
                if not keep_going:
                    return False
                return True  # back to workspace after sub-project

        # -- query commands --
        elif cmd == "/summary":
            print("\n[Generating summary...]\n")
            print(engine.summarize("comprehensive"))

        elif cmd == "/headline":
            print("\n[Generating headline...]\n")
            print(engine.summarize("headline"))

        elif cmd == "/brief":
            print("\n[Generating brief overview...]\n")
            print(engine.summarize("brief"))

        elif cmd == "/outline":
            print("\n[Building outline...]\n")
            print(engine.get_topic_outline())

        elif cmd == "/slides":
            slides = db.get_slide_index()
            if slides:
                print(f"\n  {len(slides)} slide changes:\n")
                for s in slides:
                    print(f"  {s['timestamp']}  {s.get('slide_title') or '(no title)'}")
            else:
                print("  No slide changes detected.")

        elif cmd == "/transcript":
            print()
            print(db.get_full_transcript())

        elif cmd == "/at":
            sub = rest.split(None, 1)
            if not sub:
                print("  Usage: /at MM:SS [question]")
            else:
                ts_ms = _parse_timestamp(sub[0])
                q = sub[1] if len(sub) > 1 else "这个时刻屏幕上显示的是什么？"
                print(f"\n[Querying at {sub[0]}...]\n")
                print(engine.query_at_time(ts_ms, q))

        elif cmd == "/knowledge":
            if not rest:
                print("  Usage: /knowledge <topic>")
            else:
                print(f"\n[Extracting knowledge about '{rest}'...]\n")
                print(engine.extract_knowledge(rest))

        else:
            print("\n[Searching...]\n")
            print(engine.ask(raw))

        print()


# ---------------------------------------------------------------------------
#  Workspace-level REPL
# ---------------------------------------------------------------------------

_HELP_WORKSPACE = """
  list / ls                   list all processed projects
  open <name|#>               enter a project
  process <path|URL|BVcode>   run pipeline on a new video
  <BVcode / YouTube ID / URL> download, analyse, and open directly
  help                        show this help
  quit / exit                 exit"""


def _workspace_repl(db_root: Path, open_immediately=None):
    db_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Video Understanding -- Workspace")
    print(f"  Projects dir: {db_root.resolve()}")
    print("=" * 60)

    if open_immediately is not None:
        keep = _project_repl(open_immediately, db_root)
        if not keep:
            return

    print(_HELP_WORKSPACE)
    print()

    while True:
        projects = _list_projects(db_root)

        try:
            raw = input("workspace > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "/quit", "/exit"):
            print("Bye.")
            return

        if cmd in ("help", "/help"):
            print(_HELP_WORKSPACE)

        elif cmd in ("list", "ls", "/list", "/ls"):
            print(f"\n  {len(projects)} project(s):\n")
            _print_project_list(projects)
            print()

        elif cmd in ("open", "/open"):
            if not rest:
                print(f"\n  {len(projects)} project(s):\n")
                _print_project_list(projects)
                print()
                try:
                    rest = input("  Open #/name: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if not rest:
                    continue
            target = _resolve_project(rest, projects)
            if target:
                keep = _project_repl(target, db_root)
                if not keep:
                    return

        elif cmd in ("process", "/process"):
            if not rest:
                print("  Usage: process <local path | URL | BV code>")
                continue
            ok = _run_pipeline(rest)
            if ok:
                print("\n  Pipeline complete. Use 'open' to query the results.\n")
            else:
                print("\n  Pipeline exited with errors.\n")

        else:
            # Bare input: try as project name/number, then as a new video source
            target = _resolve_project(raw, projects)
            if target:
                keep = _project_repl(target, db_root)
                if not keep:
                    return
            elif _is_video_source(raw):
                # Looks like a BV code / YouTube ID / URL -- process it
                ok = _run_pipeline(raw)
                if ok:
                    new_projects = _list_projects(db_root)
                    if new_projects:
                        keep = _project_repl(new_projects[0], db_root)
                        if not keep:
                            return
                else:
                    print("\n  Pipeline exited with errors.\n")
            else:
                print("  Unknown command. Type 'help', a project name/#, or a BV/YouTube code.")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Video Understanding CLI')
    parser.add_argument('source', nargs='?', help='Video path, URL, or BV code')
    parser.add_argument('--base-url', help='LM Studio base URL')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--vlm-model', help='Vision model name')
    parser.add_argument('--llm-model', help='Language model name')
    args = parser.parse_args()

    if args.base_url:
        cfg.LM_STUDIO_BASE_URL = args.base_url
    if args.api_key:
        cfg.LM_STUDIO_API_KEY = args.api_key
    if args.vlm_model:
        cfg.VLM_MODEL = args.vlm_model
    if args.llm_model:
        cfg.LLM_MODEL = args.llm_model


    db_root = Path(cfg.DB_DIR)

    if args.source:
        arg = args.source
        arg_path = Path(arg)

        if arg_path.is_dir() and (arg_path / 'transcript.json').exists():
            _workspace_repl(db_root, open_immediately=arg_path)
            return

        candidate = db_root / arg
        if candidate.is_dir() and (candidate / 'transcript.json').exists():
            _workspace_repl(db_root, open_immediately=candidate)
            return

        print(f"\n  '{arg}' not found as an existing project -- running pipeline first...")
        ok = _run_pipeline(arg, args.base_url, args.api_key, args.vlm_model, args.llm_model)
        if ok:
            projects = _list_projects(db_root)
            if projects:
                _workspace_repl(db_root, open_immediately=projects[0])
                return
        _workspace_repl(db_root)
    else:
        _workspace_repl(db_root)


if __name__ == '__main__':
    main()
