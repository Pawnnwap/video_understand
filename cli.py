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
    <question>               ask anything in natural language (recommended)
    /summary [style]        whole-video summary: comprehensive (default), brief, headline
    /outline                topic outline from slides
    /slides                 list all slide changes with timestamps
    /transcript             full spoken transcript
    /at MM:SS [question]    what was on screen at a specific moment
    /open <name|#>          switch to a different project
    /help                   show this help
    /quit or /back          return to workspace (quit again there to exit)
    <anything else>         semantic search + RAG answer
"""

from __future__ import annotations

import argparse
import io
import sys

import config as cfg
from core.database import VideoDatabase
from downloader import _expand_short_code, is_url

# Force UTF-8 on Windows consoles so CJK / box-drawing chars render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import logging
import os
import subprocess
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


logging.basicConfig(level=logging.WARNING)   # quiet in interactive mode

from query.crosscheck import crosscheck
from query.query_engine import QueryEngine, _parse_timestamp


def _is_video_source(s: str) -> bool:
    """Return True if s looks like something the pipeline can process."""
    return is_url(s) or (_expand_short_code(s) is not None)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_llm():
    """Build the opencode text-LLM (pure mode) for the query engine.

    Spawns a local `opencode serve` subprocess and reuses its single session
    for every ``call_text`` invocation.  Caller is responsible for closing it
    (``llm.close()``) on shutdown.
    """
    from core.vision.opencode_vlm import OpencodeVLM
    return OpencodeVLM(
        model=getattr(cfg, "LLM_MODEL", cfg.VLM_MODEL),
        port=cfg.OPENCODE_SERVER_PORT,
        variant=getattr(cfg, "LLM_VARIANT", None),
    )


# ---------------------------------------------------------------------------
#  Opencode model overrides (shared by cli.py and process_queue.py)
# ---------------------------------------------------------------------------

def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Register the opencode model/variant/port override flags on a parser."""
    parser.add_argument("--vlm-model", help="opencode vision model id (default opencode/mimo-v2.5-free)")
    parser.add_argument("--vlm-variant", help="opencode vision model variant (low|medium|high, or none)")
    parser.add_argument(
        "--text-model", "--llm-model", dest="text_model",
        help="opencode text model for fusion, queries, and crosscheck",
    )
    parser.add_argument(
        "--text-variant", "--llm-variant", dest="text_variant",
        help="opencode text-model reasoning variant (low|medium|high, or none)",
    )
    parser.add_argument(
        "--no-thinking", "--no-variant", dest="no_thinking", action="store_true",
        help="disable reasoning/thinking on both models (for custom models with no thinking level)",
    )
    parser.add_argument("--opencode-port", type=int, help="Fixed port for the opencode server (0 = random)")


def apply_model_overrides(args: argparse.Namespace) -> None:
    """Write any supplied model overrides into the global config.

    A variant of ``none``/``off``/`""` (or the ``--no-thinking`` switch)
    disables reasoning: the variant is stored empty and later omitted from
    opencode requests, which custom models without a thinking level require.
    """
    if getattr(args, "vlm_model", None):
        cfg.VLM_MODEL = args.vlm_model
    if getattr(args, "text_model", None):
        cfg.LLM_MODEL = args.text_model
    if getattr(args, "no_thinking", False):
        cfg.VLM_VARIANT = cfg.LLM_VARIANT = ""
    else:
        if getattr(args, "vlm_variant", None) is not None:
            cfg.VLM_VARIANT = cfg.normalize_variant(args.vlm_variant)
        if getattr(args, "text_variant", None) is not None:
            cfg.LLM_VARIANT = cfg.normalize_variant(args.text_variant)
    if getattr(args, "opencode_port", None) is not None:
        cfg.OPENCODE_SERVER_PORT = args.opencode_port


def pipeline_model_flags() -> list[str]:
    """Effective model settings as pipeline.py CLI flags.

    Forwarded to every pipeline subprocess so it runs with the same models
    this process resolved, whether from defaults or CLI overrides. A disabled
    (empty) variant is forwarded as the ``none`` sentinel so the subprocess
    also omits it rather than falling back to its own default.
    """
    vlm_v = getattr(cfg, "VLM_VARIANT", None)
    txt_v = getattr(cfg, "LLM_VARIANT", None)
    flags = [
        "--vlm-model", cfg.VLM_MODEL,
        "--vlm-variant", vlm_v if vlm_v else "none",
        "--text-model", getattr(cfg, "LLM_MODEL", cfg.VLM_MODEL),
        "--text-variant", txt_v if txt_v else "none",
    ]
    if cfg.OPENCODE_SERVER_PORT:
        flags += ["--opencode-port", str(cfg.OPENCODE_SERVER_PORT)]
    return flags


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
    """Load DB + engine. Returns (db, engine, llm) or None on failure.

    The ``llm`` is an ``OpencodeVLM`` instance that must be closed by the
    caller via ``llm.close()`` once the REPL exits.
    """
    try:
        db = VideoDatabase.load(str(db_path), cfg)
    except Exception as e:
        print(f"  Error loading database: {e}")
        return None
    if db.count() == 0:
        print("  Database is empty -- run 'process' first.")
        return None
    llm = _build_llm()
    engine = QueryEngine(db, llm, cfg)
    return db, engine, llm


def _run_pipeline(source: str) -> bool:
    """Run pipeline.py on source, inheriting stdio for live output.

    The current cfg model settings (defaults or CLI overrides) are forwarded
    so the subprocess uses the same models as this process.
    """
    print(f"\n  Launching pipeline for: {source}")
    print("  (runs in the foreground -- please wait)\n")
    cmd = [
        sys.executable, str(Path(__file__).parent / "pipeline.py"),
        source, *pipeline_model_flags(),
    ]
    r = subprocess.run(cmd)
    return r.returncode == 0


# ---------------------------------------------------------------------------
#  Project-level REPL
# ---------------------------------------------------------------------------

_HELP_PROJECT = """
  Ask anything (recommended): type a normal question with no command prefix.

  /summary [style]      whole-video summary: comprehensive (default), brief, headline
  /outline              topic outline from slides
  /slides               all slide changes with timestamps
  /transcript           full spoken transcript
  /at MM:SS [question]  what was on screen at this moment
  /crosscheck [n]       fact-check top N claims against the web (default 5)
  /open <name|#>        switch to another project
  /help                 show this help
  /quit or /back        return to workspace (quit again there to exit)"""


def _project_repl(db_path: Path, db_root: Path) -> tuple[str, Path | None]:
    """Inner REPL for one project.

    Returns ``("open", path)`` to switch projects, ``("workspace", None)``
    to return to the workspace, or ``("exit", None)`` to quit.  Keeping the
    switch request outside this function ensures the current OpenCode server is
    closed before a replacement project session starts.
    """
    result = _load_project(db_path)
    if result is None:
        return "workspace", None

    db, engine, llm = result
    name = db_path.name
    short = name[:52] + "..." if len(name) > 55 else name

    print("\n" + "=" * 60)
    print(f"  Project : {short}")
    print(f"  Segments: {db.count()}")
    print("=" * 60)
    print(_HELP_PROJECT)
    print()

    try:
        while True:
            try:
                raw = input(f"[{short[:35]}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return "exit", None

            if not raw:
                continue

            parts = raw.split(None, 1)
            cmd = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""

            # -- navigation --
            # quit is staged: inside a project it returns to the workspace;
            # only the workspace prompt exits the CLI entirely.
            if cmd in ("/quit", "/exit", "quit", "exit", "/back", "/workspace", "back"):
                return "workspace", None

            if cmd == "/help":
                print(_HELP_PROJECT)

            elif cmd == "/open":
                if not rest:
                    print("  Usage: /open <name or number>")
                    continue
                projects = _list_projects(db_root)
                target = _resolve_project(rest, projects)
                if target:
                    return "open", target

            # -- query commands --
            elif cmd == "/summary":
                style = rest.lower() or "comprehensive"
                if style not in ("comprehensive", "brief", "headline"):
                    print("  Usage: /summary [headline|brief]  (default: comprehensive)")
                else:
                    print(f"\n[Generating {style} summary...]\n")
                    print(engine.summarize(style))

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

            elif cmd == "/crosscheck":
                try:
                    n = int(rest) if rest else 5
                    n = max(1, min(n, 10))
                except ValueError:
                    print("  Usage: /crosscheck [n]  (n = number of claims, 1-10; default 5)")
                    print()
                    continue
                print(f"\n[Crosschecking top {n} claims against the web...]\n")
                print(crosscheck(engine, n))

            else:
                print("\n[Searching...]\n")
                print(engine.ask(raw))

            print()
    finally:
        try:
            llm.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Workspace-level REPL
# ---------------------------------------------------------------------------

def _open_project_flow(db_path: Path, db_root: Path) -> bool:
    """Open a project and process any in-session project switch requests.

    Returns ``False`` only when the user chose to exit the whole CLI.
    """
    current = db_path
    while True:
        action, target = _project_repl(current, db_root)
        if action == "exit":
            return False
        if action == "open" and target is not None:
            current = target
            continue
        return True

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
        if not _open_project_flow(open_immediately, db_root):
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
                if not _open_project_flow(target, db_root):
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
                if not _open_project_flow(target, db_root):
                    return
            elif _is_video_source(raw):
                # Looks like a BV code / YouTube ID / URL -- process it
                ok = _run_pipeline(raw)
                if ok:
                    new_projects = _list_projects(db_root)
                    if new_projects:
                        if not _open_project_flow(new_projects[0], db_root):
                            return
                else:
                    print("\n  Pipeline exited with errors.\n")
            else:
                print("  Unknown command. Type 'help', a project name/#, or a BV/YouTube code.")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video Understanding CLI")
    parser.add_argument("source", nargs="?", help="Video path, URL, or BV code")
    add_model_args(parser)
    args = parser.parse_args()

    apply_model_overrides(args)

    db_root = Path(cfg.DB_DIR)

    if args.source:
        arg = args.source
        arg_path = Path(arg)

        if arg_path.is_dir() and (arg_path / "transcript.json").exists():
            _workspace_repl(db_root, open_immediately=arg_path)
            return

        candidate = db_root / arg
        if candidate.is_dir() and (candidate / "transcript.json").exists():
            _workspace_repl(db_root, open_immediately=candidate)
            return

        print(f"\n  '{arg}' not found as an existing project -- running pipeline first...")
        ok = _run_pipeline(arg)
        if ok:
            projects = _list_projects(db_root)
            if projects:
                _workspace_repl(db_root, open_immediately=projects[0])
                return
        _workspace_repl(db_root)
    else:
        _workspace_repl(db_root)


if __name__ == "__main__":
    main()
