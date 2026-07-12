"""process_queue.py -- sequential batch runner: download, analyse, report.

Feed it video sources (BV codes, YouTube IDs, URLs, or local paths). Each one
is processed in order: the full pipeline runs first (download + STT + vision +
fusion + DB), then /summary and /crosscheck outputs are captured in-process and
saved as one markdown report per video. OpenCode never writes files itself --
the report is assembled here from the captured strings.

Usage:
    python process_queue.py BV1iatTeGENk dQw4w9WgXcQ https://...
    python process_queue.py BV1aaa...,BV1bbb...,BV1ccc...     # commas work too
    python process_queue.py --file queue.txt
    python process_queue.py BV1... --out-dir reports --crosscheck-n 3

Queue file format: one source per line; blank lines and #-comments ignored.
A failing item is recorded and the queue moves on to the next one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config as cfg
from cli import _list_projects, _load_project
from query.crosscheck import crosscheck


def _read_queue_file(path: Path) -> list[str]:
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sources.append(line)
    return sources


def _run_pipeline(source: str) -> bool:
    """Run pipeline.py on one source with live output; True on success."""
    cmd = [sys.executable, str(Path(__file__).parent / "pipeline.py"), source]
    return subprocess.run(cmd).returncode == 0


def _find_new_project(db_root: Path, before: set[Path]) -> Path | None:
    """Locate the project the pipeline just produced.

    Prefer a directory that did not exist before the run; if the source was
    already processed (resume/cache), fall back to the most recently touched
    project.
    """
    projects = _list_projects(db_root)
    fresh = [p for p in projects if p not in before]
    if fresh:
        return max(fresh, key=lambda p: p.stat().st_mtime)
    return projects[0] if projects else None


def _build_report(source: str, db_path: Path, crosscheck_n: int) -> str:
    """Query the processed project and assemble the markdown report text."""
    loaded = _load_project(db_path)
    if loaded is None:
        raise RuntimeError(f"could not load processed project at {db_path}")
    db, engine, llm = loaded
    try:
        print("  [1/2] Generating summary ...")
        summary = engine.summarize("comprehensive")
        print("  [2/2] Crosschecking claims against the web ...")
        try:
            check = crosscheck(engine, crosscheck_n)
        except Exception as e:
            check = f"[crosscheck failed: {e}]"
        return "\n".join(
            [
                f"# {db_path.name}",
                "",
                f"- Source: `{source}`",
                f"- Processed: {datetime.now():%Y-%m-%d %H:%M}",
                f"- Segments: {db.count()}",
                f"- Project dir: `{db_path}`",
                "",
                "## Summary",
                "",
                summary,
                "",
                f"## Crosscheck (top {crosscheck_n} claims)",
                "",
                check,
                "",
            ]
        )
    finally:
        llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially process videos and save summary + crosscheck reports"
    )
    parser.add_argument("sources", nargs="*", help="BV codes, YouTube IDs, URLs, or local paths")
    parser.add_argument("--file", type=Path, help="Queue file with one source per line")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="Report output directory (default: reports/)")
    parser.add_argument("--crosscheck-n", type=int, default=5, help="Number of claims to fact-check, 1-10 (default 5)")
    args = parser.parse_args()

    # Accept any mix of space- and comma-separated codes, plus a queue file.
    raw = list(args.sources)
    if args.file:
        raw.extend(_read_queue_file(args.file))
    sources = [s for item in raw for s in item.replace(",", " ").split() if s]
    # De-duplicate while preserving order.
    sources = list(dict.fromkeys(sources))
    if not sources:
        parser.error("no sources given (positional arguments or --file)")
    crosscheck_n = max(1, min(args.crosscheck_n, 10))

    db_root = Path(cfg.DB_DIR)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []  # (source, outcome)
    for i, source in enumerate(sources, 1):
        print(f"\n{'=' * 60}\n  [{i}/{len(sources)}] {source}\n{'=' * 60}")
        started = time.time()
        try:
            before = set(_list_projects(db_root))
            if not _run_pipeline(source):
                results.append((source, "FAILED: pipeline error"))
                continue
            db_path = _find_new_project(db_root, before)
            if db_path is None:
                results.append((source, "FAILED: processed project not found"))
                continue
            report = _build_report(source, db_path, crosscheck_n)
            out_file = args.out_dir / f"{db_path.name}.md"
            out_file.write_text(report, encoding="utf-8")
            mins = (time.time() - started) / 60
            print(f"\n  Report saved: {out_file}  ({mins:.1f} min)")
            results.append((source, f"OK -> {out_file}"))
        except KeyboardInterrupt:
            results.append((source, "INTERRUPTED"))
            print("\nQueue interrupted by user.")
            break
        except Exception as e:
            results.append((source, f"FAILED: {e}"))

    print(f"\n{'=' * 60}\n  Queue finished: {len(sources)} item(s)\n{'=' * 60}")
    for source, outcome in results:
        print(f"  {source}\n      {outcome}")
    if any(not o.startswith("OK") for _, o in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
