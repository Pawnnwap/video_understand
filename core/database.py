"""Timeline-backed store for one processed video.

This replaces the former ChromaDB + SentenceTransformer ("paraphrase") RAG
store. Querying is now agent-driven: the query engine reads ``context.md`` /
``timeline.json`` directly (feeding the whole thing when short, or letting an
OpenCode agent grep/read them when long), so ingest only writes those files —
no vector index, no embedding model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger(__name__)


class VideoDatabase:
    """Read/write the timeline + context files for one processed video."""

    def __init__(self, db_dir: Path, cfg):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._timeline_path = self.db_dir / "timeline.json"
        self._context_path = self.db_dir / "context.md"
        self._meta_path = self.db_dir / "meta.json"

    # ── ingest ─────────────────────────────────────────────────────────

    def ingest(self, segments, video_path: str, video_duration_sec: float):
        seg_dicts = []
        for s in segments:
            d = asdict(s)
            d["start_ts"] = _fmt_ms(s.start_ms)
            d["end_ts"] = _fmt_ms(s.end_ms)
            seg_dicts.append(d)
        self._write_timeline(seg_dicts, video_path, video_duration_sec)
        self._context_path.write_text(
            _build_context_text(seg_dicts, video_path, video_duration_sec),
            encoding="utf-8",
        )
        log.info(f"Database ready at {self.db_dir}")

    def _write_timeline(self, seg_dicts: list[dict], video_path: str, duration_sec: float):
        timeline = {
            "video_path": video_path,
            "duration_sec": duration_sec,
            "total_segments": len(seg_dicts),
            "segments": seg_dicts,
            "slide_index": self._build_slide_index(seg_dicts),
        }
        with open(self._timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, ensure_ascii=False, indent=2)
        log.info(f"Timeline saved -> {self._timeline_path}")

    def _build_slide_index(self, seg_dicts: list[dict]) -> list[dict]:
        index = []
        for seg in seg_dicts:
            if seg.get("is_slide_change") or seg.get("segment_id") == 0:
                index.append({
                    "timestamp_ms": seg["start_ms"],
                    "timestamp": seg.get("start_ts") or _fmt_ms(seg["start_ms"]),
                    "slide_title": seg.get("slide_title", ""),
                    "slide_type": seg.get("slide_type", ""),
                    "segment_id": seg.get("segment_id"),
                    "frame_path": seg.get("frame_path", ""),
                })
        return index

    # ── reads ──────────────────────────────────────────────────────────

    def get_segment_by_time(self, timestamp_ms: int) -> dict | None:
        timeline = self._load_timeline()
        if not timeline:
            return None
        for seg in timeline["segments"]:
            if seg["start_ms"] <= timestamp_ms <= seg["end_ms"]:
                return seg
        return None

    def get_timeline(self) -> dict | None:
        return self._load_timeline()

    def get_slide_index(self) -> list[dict]:
        timeline = self._load_timeline()
        return timeline.get("slide_index", []) if timeline else []

    def get_all_segments(self) -> list[dict]:
        timeline = self._load_timeline()
        return timeline.get("segments", []) if timeline else []

    def get_full_transcript(self) -> str:
        segs = self.get_all_segments()
        return "\n".join(f"[{s['start_ts']}] {s['transcript']}" for s in segs)

    def context_text(self) -> str:
        """Whole-video context as Markdown (for feeding a short video inline).

        Reads the cached ``context.md`` when present; otherwise rebuilds it from
        ``timeline.json`` so projects processed before this file existed still
        work without reprocessing.
        """
        if self._context_path.exists():
            return self._context_path.read_text(encoding="utf-8")
        timeline = self._load_timeline() or {}
        segs = timeline.get("segments", [])
        return _build_context_text(
            segs, timeline.get("video_path", ""), timeline.get("duration_sec", 0.0)
        ) if segs else ""

    def context_path(self) -> Path:
        """Path to ``context.md``, materializing it from the timeline if missing.

        The agent (long-video) query path greps/reads this file, so an older
        project without it self-heals on first query.
        """
        if not self._context_path.exists():
            text = self.context_text()
            if text:
                self._context_path.write_text(text, encoding="utf-8")
        return self._context_path

    def count(self) -> int:
        timeline = self._load_timeline()
        if not timeline:
            return 0
        return timeline.get("total_segments") or len(timeline.get("segments", []))

    def _load_timeline(self) -> dict | None:
        if not self._timeline_path.exists():
            return None
        with open(self._timeline_path, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load(cls, db_dir: str, cfg) -> VideoDatabase:
        db = cls(Path(db_dir), cfg)
        log.info(f"Loaded database from {db_dir} ({db.count()} segments)")
        return db


def _segment_markdown(seg: dict) -> str:
    """Render one timeline segment as a grep-friendly Markdown block."""
    title = (seg.get("slide_title") or "").strip()
    head = f"## [{seg.get('start_ts', '??:??')}–{seg.get('end_ts', '??:??')}]"
    if title:
        head += f"  {title}"
    lines = [head]

    def add(label: str, value) -> None:
        text = (value or "").strip() if isinstance(value, str) else value
        if text:
            lines.append(f"{label}{text}" if label else str(text))

    add("", seg.get("transcript"))
    bullets = [b for b in (seg.get("slide_bullets") or []) if b]
    if bullets:
        add("Slide bullets: ", "; ".join(bullets))
    add("Screen text: ", seg.get("ocr_text"))
    add("Visual: ", seg.get("scene_description"))
    add("Diagram: ", seg.get("diagram_description"))
    add("Summary: ", seg.get("fused_summary"))
    return "\n".join(lines)


def _build_context_text(seg_dicts: list[dict], video_path: str, duration_sec: float) -> str:
    header = (
        f"# Video context\n\n"
        f"Source: {video_path}\n"
        f"Duration: {duration_sec:.0f}s | Segments: {len(seg_dicts)}\n"
    )
    body = "\n\n".join(_segment_markdown(seg) for seg in seg_dicts)
    return f"{header}\n{body}\n"


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"
