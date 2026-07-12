"""RAG-backed query operations for a processed video database.

Interactive command dispatch lives exclusively in :mod:`cli`. This module
contains only reusable query operations so CLI, pipeline, and one-shot callers
cannot drift into separate command implementations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from utils.retry import RetryConfig, retry_sync

_QUERY_RETRY = RetryConfig(max_attempts=4, base_delay_s=2.0, max_delay_s=20.0)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a video-understanding assistant. Answer only from \
the supplied video context (transcripts, slide text, chart descriptions, \
visual summaries); say when it is insufficient. Reply in the user's language \
and cite video timestamps when available."""


class QueryEngine:
    """Answer questions and create views over one processed video database."""

    def __init__(self, db, llm, cfg):
        self.db = db
        self.llm = llm
        self.cfg = cfg

    def ask(self, question: str, n_context: int = 10) -> str:
        """Answer a question or produce a topic deep dive from a wide RAG context.

        A single code path serves both narrow questions (direct, precise
        answer) and broad topic requests (thorough coverage and synthesis);
        the prompt lets the model pick the appropriate depth.
        """
        hits = self.db.search(question, n_results=n_context)
        if not hits:
            return f"No relevant content found for: {question}"
        prompt = (
            f"{_format_context(hits)}\n\n"
            f"Request: {question}\n\n"
            "Answer a specific question directly and precisely; for a broad "
            "topic, write a thorough briefing covering the relevant details, "
            "examples, caveats, and visual material in the context."
        )
        return self._llm(prompt, max_tokens=1200)

    def summarize(self, style: str = "comprehensive") -> str:
        """Create a headline, brief, or comprehensive video summary.

        Parameterless per style, so the result is cached in the project dir;
        repeat calls cost no model engagement until the video is reprocessed.
        """
        cached = self._cache_get(f"summary:{style}")
        if cached is not None:
            return cached
        sampled = _sample_segments(self.db.get_all_segments(), max_tokens_budget=6000)
        context = "\n\n".join(
            f"[{segment.get('start_ts', '??:??')}] {segment.get('fused_summary', '')}"
            for segment in sampled
        )
        instructions = {
            "headline": "Write one accurate headline of at most 20 words.",
            "brief": "Write a concise 3-5 sentence overview.",
            "comprehensive": "Write a structured, comprehensive summary of the main topics, evidence, and conclusions.",
        }
        result = self._llm(
            f"VIDEO TIMELINE:\n{context}\n\n{instructions.get(style, instructions['comprehensive'])}",
            max_tokens=1500,
        )
        self._cache_put(f"summary:{style}", result)
        return result

    def get_topic_outline(self) -> str:
        """Build an outline from indexed slide changes when available.

        Parameterless, so the result is cached like :meth:`summarize`.
        """
        cached = self._cache_get("outline")
        if cached is not None:
            return cached
        slides = self.db.get_slide_index()
        if not slides:
            result = self.ask("What are the main topics covered in this video?")
        else:
            slide_list = "\n".join(
                f"[{slide['timestamp']}] {slide['slide_title'] or '(no title)'}"
                for slide in slides
            )
            result = self._llm(
                "Create a clear topic outline from these slide changes. Show the progression "
                f"of the presentation.\n\nSLIDE TIMELINE:\n{slide_list}"
            )
        self._cache_put("outline", result)
        return result

    def query_at_time(self, timestamp_ms: int, question: str) -> str:
        """Answer a question about the segment containing a timestamp."""
        segment = self.db.get_segment_by_time(timestamp_ms)
        if not segment:
            return f"No segment found at {timestamp_ms}ms."
        return self._llm(
            f"{_format_single_segment(segment)}\n\nQuestion about this moment: {question}"
        )

    def get_full_transcript(self) -> str:
        return self.db.get_full_transcript()

    # ── parameterless-result cache (stored inside the project dir) ─────

    def _cache_file(self) -> Path:
        return Path(self.db.db_dir) / "query_cache.json"

    def _cache_load(self) -> dict:
        """Load the cache, dropping it when the video was reprocessed."""
        try:
            data = json.loads(self._cache_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict) or data.get("_segments") != self.db.count():
            return {}
        return data

    def _cache_get(self, key: str) -> str | None:
        value = self._cache_load().get(key)
        if value is not None:
            log.info("query cache hit: %s", key)
        return value

    def _cache_put(self, key: str, value: str) -> None:
        if not value or value.startswith("[LLM error"):
            return  # never cache failures
        data = self._cache_load()
        data["_segments"] = self.db.count()
        data[key] = value
        try:
            self._cache_file().write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("query cache write failed: %s", exc)

    def _llm(self, prompt: str, max_tokens: int = 800) -> str:
        variant = getattr(self.cfg, "LLM_VARIANT", None)

        def _call() -> str:
            payload = f"{SYSTEM_PROMPT}\n\n---\n\n{prompt}"
            return self.llm.call_text(payload, variant=variant)

        try:
            return retry_sync(_call, cfg=_QUERY_RETRY, label="query_llm")
        except Exception as exc:
            log.error("LLM call failed after retries: %s", exc)
            return f"[LLM error: {exc}]"


def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant video segments were retrieved."
    entries = []
    for hit in hits:
        entries.append(
            f"[{hit.get('timestamp', '??:??')}] relevance={hit.get('score', 0):.2f}\n"
            f"Summary: {hit.get('fused_summary', '')}\n"
            f"Transcript: {hit.get('transcript', '')}\n"
            + (f"Slide title: {hit['slide_title']}\n" if hit.get("slide_title") else "")
        )
    return "RELEVANT VIDEO CONTEXT:\n\n" + "\n---\n".join(entries)


def _format_single_segment(segment: dict) -> str:
    return (
        f"Segment time: [{segment.get('start_ts', '??:??')}]\n"
        f"Transcript: {segment.get('transcript', '')}\n"
        f"Visual summary: {segment.get('fused_summary', '')}\n"
        f"Slide title: {segment.get('slide_title', '')}\n"
        f"Slide bullets: {'; '.join(segment.get('slide_bullets', []))}\n"
        f"Screen text: {segment.get('ocr_text', '')[:300]}\n"
        f"Diagram description: {segment.get('diagram_description', '')}"
    )


def _sample_segments(segments: list[dict], max_tokens_budget: int = 6000) -> list[dict]:
    """Evenly sample a timeline to stay within the model context budget."""
    if not segments:
        return []
    max_segments = max_tokens_budget // 80
    if len(segments) <= max_segments:
        return segments
    step = len(segments) / max_segments
    return [segments[int(index * step)] for index in range(max_segments)]


def _parse_timestamp(ts_str: str) -> int:
    """Parse MM:SS or HH:MM:SS as milliseconds; return zero when invalid."""
    parts = ts_str.split(":")
    try:
        if len(parts) == 2:
            return (int(parts[0]) * 60 + int(parts[1])) * 1000
        if len(parts) == 3:
            return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
    except ValueError:
        pass
    return 0
