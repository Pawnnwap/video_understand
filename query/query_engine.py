"""Agent-driven query operations for a processed video database.

Querying no longer uses RAG retrieval. Instead the whole video context
(``context.md``) is fed to the model directly when it is short enough, and when
it is long the work is delegated to the read-only ``video-qa`` OpenCode agent,
which greps/reads the project's context files to find what it needs — the way a
coding agent searches a repository.

Interactive command dispatch lives exclusively in :mod:`cli`. This module
contains only reusable query operations so CLI, pipeline, and one-shot callers
cannot drift into separate command implementations.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from core.vision.opencode_vlm import AgentTimeout
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
        self._ctx_limit: int | None = None

    def ask(self, question: str) -> str:
        """Answer a question or produce a topic deep dive over the whole video.

        Short videos are answered from their entire context in one call; long
        videos are handed to the ``video-qa`` agent, which searches the context
        files itself. A single code path serves both narrow questions (direct,
        precise answer) and broad topic requests (thorough coverage); the prompt
        lets the model pick the appropriate depth.
        """
        context = self.db.context_text()
        if not context.strip():
            return f"No video content available to answer: {question}"
        request = (
            "Answer a specific question directly and precisely; for a broad "
            "topic, write a thorough briefing covering the relevant details, "
            "examples, caveats, and visual material. Cite [MM:SS] timestamps."
        )
        if self._fits_inline(context):
            prompt = (
                f"VIDEO CONTEXT (whole video):\n\n{context}\n\n"
                f"Request: {question}\n\n{request}"
            )
            return self._llm(prompt, max_tokens=1200)
        return self._agent_answer(self._agent_prompt(f"{question}\n\n{request}"))

    def summarize(self, style: str = "comprehensive") -> str:
        """Create a headline, brief, or comprehensive video summary.

        Parameterless per style, so the result is cached in the project dir;
        repeat calls cost no model engagement until the video is reprocessed.
        """
        cached = self._cache_get(f"summary:{style}")
        if cached is not None:
            return cached
        instructions = {
            "headline": "Write one accurate headline of at most 20 words.",
            "brief": "Write a concise 3-5 sentence overview.",
            "comprehensive": "Write a structured, comprehensive summary of the main topics, evidence, and conclusions.",
        }
        instruction = instructions.get(style, instructions["comprehensive"])
        context = self.db.context_text()
        if not context.strip():
            return "No video content available to summarize."
        if self._fits_inline(context):
            result = self._llm(
                f"VIDEO CONTEXT (whole video):\n\n{context}\n\n{instruction}",
                max_tokens=1500,
            )
        else:
            result = self._agent_answer(
                self._agent_prompt(f"Summarize the whole video. {instruction}", whole=True)
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

    # ── short/long routing ─────────────────────────────────────────────

    def _context_limit(self) -> int:
        """Text model's context window (cached), used to size the threshold."""
        if self._ctx_limit is None:
            fallback = int(getattr(self.cfg, "QA_CONTEXT_LIMIT_FALLBACK", 128000))
            overrides = getattr(self.cfg, "MODEL_CONTEXT_LIMITS", None)
            getter = getattr(self.llm, "text_context_limit", None)
            self._ctx_limit = getter(fallback, overrides) if getter else fallback
        return self._ctx_limit

    def _fits_inline(self, context: str) -> bool:
        """True when the whole context fits the configured token fraction."""
        fraction = float(getattr(self.cfg, "QA_CONTEXT_TOKEN_FRACTION", 0.10))
        return _estimate_tokens(context) < fraction * self._context_limit()

    def _agent_prompt(self, request: str, whole: bool = False) -> str:
        ctx = self.db.context_path()
        guidance = (
            "Read through the whole context file (in chunks if it is large) and "
            "synthesize across it."
            if whole else
            "Use grep to find the timestamps and topics relevant to the request, "
            "then read to pull the surrounding context; widen or refine your "
            "search until you can answer well."
        )
        return (
            "You answer a request about ONE processed video.\n"
            f"Context file (Markdown, one block per timestamped segment):\n  {ctx}\n"
            f"Project directory (timeline.json, frames, etc.):\n  {self.db.db_dir}\n\n"
            f"{guidance}\n"
            "Answer ONLY from these files; say so if they are insufficient. "
            "Reply in the user's language and cite [MM:SS] timestamps.\n\n"
            f"Request: {request}"
        )

    def _agent_answer(self, prompt: str) -> str:
        """Run the read-only video-qa agent over the project's context files."""
        agent = getattr(self.cfg, "QA_AGENT", "video-qa")
        variant = getattr(self.cfg, "LLM_VARIANT", None)
        idle = int(getattr(self.cfg, "QA_IDLE_TIMEOUT_S", 300))
        monitored = getattr(self.llm, "call_text_monitored", None)
        progress = _AgentProgress() if monitored else None
        try:
            if monitored:
                return self.llm.call_text_monitored(
                    prompt, variant=variant, agent=agent,
                    on_progress=progress, idle_timeout_s=idle,
                )
            return self.llm.call_text(prompt, variant=variant, agent=agent)
        except AgentTimeout as exc:
            log.error("video-qa agent %s-stopped: %s", exc.reason, exc)
            if exc.partial_text:
                return f"{exc.partial_text}\n\n_(answer truncated — agent {exc.reason}-stopped)_"
            return f"[agent stopped: {exc}]"
        except TimeoutError:
            log.error("video-qa agent idle-timeout after %ss", idle)
            return f"[agent stopped: no activity for {idle}s]"
        except Exception as exc:
            log.exception("video-qa agent failed")
            return f"[agent error: {exc}]"
        finally:
            if progress:
                progress.done()

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
        if not value or value.startswith(("[LLM error", "[agent error", "[agent stopped")):
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


class _AgentProgress:
    """Single-line progress for one video-qa agent session."""

    def __init__(self, label: str = "video-qa"):
        self.label = label
        self.started = time.time()

    def __call__(self, stats: dict) -> None:
        elapsed = int(time.time() - self.started)
        line = (
            f"  [{self.label}] tools {stats.get('tools', 0)}"
            f" | {elapsed // 60}:{elapsed % 60:02d}"
            f" | {stats.get('last_tool', '')}"
        )
        sys.stdout.write("\r" + line[:110].ljust(110))
        sys.stdout.flush()

    def done(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _estimate_tokens(text: str) -> int:
    """Rough token count for mixed CJK/Latin text.

    CJK characters tokenize at roughly one token each; other text averages ~4
    characters per token. The estimate deliberately leans high so the inline
    (whole-context) path is only taken when the content is safely small.
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
        or 0x3000 <= cp <= 0x303F  # CJK punctuation
        or 0xFF00 <= cp <= 0xFFEF  # full-width forms
    )


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
