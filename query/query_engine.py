"""
query/query_engine.py
Uses the VideoDatabase + local LLM to answer natural language questions
about the video. All queries go through RAG over the vector store.
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Optional

from utils.retry import RetryConfig, retry_sync

_QUERY_RETRY = RetryConfig(max_attempts=4, base_delay_s=2.0, max_delay_s=20.0)

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
你是一个视频理解助手。你可以访问从视频中提取的结构化知识库，包含文字转录、幻灯片内容、图表描述和视觉描述。

请根据提供的上下文准确回答问题。
如果上下文中没有相关答案，请明确说明。
引用视频中的特定时刻时，请使用[分:秒]格式包含时间戳。
使用与用户相同的语言进行回应。"""


class QueryEngine:
    """
    Wraps a VideoDatabase and a local LLM client to answer
    natural language questions about a processed video.
    """

    def __init__(self, db, client, cfg):
        self.db     = db
        self.client = client
        self.cfg    = cfg

    # ──────────────────────────────────────────────────────────────────────
    #  High-level query methods
    # ──────────────────────────────────────────────────────────────────────

    def ask(self, question: str, n_context: int = 6) -> str:
        """
        General-purpose RAG query.
        Retrieves the most relevant segments and asks the LLM.
        """
        hits = self.db.search(question, n_results=n_context)
        context = _format_context(hits)
        prompt = f"{context}\n\nQuestion: {question}"
        return self._llm(prompt)

    def summarize(self, style: str = "comprehensive") -> str:
        """
        Generate a full video summary.
        style: "brief" | "comprehensive" | "bullet"
        """
        all_segs = self.db.get_all_segments()
        # Sample evenly to fit context window
        sampled  = _sample_segments(all_segs, max_tokens_budget=6000)
        context  = "\n\n".join(
            f"[{s['start_ts']}] {s['fused_summary']}" for s in sampled
        )

        style_instructions = {
            "brief":         "请用3-5句话写一个简洁的概述。",
            "comprehensive": "请写一个结构化摘要，涵盖主要话题、关键要点和结论，使用章节标题组织内容。",
            "bullet":        "请用要点列表形式总结所有关键话题和核心要点。",
        }
        instruction = style_instructions.get(style, style_instructions["comprehensive"])

        prompt = f"以下是视频的时间线内容：\n\n{context}\n\n{instruction}"
        return self._llm(prompt, max_tokens=1500)

    def get_topic_outline(self) -> str:
        """Extract a structured topic outline from the slide index + summaries."""
        slides = self.db.get_slide_index()
        if not slides:
            return self.ask("What are the main topics covered in this video?")

        slide_list = "\n".join(
            f"[{s['timestamp']}] {s['slide_title'] or '(no title)'}"
            for s in slides
        )
        prompt = (
            f"以下是演示视频的幻灯片时间线：\n\n{slide_list}\n\n"
            "请创建一个清晰的主题大纲，展示演示文稿的结构和内容进展。"
        )
        return self._llm(prompt)

    def query_at_time(self, timestamp_ms: int, question: str) -> str:
        """Answer a question specifically about what was happening at a given time."""
        seg = self.db.get_segment_by_time(timestamp_ms)
        if not seg:
            return f"No segment found at {timestamp_ms}ms."
        context = _format_single_segment(seg)
        prompt  = f"{context}\n\n关于这个时刻的问题：{question}"
        return self._llm(prompt)

    def extract_knowledge(self, topic: str) -> str:
        """
        Deep extraction: retrieve all segments mentioning a topic,
        then ask the LLM to synthesize the complete knowledge.
        """
        hits = self.db.search(topic, n_results=10)
        if not hits:
            return f"No content found about: {topic}"
        context = _format_context(hits)
        prompt = (
            f"{context}\n\n"
            f"根据以上内容，请对以下主题进行完整详细的解释：{topic}\n"
            "请包含所有相关细节、示例以及图表描述中提及的内容。"
        )
        return self._llm(prompt, max_tokens=1200)

    def find_slides_about(self, topic: str) -> List[Dict]:
        """Return all slide change points related to a topic."""
        hits = self.db.search(topic, n_results=8)
        return [
            {
                "timestamp":    h["timestamp"],
                "slide_title":  h["slide_title"],
                "summary":      h["fused_summary"],
                "frame_path":   h["frame_path"],
            }
            for h in hits
            if h.get("slide_title")
        ]

    def get_full_transcript(self) -> str:
        return self.db.get_full_transcript()

    # ──────────────────────────────────────────────────────────────────────
    #  Interactive REPL
    # ──────────────────────────────────────────────────────────────────────

    def repl(self):
        """
        Launch an interactive query session in the terminal.
        Special commands:
          /summary          — comprehensive summary
          /outline          — topic outline
          /slides           — list all slide changes
          /transcript       — full transcript
          /at MM:SS <q>     — query at specific timestamp
          /knowledge <topic>— deep extraction on a topic
          /quit             — exit
        """
        print("\n" + "═"*60)
        print("  Video Understanding Query Engine")
        print(f"  Database: {self.db.db_dir}")
        print(f"  Segments: {self.db.count()}")
        print("═"*60)
        print("Commands: /summary  /outline  /slides  /transcript")
        print("          /at MM:SS <question>  /knowledge <topic>  /quit")
        print("Or just type any question.\n")

        while True:
            try:
                user_input = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("Bye.")
                break

            elif user_input == "/summary":
                print("\n[Generating summary…]\n")
                print(self.summarize("comprehensive"))

            elif user_input == "/outline":
                print("\n[Building outline…]\n")
                print(self.get_topic_outline())

            elif user_input == "/slides":
                slides = self.db.get_slide_index()
                print(f"\n{len(slides)} slide changes detected:\n")
                for s in slides:
                    print(f"  {s['timestamp']}  {s['slide_title'] or '(no title)'}")

            elif user_input == "/transcript":
                print("\n" + self.get_full_transcript())

            elif user_input.startswith("/at "):
                parts = user_input[4:].split(" ", 1)
                if len(parts) >= 1:
                    ts_ms = _parse_timestamp(parts[0])
                    q     = parts[1] if len(parts) > 1 else "What is happening here?"
                    print(f"\n[Querying at {parts[0]}…]\n")
                    print(self.query_at_time(ts_ms, q))

            elif user_input.startswith("/knowledge "):
                topic = user_input[11:].strip()
                print(f"\n[Extracting knowledge about '{topic}'…]\n")
                print(self.extract_knowledge(topic))

            else:
                print("\n[Searching…]\n")
                print(self.ask(user_input))

            print()

    # ──────────────────────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────────────────────

    def _llm(self, prompt: str, max_tokens: int = 800) -> str:
        def _call():
            return self.client.chat.completions.create(
                model      = self.cfg.LLM_MODEL,
                messages   = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens = max_tokens,
                temperature= 0.3,
            ).choices[0].message.content.strip()

        try:
            return retry_sync(_call, cfg=_QUERY_RETRY, label="query_llm")
        except Exception as e:
            log.error(f"LLM call failed after retries: {e}")
            return f"[LLM error: {e}]"


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_context(hits: List[Dict]) -> str:
    parts = []
    for h in hits:
        parts.append(
            f"[{h['timestamp']}]（相关度 {h['score']:.2f}）\n"
            f"摘要：{h['fused_summary']}\n"
            f"转录：{h['transcript']}\n"
            + (f"幻灯片标题：{h['slide_title']}\n" if h.get("slide_title") else "")
        )
    return "视频相关内容：\n\n" + "\n---\n".join(parts)


def _format_single_segment(seg: Dict) -> str:
    return (
        f"片段时间：[{seg.get('start_ts', '??:??')}]\n"
        f"转录内容：{seg.get('transcript', '')}\n"
        f"视觉摘要：{seg.get('fused_summary', '')}\n"
        f"幻灯片标题：{seg.get('slide_title', '')}\n"
        f"幻灯片要点：{'; '.join(seg.get('slide_bullets', []))}\n"
        f"屏幕文字：{seg.get('ocr_text', '')[:300]}\n"
        f"图表描述：{seg.get('diagram_description', '')}\n"
    )


def _sample_segments(segments: List[Dict], max_tokens_budget: int = 6000) -> List[Dict]:
    """Evenly sample segments to stay within context budget."""
    if not segments:
        return []
    avg_tokens = 80
    max_segs   = max_tokens_budget // avg_tokens
    if len(segments) <= max_segs:
        return segments
    step = len(segments) / max_segs
    return [segments[int(i * step)] for i in range(max_segs)]


def _parse_timestamp(ts_str: str) -> int:
    """Parse MM:SS or HH:MM:SS to milliseconds."""
    parts = ts_str.split(":")
    try:
        if len(parts) == 2:
            return (int(parts[0]) * 60 + int(parts[1])) * 1000
        elif len(parts) == 3:
            return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
    except ValueError:
        pass
    return 0
