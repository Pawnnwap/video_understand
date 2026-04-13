"""
core/fusion.py — Phase 3
Merges STT transcript segments with VLM frame analyses into coherent,
temporally-grounded segment records. Each record becomes one unit in the DB.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from utils.retry import RetryConfig, retry_sync

_FUSION_RETRY = RetryConfig(max_attempts=4, base_delay_s=2.0, max_delay_s=20.0)

log = logging.getLogger(__name__)


@dataclass
class FusedSegment:
    """One storable unit of understanding — sentence(s) + visual context + summary."""
    segment_id:         int
    start_ms:           int
    end_ms:             int
    duration_sec:       float

    # Speech
    transcript:         str
    sentence_ids:       List[int] = field(default_factory=list)

    # Visual (best frame for this segment)
    slide_title:        str = ""
    slide_bullets:      List[str] = field(default_factory=list)
    slide_type:         str = ""
    ocr_text:           str = ""
    scene_description:  str = ""
    diagram_description:str = ""
    visual_delta:       str = ""
    frame_timestamp_ms: int = 0
    frame_path:         str = ""

    # Visual continuity
    is_slide_change:    bool = False
    is_new_topic:       bool = False

    # Fused understanding (LLM-generated)
    fused_summary:      str = ""

    # For vector store
    embedding_text:     str = ""   # what we actually embed


# ─────────────────────────────────────────────────────────────────────────────
#  Frame-to-sentence matching
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_frame(sentence_ids: List[int], analyses):
    """
    Given a list of sentence IDs for this segment, pick the most
    informative frame associated with any of those sentences.
    Priority: slide_change > has_text_content > any
    """
    candidates = [a for a in analyses if a.sentence_id in sentence_ids]
    if not candidates:
        return None

    # Prefer frames that detected a slide or content change
    for priority in ["slide_change", "new_content", "major_change"]:
        hits = [c for c in candidates if c.visual_delta == priority]
        if hits:
            return hits[0]

    # Prefer frames with OCR text (slide content)
    text_frames = [c for c in candidates if c.has_text_content]
    if text_frames:
        return text_frames[0]

    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion prompt
# ─────────────────────────────────────────────────────────────────────────────

FUSION_PROMPT = """\
你正在分析一段视频录像的片段。以下是该片段的语音转录内容和视觉画面描述。

语音转录：
{transcript}

视觉内容：
- 画面描述：{scene}
- 幻灯片标题：{slide_title}
- 幻灯片要点：{slide_bullets}
- 屏幕文字（OCR）：{ocr_text}
- 图表/图形：{diagram}
- 画面变化：{delta}

请用2-4句话写一段简洁的中文摘要，将说话内容与画面内容融合在一起。
将指示性表达（如"如图所示"、"这里"、"如下"）替换为实际显示的内容。
只关注知识和信息内容。"""


def _build_fusion_prompt(seg: FusedSegment) -> str:
    bullets_str = "; ".join(seg.slide_bullets) if seg.slide_bullets else "—"
    return FUSION_PROMPT.format(
        transcript  = seg.transcript or "—",
        scene       = seg.scene_description or "—",
        slide_title = seg.slide_title or "—",
        slide_bullets = bullets_str,
        ocr_text    = (seg.ocr_text[:400] + "…") if len(seg.ocr_text) > 400 else (seg.ocr_text or "—"),
        diagram     = seg.diagram_description or "—",
        delta       = seg.visual_delta or "—",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main fusion logic
# ─────────────────────────────────────────────────────────────────────────────

def fuse(sentences, analyses, client, cfg) -> List[FusedSegment]:
    """
    Group sentences into chunks, attach best visual frame per chunk,
    call LLM for fused summary.
    """
    analyses_list = list(analyses)
    n             = len(sentences)
    chunk_size    = cfg.FUSION_SEGMENT_SIZE
    fused_segments: List[FusedSegment] = []

    log.info(f"Fusing {n} sentences in chunks of {chunk_size} …")

    for chunk_start in range(0, n, chunk_size):
        chunk = sentences[chunk_start : chunk_start + chunk_size]
        sid   = chunk_start // chunk_size

        combined_text = " ".join(s.text for s in chunk)
        sentence_ids  = [s.id for s in chunk]
        start_ms      = chunk[0].start_ms
        end_ms        = chunk[-1].end_ms

        best_frame = _find_best_frame(sentence_ids, analyses_list)

        seg = FusedSegment(
            segment_id   = sid,
            start_ms     = start_ms,
            end_ms       = end_ms,
            duration_sec = (end_ms - start_ms) / 1000.0,
            transcript   = combined_text,
            sentence_ids = sentence_ids,
        )

        if best_frame:
            seg.slide_title         = best_frame.slide_title
            seg.slide_bullets       = best_frame.slide_bullets or best_frame.ocr_lines[:6]
            seg.slide_type          = best_frame.slide_type
            seg.ocr_text            = best_frame.ocr_text
            seg.scene_description   = best_frame.scene_description
            seg.diagram_description = best_frame.diagram_description
            seg.visual_delta        = best_frame.visual_delta
            seg.frame_timestamp_ms  = best_frame.timestamp_ms
            seg.frame_path          = best_frame.frame_path
            seg.is_slide_change     = best_frame.visual_delta in ("slide_change", "major_change")

        # ── LLM fusion call (with retry) ──────────────────────────────────
        try:
            prompt = _build_fusion_prompt(seg)

            _timeout = getattr(cfg, "LLM_CALL_TIMEOUT_S", 60)

            def _call_llm():
                return client.chat.completions.create(
                    model      = cfg.LLM_MODEL,
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = cfg.LLM_MAX_TOKENS_FUSION,
                    temperature= cfg.LLM_TEMPERATURE_FUSION,
                    timeout    = _timeout,
                ).choices[0].message.content.strip()

            seg.fused_summary = retry_sync(_call_llm, cfg=_FUSION_RETRY, label=f"fusion_llm_seg{sid}")
        except Exception as e:
            log.warning(f"Fusion LLM call failed for segment {sid} after retries: {e}")
            seg.fused_summary = combined_text
            if seg.slide_title:
                seg.fused_summary = f"[{seg.slide_title}] {combined_text}"

        # ── Build embedding text ──────────────────────────────────────────
        # We embed a rich representation so semantic search captures both
        # spoken content and visual content.
        parts = [seg.fused_summary]
        if seg.slide_title:
            parts.append(f"Slide: {seg.slide_title}")
        if seg.slide_bullets:
            parts.append("Bullets: " + "; ".join(seg.slide_bullets))
        if seg.ocr_text:
            parts.append(f"Text on screen: {seg.ocr_text[:300]}")
        if seg.diagram_description:
            parts.append(f"Diagram: {seg.diagram_description}")
        seg.embedding_text = "\n".join(parts)

        log.info(f"Segment {sid:03d}  [{_fmt_ms(start_ms)} → {_fmt_ms(end_ms)}]  "
                 f"slide_change={seg.is_slide_change}")
        fused_segments.append(seg)

    log.info(f"Fusion complete: {len(fused_segments)} segments.")
    return fused_segments


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s//60:02d}:{s%60:02d}"


def save_fused(segments: List[FusedSegment], out_dir: Path) -> Path:
    path = out_dir / "fused_segments.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)
    log.info(f"Fused segments saved → {path}")
    return path


def load_fused(out_dir: Path) -> Optional[List[FusedSegment]]:
    path = out_dir / "fused_segments.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [FusedSegment(**d) for d in json.load(f)]
