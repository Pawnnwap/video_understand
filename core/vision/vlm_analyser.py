"""core/vision/vlm_analyser.py — Phase 2b
Two-track frame analysis — strictly sequential, no asyncio, no threads.

  Track A — PaddleOCR (subprocess)
    Runs in a separate process to avoid PyTorch/Paddle CUDA conflict.

  Track B — VLM (LM Studio HTTP, one blocking call at a time)
    Sequential gate: OCR runs first.  If the frame is text-rich
    (ocr_lines >= max(OCR_RICH_TEXT_MIN_LINES, running_avg)), VLM is skipped.

Frame loop order per frame:
  1. Skip if timestamp cached.
  2. Skip if perceptually identical to last analysed frame.
  3. Run OCR.
  4. Adaptive VLM gate — skip VLM if text-rich.
  5. If VLM needed: compress → scene → slide → diagram → delta (all sequential).
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

from utils.retry import (
    RetryConfig,
    compress_frame_for_vlm,
    compute_frame_similarity,
    retry_sync,
)

log = logging.getLogger(__name__)

_VLM_RETRY = RetryConfig(max_attempts=4, base_delay_s=3.0, max_delay_s=30.0)
_OCR_RETRY = RetryConfig(
    max_attempts=3, base_delay_s=1.0, max_delay_s=8.0,
    retryable=(ConnectionError, TimeoutError, OSError, RuntimeError),
)

# Keep only the most recent N frames for the running OCR-line average.
# This prevents skewing on very long videos with many text-dense frames
# at the start followed by camera footage later.
_OCR_HISTORY_WINDOW = 100


@dataclass
class FrameAnalysis:
    timestamp_ms: int
    reason: str
    sentence_id: int
    frame_path: str
    ocr_text: str = ""
    ocr_lines: list[str] = field(default_factory=list)
    has_text_content: bool = False
    scene_description: str = ""
    slide_title: str = ""
    slide_bullets: list[str] = field(default_factory=list)
    slide_type: str = ""
    diagram_description: str = ""
    visual_delta: str = ""
    frame_hash: str = ""
    image_size: str = ""
    vlm_skipped: bool = False
    ocr_error: str | None = None
    vlm_error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Track A — PaddleOCR subprocess
# ─────────────────────────────────────────────────────────────────────────────

_ocr_available: bool | None = None


def _check_ocr_available() -> bool:
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    import subprocess
    import sys
    probe = (
        "import json,os;"
        "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True';"
        "from paddleocr import PaddleOCR;"
        "print(json.dumps({'lines':[],'error':None}))"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, timeout=30, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
    )
    _ocr_available = (r.returncode == 0)
    if _ocr_available:
        log.info("PaddleOCR subprocess probe OK -- OCR track enabled.")
    else:
        log.warning("PaddleOCR probe failed -- OCR track disabled.")
    return _ocr_available


def run_ocr(frame_path: Path, cfg) -> tuple[str, list[str]]:
    """Run PaddleOCR in a subprocess.  Returns (full_text, lines)."""
    import json as _json
    import subprocess
    import sys

    if not _check_ocr_available():
        return "", []

    # ocr_worker.py lives alongside this file
    worker = Path(__file__).parent / "ocr_worker.py"

    def _run() -> list[str]:
        r = subprocess.run(
            [sys.executable, str(worker),
             str(frame_path), cfg.OCR_LANG, str(cfg.OCR_MIN_CONFIDENCE)],
            capture_output=True, text=True,
            timeout=getattr(cfg, "OCR_TIMEOUT_S", 60),
            encoding="utf-8", errors="replace",
            env={**__import__("os").environ,
                 "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                 "PYTHONIOENCODING": "utf-8"},
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:300])
        data = _json.loads(r.stdout)
        if data.get("error"):
            raise RuntimeError(data["error"])
        return data["lines"]

    try:
        lines = retry_sync(_run, cfg=_OCR_RETRY, label="paddleocr")
        return "\n".join(lines), lines
    except Exception as e:
        log.error(f"OCR failed for {frame_path.name}: {e}")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
#  Track B — VLM prompt sets  (ZH and EN)
# ─────────────────────────────────────────────────────────────────────────────


class _PromptSet(NamedTuple):
    scene: str
    slide: str
    diagram: str
    delta_tmpl: str   # contains {prev_desc}
    no_diagram: str   # sentinel string the model returns when no diagram found


_ZH = _PromptSet(
    scene=(
        "用2-3句描述图像内容。"
        "说明内容类型（幻灯片/终端/白板/演示/摄像头）及主要主题。"
        "不读出图中文字。"
    ),
    slide=(
        "判断是否为演示幻灯片。\n"
        '是：{"is_slide":true,"slide_type":"title|content|diagram|code|table|other","title":"...","bullets":["...","..."]}\n'
        '否：{"is_slide":false}\n'
        "仅输出JSON。"
    ),
    diagram=(
        "图中是否有图表/流程图/表格/代码块/公式？\n"
        "有：2-4句描述类型、概念、关键标签。\n"
        "无：回复[无图表]"
    ),
    delta_tmpl=(
        "与前一帧对比：{prev_desc}\n"
        "分类：same | slide_change | new_content | major_change\n"
        "只回复一个词。"
    ),
    no_diagram="[无图表]",
)

_EN = _PromptSet(
    scene=(
        "Describe this image in 2-3 sentences. "
        "State the content type (slide, terminal, whiteboard, demo, camera) and main topic. "
        "Do not transcribe visible text."
    ),
    slide=(
        "Is this a presentation slide?\n"
        'Yes: {"is_slide":true,"slide_type":"title|content|diagram|code|table|other","title":"...","bullets":["...","..."]}\n'
        'No: {"is_slide":false}\n'
        "Output JSON only."
    ),
    diagram=(
        "Does this image contain a chart, diagram, flowchart, table, code block, or formula?\n"
        "Yes: describe in 2-4 sentences — type, concept, key labels.\n"
        "No: reply [no diagram]"
    ),
    delta_tmpl=(
        "Compare with previous frame: {prev_desc}\n"
        "Classify: same | slide_change | new_content | major_change\n"
        "Reply with one word only."
    ),
    no_diagram="[no diagram]",
)

_VALID_DELTAS = frozenset(("same", "slide_change", "new_content", "major_change"))


def _get_prompts(lang: str) -> _PromptSet:
    return _ZH if lang == "zh" else _EN


def _vlm_call(
    client, model: str, prompt: str, image_b64: str,
    max_tokens: int, temperature: float,
    mime_type: str = "image/jpeg",
    cfg=None,
) -> str:
    """One blocking VLM HTTP call with retry."""
    _timeout = getattr(cfg, "VLM_CALL_TIMEOUT_S", 120) if cfg else 120

    def _attempt() -> str:
        return client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=_timeout,
        ).choices[0].message.content.strip()

    return retry_sync(_attempt, cfg=_VLM_RETRY, label="vlm_call")


def _parse_slide_json(analysis: FrameAnalysis, raw: str) -> None:
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)
        if data.get("is_slide"):
            analysis.slide_type = data.get("slide_type", "")
            analysis.slide_title = data.get("title", "") or ""
            analysis.slide_bullets = data.get("bullets", []) or []
    except (json.JSONDecodeError, AttributeError):
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Per-frame analysis (synchronous)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_frame(
    req,
    frame_path: Path,
    prev_analysis: FrameAnalysis | None,
    client,
    cfg,
    frame_cache: dict[str, FrameAnalysis],
    prompts: _PromptSet,
    *,
    ocr_prefetch: tuple[str, list[str]] | None = None,
    skip_vlm: bool = False,
) -> FrameAnalysis:
    from core.vision.frame_sampler import frame_hash

    fhash = frame_hash(frame_path)

    # Return a shallow copy with updated positional metadata if already cached
    if fhash in frame_cache:
        cached = frame_cache[fhash]
        result = FrameAnalysis(**asdict(cached))
        result.timestamp_ms = req.timestamp_ms
        result.reason = req.reason
        result.sentence_id = req.sentence_id
        result.frame_path = str(frame_path)
        result.visual_delta = "same"
        return result

    analysis = FrameAnalysis(
        timestamp_ms=req.timestamp_ms,
        reason=req.reason,
        sentence_id=req.sentence_id,
        frame_path=str(frame_path),
        frame_hash=fhash,
    )

    # Track A: OCR
    if ocr_prefetch is not None:
        ocr_text, ocr_lines = ocr_prefetch
    else:
        try:
            ocr_text, ocr_lines = run_ocr(frame_path, cfg)
        except Exception as e:
            log.warning(f"OCR error ts={req.timestamp_ms}ms: {e}")
            analysis.ocr_error = str(e)
            ocr_text, ocr_lines = "", []

    analysis.ocr_text = ocr_text
    analysis.ocr_lines = ocr_lines
    analysis.has_text_content = bool(ocr_text.strip())

    if skip_vlm:
        analysis.vlm_skipped = True
        analysis.visual_delta = "new_content"
        frame_cache[fhash] = analysis
        return analysis

    # Track B: VLM
    try:
        image_b64, mime_type, w, h = compress_frame_for_vlm(frame_path, cfg)
        analysis.image_size = f"{w}x{h}"
    except Exception as e:
        log.error(f"Frame compression failed ts={req.timestamp_ms}ms: {e}")
        analysis.vlm_error = f"compression: {e}"
        frame_cache[fhash] = analysis
        return analysis

    try:
        try:
            analysis.scene_description = _vlm_call(
                client, cfg.VLM_MODEL, prompts.scene,
                image_b64, cfg.VLM_MAX_TOKENS, cfg.VLM_TEMPERATURE, mime_type, cfg)
        except Exception as e:
            log.warning(f"Scene call failed ts={req.timestamp_ms}ms: {e}")

        try:
            _parse_slide_json(analysis, _vlm_call(
                client, cfg.VLM_MODEL, prompts.slide,
                image_b64, 256, cfg.VLM_TEMPERATURE, mime_type, cfg))
        except Exception as e:
            log.warning(f"Slide call failed ts={req.timestamp_ms}ms: {e}")

        try:
            diagram_raw = _vlm_call(
                client, cfg.VLM_MODEL, prompts.diagram,
                image_b64, cfg.VLM_MAX_TOKENS, cfg.VLM_TEMPERATURE, mime_type, cfg)
            if diagram_raw != prompts.no_diagram:
                analysis.diagram_description = diagram_raw
        except Exception as e:
            log.warning(f"Diagram call failed ts={req.timestamp_ms}ms: {e}")

        if prev_analysis and prev_analysis.scene_description:
            try:
                delta_prompt = prompts.delta_tmpl.format(
                    prev_desc=prev_analysis.scene_description[:250])
                delta = _vlm_call(
                    client, cfg.VLM_MODEL, delta_prompt,
                    image_b64, 16, 0.0, mime_type, cfg)
                analysis.visual_delta = delta.strip().lower()
                if analysis.visual_delta not in _VALID_DELTAS:
                    analysis.visual_delta = "new_content"
            except Exception as e:
                log.warning(f"Delta call failed ts={req.timestamp_ms}ms: {e}")
                analysis.visual_delta = "unknown"
        else:
            analysis.visual_delta = "new_content"

    except Exception as e:
        log.error(f"VLM analysis failed ts={req.timestamp_ms}ms: {e}")
        analysis.vlm_error = str(e)

    frame_cache[fhash] = analysis
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
#  Batch runner — plain for-loop, no asyncio
# ─────────────────────────────────────────────────────────────────────────────

def analyse_all_frames(
    frame_results: list[tuple],
    client,
    cfg,
    out_dir: Path,
    lang: str = "zh",
) -> list[FrameAnalysis]:
    """Process every frame one at a time — strictly sequential.

    Per frame:
      1. Skip if timestamp is already cached on disk.
      2. Skip (copy prev + mark same) if perceptually identical to last frame.
      3. Run OCR.
      4. Adaptive VLM gate: skip VLM if ocr_lines >= max(floor, windowed_avg).
      5. Run VLM (scene → slide → diagram → delta), one call at a time.

    ``lang`` selects the prompt language: "zh" for Chinese, "en" for English.
    """
    prompts = _get_prompts(lang)
    log.info(f"VLM prompt language: {lang}")
    cache_path = out_dir / "frame_analyses.json"
    frame_cache: dict[str, FrameAnalysis] = {}

    existing = _load_analyses(cache_path)
    existing_by_ts = {a.timestamp_ms: a for a in existing}

    sim_threshold = getattr(cfg, "FRAME_SIMILARITY_THRESHOLD", 0.90)
    vlm_skip_floor = getattr(cfg, "OCR_RICH_TEXT_MIN_LINES", 3)

    n = len(frame_results)
    analyses: list[FrameAnalysis] = []
    completed = 0
    skipped_sim = 0
    skipped_vlm = 0
    last_analyzed_path: Path | None = None

    # Bounded deque prevents memory growth on very long videos and keeps
    # the running average representative of recent content rather than the
    # whole video.
    ocr_line_history: collections.deque[int] = collections.deque(maxlen=_OCR_HISTORY_WINDOW)

    for i, (req, path) in enumerate(frame_results):

        # 1. Cache hit (prior run)
        if req.timestamp_ms in existing_by_ts:
            analyses.append(existing_by_ts[req.timestamp_ms])
            last_analyzed_path = path
            continue

        # 2. Similarity gate
        if last_analyzed_path is not None:
            sim = compute_frame_similarity(path, last_analyzed_path)
            if sim >= sim_threshold:
                prev = analyses[-1]
                dup = FrameAnalysis(**asdict(prev))
                dup.timestamp_ms = req.timestamp_ms
                dup.reason = req.reason
                dup.sentence_id = req.sentence_id
                dup.frame_path = str(path)
                dup.visual_delta = "same"
                analyses.append(dup)
                skipped_sim += 1
                log.debug(f"[{i + 1}/{n}] ts={req.timestamp_ms}ms  sim={sim:.3f} -- skipped identical")
                continue

        # 3. OCR
        try:
            ocr_text, ocr_lines = run_ocr(path, cfg)
        except Exception as e:
            log.warning(f"OCR error ts={req.timestamp_ms}ms: {e}")
            ocr_text, ocr_lines = "", []

        # 4. Adaptive VLM gate
        n_lines = len(ocr_lines)
        ocr_avg = sum(ocr_line_history) / len(ocr_line_history) if ocr_line_history else 0.0
        skip_vlm = n_lines >= max(vlm_skip_floor, ocr_avg)
        if skip_vlm:
            skipped_vlm += 1

        # 5. Analyse
        prev = analyses[-1] if analyses else None
        result = analyse_frame(
            req, path, prev, client, cfg, frame_cache, prompts,
            ocr_prefetch=(ocr_text, ocr_lines),
            skip_vlm=skip_vlm,
        )
        analyses.append(result)
        completed += 1
        last_analyzed_path = path
        ocr_line_history.append(n_lines)

        log.info(
            f"[{completed}/{n}] ts={req.timestamp_ms}ms  "
            f"ocr_lines={n_lines}  vlm={'skip' if skip_vlm else 'run'}"
            + (f"  delta={result.visual_delta}  size={result.image_size}" if not skip_vlm else ""),
        )

        if completed % 10 == 0:
            _save_analyses(analyses, cache_path)

    _save_analyses(analyses, cache_path)
    log.info(
        f"VLM+OCR complete: {len(analyses)} frames  "
        f"(analyzed={completed}, skipped_identical={skipped_sim}, vlm_skipped={skipped_vlm})",
    )
    return analyses


def _save_analyses(analyses: list[FrameAnalysis], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in analyses], f, ensure_ascii=False, indent=2)


def _load_analyses(path: Path) -> list[FrameAnalysis]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [FrameAnalysis(**d) for d in json.load(f)]
    except Exception as e:
        log.warning(f"Could not load cached analyses: {e}")
        return []
