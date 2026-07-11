"""core/vision/vlm_analyser.py — Phase 2b
Two-track frame analysis — strictly sequential, no asyncio, no threads.

  Track A — RapidOCR (ONNX) inline, GPU via onnxruntime CUDAExecutionProvider.
    One persistent engine instance reused across all frames (no per-frame model
    reload).  Uses the same PP-OCR ONNX models that PaddleOCR exposes.

  Track B — VLM (opencode local server / mimo-v2.5-free)
    A single structured call per frame analysing scene, slide, diagram,
    and visual delta in one shot.  OCR runs first; if the frame is text-rich
    (ocr_lines >= max(OCR_RICH_TEXT_MIN_LINES, running_avg)) the VLM is skipped.

Frame loop order per frame:
  1. Skip if timestamp cached.
  2. Skip if perceptually identical to last analysed frame.
  3. Run OCR.
  4. Adaptive VLM gate — skip VLM if text-rich.
  5. If VLM needed: compress → one merged structured VLM call.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

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
#  Track A — RapidOCR inline (GPU via onnxruntime CUDAExecutionProvider)
# ─────────────────────────────────────────────────────────────────────────────

_ocr_engine = None
_ocr_available: bool | None = None


def _torch_lib_dir() -> str | None:
    """Return torch's bundled CUDA/cuDNN DLL dir so onnxruntime can find them.

    onnxruntime-gpu's CUDAExecutionProvider needs cudnn64_9.dll + cudart64_12.dll
    on PATH.  torch (cu128) ships exactly those in torch/lib — reuse them.
    """
    try:
        import os as _os
        import torch
        d = _os.path.join(_os.path.dirname(torch.__file__), "lib")
        return d if _os.path.isdir(d) else None
    except Exception:
        return None


def _get_ocr_engine(cfg):
    """Lazily build a single RapidOCR engine (CUDA if available) and cache it."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    import os as _os
    lib = _torch_lib_dir()
    if lib:
        _os.environ["PATH"] = lib + _os.pathsep + _os.environ.get("PATH", "")
    from rapidocr_onnxruntime import RapidOCR
    use_gpu = bool(getattr(cfg, "OCR_USE_GPU", True))
    _ocr_engine = RapidOCR(
        det_use_cuda=use_gpu, cls_use_cuda=use_gpu, rec_use_cuda=use_gpu,
    )
    eng_providers = []
    try:
        for attr in ("det", "cls", "rec"):
            sub = getattr(_ocr_engine, attr, None)
            sess = getattr(sub, "session", None) if sub else None
            if sess is not None and hasattr(sess, "get_providers"):
                eng_providers.append(f"{attr}={sess.get_providers()}")
    except Exception:
        pass
    log.info(f"RapidOCR engine ready (use_gpu={use_gpu}) {', '.join(eng_providers)}")
    return _ocr_engine


def _check_ocr_available(cfg=None) -> bool:
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        _get_ocr_engine(cfg) if cfg is not None else _get_ocr_engine(_DummyCfg())
        _ocr_available = True
        log.info("RapidOCR probe OK -- OCR track enabled (GPU).")
    except Exception as e:
        _ocr_available = False
        log.warning(f"RapidOCR probe failed -- OCR track disabled: {e}")
    return _ocr_available


class _DummyCfg:
    OCR_USE_GPU = True


def run_ocr(frame_path: Path, cfg) -> tuple[str, list[str]]:
    """Run RapidOCR inline on the persistent GPU engine.  Returns (full_text, lines)."""
    if not _check_ocr_available(cfg):
        return "", []
    engine = _get_ocr_engine(cfg)
    min_conf = float(getattr(cfg, "OCR_MIN_CONFIDENCE", 0.6))

    def _run() -> list[str]:
        result = engine(str(frame_path))
        # RapidOCR 1.4.x returns (list_of_[box, text, score], timings_list)
        items = result[0] if result and len(result) >= 1 else []
        lines: list[str] = []
        for it in items:
            if isinstance(it, (list, tuple)) and len(it) >= 3:
                text, score = it[1], it[2]
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 1.0
                if score >= min_conf and text and str(text).strip():
                    lines.append(str(text).strip())
        return lines

    try:
        lines = retry_sync(_run, cfg=_OCR_RETRY, label="rapidocr")
        return "\n".join(lines), lines
    except Exception as e:
        log.error(f"OCR failed for {frame_path.name}: {e}")
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
#  Track B — VLM prompt sets  (ZH and EN)
# ─────────────────────────────────────────────────────────────────────────────


class _PromptSet(NamedTuple):
    merged: str          # one structured prompt covering scene/slide/diagram/delta
    no_diagram: str      # sentinel string returned when no diagram found
    has_delta: bool      # whether the delta-comparison section is active
    prev_desc: str       # previous-frame scene description for delta


_ZH_BASE = (
    "分析这张视频帧。输出严格的JSON，字段如下：\n"
    "scene: 2-3句描述图像内容，含内容类型(幻灯片/终端/白板/演示/摄像头)及主要主题，不读出图中文字。\n"
    "slide: 若为演示幻灯片，输出 {\"is_slide\":true,\"slide_type\":\"title|content|diagram|code|table|other\",\"title\":\"...\",\"bullets\":[\"...\"]}；否则输出 {\"is_slide\":false}。\n"
    "diagram: 若有图表/流程图/表格/代码块/公式，2-4句描述类型、概念、关键标签；否则填 \"[无图表]\"。\n"
    "delta: 分类 same | slide_change | new_content | major_change，只填一个词。"
)

_EN_BASE = (
    "Analyse this video frame. Output strict JSON with these fields:\n"
    "scene: 2-3 sentences describing content, type (slide/terminal/whiteboard/demo/camera) and topic; do not transcribe text.\n"
    "slide: if a presentation slide, output {\"is_slide\":true,\"slide_type\":\"title|content|diagram|code|table|other\",\"title\":\"...\",\"bullets\":[\"...\"]}; else {\"is_slide\":false}.\n"
    "diagram: describe any chart/flowchart/table/code/formula in 2-4 sentences, or \"[no diagram]\".\n"
    "delta: classify as same | slide_change | new_content | major_change (one word)."
)


def _build_prompt(lang: str, prev_desc: str | None) -> _PromptSet:
    base = _ZH_BASE if lang == "zh" else _EN_BASE
    no_diag = "[无图表]" if lang == "zh" else "[no diagram]"
    has_delta = bool(prev_desc)
    if has_delta:
        extra = (
            f"\n前一帧场景(t≤250字)：{prev_desc[:250]}\n" if lang == "zh"
            else f"\nPrevious frame scene (≤250 chars): {prev_desc[:250]}\n"
        )
        merged = base + extra
    else:
        merged = base
    return _PromptSet(merged=merged, no_diagram=no_diag,
                      has_delta=has_delta, prev_desc=prev_desc or "")


def _parse_vlm_json(analysis: FrameAnalysis, raw: str, prompts: _PromptSet) -> None:
    """Parse the merged JSON response into analysis fields (best-effort)."""
    clean = raw.strip()
    # strip ```json ... ``` fences if present
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1] if "\n" in clean else clean.lstrip("`jns`")
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
    clean = clean.strip().strip("`").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # last-resort: try to locate the first {...} block
        lo = clean.find("{")
        hi = clean.rfind("}")
        if lo >= 0 and hi > lo:
            try:
                data = json.loads(clean[lo:hi + 1])
            except json.JSONDecodeError:
                log.debug(f"Could not parse VLM JSON: {raw[:200]!r}")
                return
        else:
            return

    if not isinstance(data, dict):
        return

    scene = data.get("scene")
    if isinstance(scene, str):
        analysis.scene_description = scene.strip()

    slide = data.get("slide")
    if isinstance(slide, dict) and slide.get("is_slide"):
        analysis.slide_type = slide.get("slide_type", "") or ""
        analysis.slide_title = slide.get("title", "") or ""
        bullets = slide.get("bullets") or []
        if isinstance(bullets, list):
            analysis.slide_bullets = [str(b) for b in bullets]

    diag = data.get("diagram")
    if isinstance(diag, str) and diag.strip() and diag.strip() != prompts.no_diagram:
        analysis.diagram_description = diag.strip()

    delta = data.get("delta")
    if isinstance(delta, str):
        d = delta.strip().lower()
        analysis.visual_delta = d if d in _VALID_DELTAS else "new_content"
    elif not prompts.has_delta:
        analysis.visual_delta = "new_content"


_VALID_DELTAS = frozenset(("same", "slide_change", "new_content", "major_change"))

# ─────────────────────────────────────────────────────────────────────────────
#  Per-frame analysis (synchronous)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_frame(
    req,
    frame_path: Path,
    prev_analysis: FrameAnalysis | None,
    vlm,
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
        raw = vlm.call(image_b64, prompts.merged, mime_type)
        _parse_vlm_json(analysis, raw, prompts)
        if not analysis.scene_description:
            log.warning(f"VLM returned no scene ts={req.timestamp_ms}ms: {raw[:120]!r}")
    except Exception as e:
        log.error(f"VLM analysis failed ts={req.timestamp_ms}ms: {e}")
        analysis.vlm_error = str(e)
        if not analysis.visual_delta:
            analysis.visual_delta = "unknown"
    frame_cache[fhash] = analysis
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
#  Batch runner — plain for-loop, no asyncio
# ─────────────────────────────────────────────────────────────────────────────

def analyse_all_frames(
    frame_results: list[tuple],
    vlm,
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
      5. Run one merged VLM call (scene + slide + diagram + delta).

    ``lang`` selects the prompt language: "zh" for Chinese, "en" for English.
    """
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

    skipped_cache = 0
    pbar = tqdm(enumerate(frame_results), total=n, desc="Phase 2b VLM+OCR", unit="frame", leave=True)
    for i, (req, path) in pbar:

        # 1. Cache hit (prior run)
        if req.timestamp_ms in existing_by_ts:
            analyses.append(existing_by_ts[req.timestamp_ms])
            last_analyzed_path = path
            skipped_cache += 1
            pbar.set_postfix(cache=skipped_cache, sim=skipped_sim, vlm_skip=skipped_vlm, done=completed, status="cached")
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
                pbar.set_postfix(cache=skipped_cache, sim=skipped_sim, vlm_skip=skipped_vlm, done=completed, status=f"sim={sim:.2f}")
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
        prev_desc = prev.scene_description if prev and prev.scene_description else None
        prompts = _build_prompt(lang, prev_desc)
        result = analyse_frame(
            req, path, prev, vlm, cfg, frame_cache, prompts,
            ocr_prefetch=(ocr_text, ocr_lines),
            skip_vlm=skip_vlm,
        )
        analyses.append(result)
        completed += 1
        last_analyzed_path = path
        ocr_line_history.append(n_lines)

        if skip_vlm:
            pbar.set_postfix(cache=skipped_cache, sim=skipped_sim, vlm_skip=skipped_vlm, done=completed, ocr=n_lines, status="ocr-only")
        else:
            pbar.set_postfix(cache=skipped_cache, sim=skipped_sim, vlm_skip=skipped_vlm, done=completed, ocr=n_lines, delta=result.visual_delta, status="vlm")
            log.info(
                f"ts={req.timestamp_ms}ms ocr_lines={n_lines} delta={result.visual_delta} size={result.image_size}",
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
