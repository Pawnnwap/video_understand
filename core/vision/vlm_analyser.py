"""core/vision/vlm_analyser.py — Phase 2b
Two-track frame analysis, pipelined in four phases:

  Phase 0 (sequential, cheap): cache and frame-similarity gates.
  Phase 1 (parallel): RapidOCR over the surviving frames on a thread pool
    sized to OCR_PARALLEL_FRACTION of the CPU cores; each worker thread owns
    its own RapidOCR engine (onnxruntime sessions are not shared).
  Phase 2 (sequential, cheap): adaptive VLM gate — replays the running
    OCR-line average in frame order, so gating decisions are identical to the
    old sequential implementation.
  Phase 3 (parallel): merged VLM calls (scene/slide/diagram/delta) on a pool
    capped at VLM_MAX_PARALLEL; each call uses a fresh opencode session.
    The delta prompt uses the nearest earlier frame whose scene description
    has already completed (under parallelism the immediate predecessor may
    still be in flight).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
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

_ocr_tls = threading.local()  # one RapidOCR engine per worker thread/device mode
_ocr_available: bool | None = None
_ocr_gpu_disabled = False
_ocr_state_lock = threading.Lock()


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


def _get_ocr_engine(cfg, *, use_gpu: bool | None = None):
    """Lazily build one RapidOCR engine per thread and device mode."""
    global _ocr_gpu_disabled
    requested_gpu = bool(getattr(cfg, "OCR_USE_GPU", True)) if use_gpu is None else bool(use_gpu)
    actual_gpu = requested_gpu and not _ocr_gpu_disabled
    attr = "gpu_engine" if actual_gpu else "cpu_engine"
    engine = getattr(_ocr_tls, attr, None)
    if engine is not None:
        return engine
    import os as _os
    lib = _torch_lib_dir()
    if lib:
        _os.environ["PATH"] = lib + _os.pathsep + _os.environ.get("PATH", "")
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(
        det_use_cuda=actual_gpu, cls_use_cuda=actual_gpu, rec_use_cuda=actual_gpu,
    )
    setattr(_ocr_tls, attr, engine)
    log.info(
        f"RapidOCR engine ready (use_gpu={actual_gpu}, thread={threading.current_thread().name})"
    )
    return engine


def _check_ocr_available(cfg=None) -> bool:
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    cfg = cfg if cfg is not None else _DummyCfg()
    try:
        _get_ocr_engine(cfg)
        _ocr_available = True
        mode = "GPU" if bool(getattr(cfg, "OCR_USE_GPU", True)) and not _ocr_gpu_disabled else "CPU"
        log.info(f"RapidOCR probe OK -- OCR track enabled ({mode}).")
    except Exception as e:
        if bool(getattr(cfg, "OCR_USE_GPU", True)):
            log.warning(f"RapidOCR GPU probe failed; retrying OCR on CPU: {e}")
            try:
                _disable_ocr_gpu()
                _get_ocr_engine(cfg, use_gpu=False)
                _ocr_available = True
                log.info("RapidOCR probe OK -- OCR track enabled (CPU fallback).")
            except Exception as cpu_e:
                _ocr_available = False
                log.warning(f"RapidOCR CPU probe failed -- OCR track disabled: {cpu_e}")
        else:
            _ocr_available = False
            log.warning(f"RapidOCR probe failed -- OCR track disabled: {e}")
    return _ocr_available


class _DummyCfg:
    OCR_USE_GPU = True
    OCR_GPU_MAX_WORKERS = 1
    OCR_MAX_DIM = 1600
    OCR_RESIZE_QUALITY = 92


def _disable_ocr_gpu() -> None:
    global _ocr_gpu_disabled
    with _ocr_state_lock:
        if not _ocr_gpu_disabled:
            _ocr_gpu_disabled = True
            log.warning("RapidOCR CUDA OOM detected; falling back to CPU OCR for the rest of this run.")


def _is_cuda_oom(exc: BaseException) -> bool:
    text = " ".join(str(arg) for arg in getattr(exc, "args", ()) if arg)
    text = f"{type(exc).__name__}: {exc} {text}".lower()
    return (
        ("cuda" in text and ("out of memory" in text or "cuda failure 2" in text))
        or "failed to allocate memory" in text
        or "bfcarena::allocaterawinternal" in text
    )


def _prepare_ocr_image(frame_path: Path, cfg) -> tuple[Path, Path | None]:
    """Return an OCR-sized image path and optional temporary file to delete."""
    max_dim = int(getattr(cfg, "OCR_MAX_DIM", 1600) or 0)
    if max_dim <= 0:
        return frame_path, None

    from PIL import Image

    with Image.open(frame_path) as img:
        width, height = img.size
        current_max = max(width, height)
        if current_max <= max_dim:
            return frame_path, None

        scale = max_dim / current_max
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = img.convert("RGB").resize(resized_size, Image.Resampling.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{frame_path.stem}_ocr_",
            suffix=".jpg",
            delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        quality = int(getattr(cfg, "OCR_RESIZE_QUALITY", 92) or 92)
        resized.save(tmp_path, "JPEG", quality=quality, optimize=True)
        log.debug(
            "Prepared OCR image %s -> %s (%sx%s -> %sx%s)",
            frame_path.name,
            tmp_path.name,
            width,
            height,
            resized_size[0],
            resized_size[1],
        )
        return tmp_path, tmp_path


def run_ocr(frame_path: Path, cfg) -> tuple[str, list[str]]:
    """Run RapidOCR; fall back from CUDA to CPU on ONNXRuntime OOM."""
    if not _check_ocr_available(cfg):
        return "", []
    min_conf = float(getattr(cfg, "OCR_MIN_CONFIDENCE", 0.6))
    cleanup_path: Path | None = None
    try:
        ocr_path, cleanup_path = _prepare_ocr_image(frame_path, cfg)
    except Exception as e:
        log.warning(f"OCR image resize failed for {frame_path.name}; using original frame: {e}")
        ocr_path = frame_path

    def _run_with_engine(engine) -> list[str]:
        result = engine(str(ocr_path))
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
        engine = _get_ocr_engine(cfg)
        lines = retry_sync(lambda: _run_with_engine(engine), cfg=_OCR_RETRY, label="rapidocr")
        return "\n".join(lines), lines
    except Exception as e:
        if bool(getattr(cfg, "OCR_USE_GPU", True)) and _is_cuda_oom(e):
            _disable_ocr_gpu()
            try:
                cpu_engine = _get_ocr_engine(cfg, use_gpu=False)
                lines = retry_sync(lambda: _run_with_engine(cpu_engine), cfg=_OCR_RETRY, label="rapidocr_cpu")
                return "\n".join(lines), lines
            except Exception as cpu_e:
                log.warning(f"OCR CPU fallback failed for {frame_path.name}: {cpu_e}")
                return "", []
        log.warning(f"OCR failed for {frame_path.name}: {e}")
        return "", []
    finally:
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove temporary OCR image %s", cleanup_path)


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
            f"\n前一帧场景(≤250字)：{prev_desc[:250]}\n" if lang == "zh"
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
#  Batch runner — pipelined: parallel OCR pool + capped parallel VLM pool
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_worker_count(cfg) -> int:
    fraction = float(getattr(cfg, "OCR_PARALLEL_FRACTION", 0.9))
    workers = max(1, int((os.cpu_count() or 4) * fraction))
    if bool(getattr(cfg, "OCR_USE_GPU", True)) and not _ocr_gpu_disabled:
        gpu_cap = max(1, int(getattr(cfg, "OCR_GPU_MAX_WORKERS", 1)))
        workers = min(workers, gpu_cap)
    return workers


def analyse_all_frames(
    frame_results: list[tuple],
    vlm,
    cfg,
    out_dir: Path,
    lang: str = "zh",
) -> list[FrameAnalysis]:
    """Four-phase pipelined analysis; results keep frame order.

    Gate decisions (disk cache, similarity, adaptive VLM skip) replicate the
    old sequential semantics exactly. Only the delta-prompt reference frame
    is relaxed: under parallel VLM it is the nearest earlier frame whose
    scene description has completed.
    """
    from core.vision.frame_sampler import frame_hash

    log.info(f"VLM prompt language: {lang}")
    cache_path = out_dir / "frame_analyses.json"
    existing_by_ts = {a.timestamp_ms: a for a in _load_analyses(cache_path)}

    sim_threshold = getattr(cfg, "FRAME_SIMILARITY_THRESHOLD", 0.90)
    vlm_skip_floor = getattr(cfg, "OCR_RICH_TEXT_MIN_LINES", 3)

    n = len(frame_results)
    analyses: list[FrameAnalysis | None] = [None] * n

    # ── Phase 0: cache + similarity gates (sequential, no OCR needed) ──
    ocr_indices: list[int] = []   # frames that need OCR (original step 3)
    dup_of: dict[int, int] = {}   # frame index -> earlier index to copy from
    skipped_cache = skipped_sim = 0
    last_analyzed_idx: int | None = None
    for i, (req, path) in enumerate(frame_results):
        if req.timestamp_ms in existing_by_ts:
            analyses[i] = existing_by_ts[req.timestamp_ms]
            last_analyzed_idx = i
            skipped_cache += 1
            continue
        if last_analyzed_idx is not None:
            sim = compute_frame_similarity(path, frame_results[last_analyzed_idx][1])
            if sim >= sim_threshold:
                dup_of[i] = i - 1  # copy the immediately preceding entry
                skipped_sim += 1
                continue
        ocr_indices.append(i)
        last_analyzed_idx = i

    # ── Phase 1: OCR pool ──────────────────────────────────────────────
    ocr_results: dict[int, tuple[str, list[str]]] = {}
    if ocr_indices:
        workers = min(_ocr_worker_count(cfg), len(ocr_indices))
        log.info(f"OCR pool: {workers} workers for {len(ocr_indices)} frames")

        def _ocr_one(idx: int) -> tuple[int, tuple[str, list[str]]]:
            req, path = frame_results[idx]
            try:
                return idx, run_ocr(path, cfg)
            except Exception as e:
                log.warning(f"OCR error ts={req.timestamp_ms}ms: {e}")
                return idx, ("", [])

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr") as pool:
            for idx, result in tqdm(
                pool.map(_ocr_one, ocr_indices), total=len(ocr_indices),
                desc=f"Phase 2b OCR x{workers}", unit="frame", leave=True,
            ):
                ocr_results[idx] = result

    # ── Phase 2: adaptive VLM gate (sequential replay, exact semantics) ─
    frame_cache: dict[str, int] = {}          # frame hash -> first index
    vlm_indices: list[int] = []
    skipped_vlm = 0
    ocr_line_history: collections.deque[int] = collections.deque(maxlen=_OCR_HISTORY_WINDOW)
    for i in ocr_indices:
        req, path = frame_results[i]
        ocr_text, ocr_lines = ocr_results[i]
        n_lines = len(ocr_lines)
        ocr_avg = sum(ocr_line_history) / len(ocr_line_history) if ocr_line_history else 0.0
        skip_vlm = n_lines >= max(vlm_skip_floor, ocr_avg)
        ocr_line_history.append(n_lines)

        fhash = frame_hash(path)
        if fhash in frame_cache:               # identical frame seen earlier
            dup_of[i] = frame_cache[fhash]
            continue
        frame_cache[fhash] = i

        analysis = FrameAnalysis(
            timestamp_ms=req.timestamp_ms,
            reason=req.reason,
            sentence_id=req.sentence_id,
            frame_path=str(path),
            frame_hash=fhash,
            ocr_text=ocr_text,
            ocr_lines=ocr_lines,
            has_text_content=bool(ocr_text.strip()),
        )
        analyses[i] = analysis
        if skip_vlm:
            skipped_vlm += 1
            analysis.vlm_skipped = True
            analysis.visual_delta = "new_content"
        else:
            vlm_indices.append(i)

    # ── Phase 3: VLM pool (capped, fresh session per call) ─────────────
    completed_scenes: dict[int, str] = {}
    scenes_lock = threading.Lock()
    save_lock = threading.Lock()
    done_count = 0

    def _vlm_one(idx: int) -> None:
        analysis = analyses[idx]
        req = frame_results[idx][0]
        try:
            image_b64, mime_type, w, h = compress_frame_for_vlm(analysis.frame_path, cfg)
            analysis.image_size = f"{w}x{h}"
        except Exception as e:
            log.error(f"Frame compression failed ts={req.timestamp_ms}ms: {e}")
            analysis.vlm_error = f"compression: {e}"
            analysis.visual_delta = "unknown"
            return
        with scenes_lock:
            earlier = [j for j in completed_scenes if j < idx]
            prev_desc = completed_scenes[max(earlier)] if earlier else None
        prompts = _build_prompt(lang, prev_desc)
        try:
            raw = vlm.call(image_b64, prompts.merged, mime_type, fresh_session=True)
            _parse_vlm_json(analysis, raw, prompts)
            if not analysis.scene_description:
                log.warning(f"VLM returned no scene ts={req.timestamp_ms}ms: {raw[:120]!r}")
        except Exception as e:
            log.error(f"VLM analysis failed ts={req.timestamp_ms}ms: {e}")
            analysis.vlm_error = str(e)
            if not analysis.visual_delta:
                analysis.visual_delta = "unknown"
        with scenes_lock:
            if analysis.scene_description:
                completed_scenes[idx] = analysis.scene_description

    if vlm_indices:
        vlm_workers = min(int(getattr(cfg, "VLM_MAX_PARALLEL", 4)), len(vlm_indices))
        log.info(f"VLM pool: {vlm_workers} parallel for {len(vlm_indices)} frames")
        with ThreadPoolExecutor(max_workers=vlm_workers, thread_name_prefix="vlm") as pool:
            for _ in tqdm(
                pool.map(_vlm_one, vlm_indices), total=len(vlm_indices),
                desc=f"Phase 2b VLM x{vlm_workers}", unit="frame", leave=True,
            ):
                done_count += 1
                if done_count % 10 == 0:
                    with save_lock:
                        _save_analyses([a for a in analyses if a is not None], cache_path)

    # ── Resolve similarity/hash duplicates (transitively) ──────────────
    for i in sorted(dup_of):
        src = dup_of[i]
        while src in dup_of:
            src = dup_of[src]
        req, path = frame_results[i]
        dup = FrameAnalysis(**asdict(analyses[src]))
        dup.timestamp_ms = req.timestamp_ms
        dup.reason = req.reason
        dup.sentence_id = req.sentence_id
        dup.frame_path = str(path)
        dup.visual_delta = "same"
        analyses[i] = dup

    final = [a for a in analyses if a is not None]
    _save_analyses(final, cache_path)
    log.info(
        f"VLM+OCR complete: {len(final)} frames  "
        f"(vlm={len(vlm_indices)}, ocr_only={skipped_vlm}, "
        f"duplicates={len(dup_of)}, cached={skipped_cache})",
    )
    return final


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
