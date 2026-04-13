from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    import json as _json
    import subprocess
    import sys
    if not _check_ocr_available():
        return "", []
    worker = Path(__file__).parent / "ocr_worker.py"

    def _run():
        r = subprocess.run(
            [sys.executable, str(worker),
             str(frame_path), cfg.OCR_LANG, str(cfg.OCR_MIN_CONFIDENCE)],
            capture_output=True, text=True, timeout=getattr(cfg, "OCR_TIMEOUT_S", 60),
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


PROMPT_SCENE = """\
请用2-3句话描述这张图像中的内容。
重点描述：显示的是什么类型的内容（幻灯片、终端、白板、演示、摄像头画面），以及主要主题是什么。
请简洁客观，不要读出图中的文字内容。"""

PROMPT_SLIDE = """\
这张图像是否是演示文稿的幻灯片？
如果是，只输出以下JSON格式：
{"is_slide":true,"slide_type":"title|content|diagram|code|table|other","title":"...","bullets":["...","..."]}
如果不是，只输出：{"is_slide":false}
只输出纯JSON，不要任何markdown或解释。"""

PROMPT_DIAGRAM = """\
这张图像是否包含图表、示意图、流程图、数据表、代码块或公式？
如果有，用2-4句话描述：视觉类型、所表示的概念或数据，以及关键标签。
如果没有，只回复：[无图表]"""

PROMPT_DELTA = """\
将此图像与前一帧的描述进行对比：{prev_desc}
将变化分类为以下其中之一：
same（相同）| slide_change（幻灯片切换）| new_content（新内容）| major_change（重大变化）
只回复那一个英文词。"""


def _vlm_call(client, model, prompt, image_b64, max_tokens, temperature,
              mime_type: str = "image/jpeg", cfg=None) -> str:
    _timeout = getattr(cfg, "VLM_CALL_TIMEOUT_S", 120) if cfg else 120

    def _attempt():
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


def _parse_slide_json(analysis: FrameAnalysis, raw: str):
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)
        if data.get("is_slide"):
            analysis.slide_type = data.get("slide_type", "")
            analysis.slide_title = data.get("title", "") or ""
            analysis.slide_bullets = data.get("bullets", []) or []
    except (json.JSONDecodeError, AttributeError):
        pass


def analyse_frame(
    req,
    frame_path: Path,
    prev_analysis: FrameAnalysis | None,
    client,
    cfg,
    frame_cache: dict[str, FrameAnalysis],
    *,
    ocr_prefetch: tuple[str, list[str]] | None = None,
    skip_vlm: bool = False,
) -> FrameAnalysis:
    from core.frame_sampler import frame_hash
    fhash = frame_hash(frame_path)
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
                client, cfg.VLM_MODEL, PROMPT_SCENE,
                image_b64, cfg.VLM_MAX_TOKENS, cfg.VLM_TEMPERATURE, mime_type, cfg)
        except Exception as e:
            log.warning(f"Scene call failed ts={req.timestamp_ms}ms: {e}")
        try:
            _parse_slide_json(analysis, _vlm_call(
                client, cfg.VLM_MODEL, PROMPT_SLIDE,
                image_b64, 256, cfg.VLM_TEMPERATURE, mime_type, cfg))
        except Exception as e:
            log.warning(f"Slide call failed ts={req.timestamp_ms}ms: {e}")
        try:
            diagram_raw = _vlm_call(
                client, cfg.VLM_MODEL, PROMPT_DIAGRAM,
                image_b64, cfg.VLM_MAX_TOKENS, cfg.VLM_TEMPERATURE, mime_type, cfg)
            if diagram_raw != "[无图表]" and diagram_raw != "[no diagram]":
                analysis.diagram_description = diagram_raw
        except Exception as e:
            log.warning(f"Diagram call failed ts={req.timestamp_ms}ms: {e}")
        if prev_analysis and prev_analysis.scene_description:
            try:
                delta = _vlm_call(
                    client, cfg.VLM_MODEL,
                    PROMPT_DELTA.format(prev_desc=prev_analysis.scene_description[:250]),
                    image_b64, 16, 0.0, mime_type, cfg)
                analysis.visual_delta = delta.strip().lower()
                if analysis.visual_delta not in ("same", "slide_change", "new_content", "major_change"):
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


def analyse_all_frames(
    frame_results: list[tuple],
    client,
    cfg,
    out_dir: Path,
) -> list[FrameAnalysis]:
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
    ocr_line_history: list[int] = []
    for i, (req, path) in enumerate(frame_results):
        if req.timestamp_ms in existing_by_ts:
            analyses.append(existing_by_ts[req.timestamp_ms])
            last_analyzed_path = path
            continue
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
        try:
            ocr_text, ocr_lines = run_ocr(path, cfg)
        except Exception as e:
            log.warning(f"OCR error ts={req.timestamp_ms}ms: {e}")
            ocr_text, ocr_lines = "", []
        n_lines = len(ocr_lines)
        ocr_avg = sum(ocr_line_history) / len(ocr_line_history) if ocr_line_history else 0.0
        skip_vlm = n_lines >= max(vlm_skip_floor, ocr_avg)
        if skip_vlm:
            skipped_vlm += 1
        prev = analyses[-1] if analyses else None
        result = analyse_frame(
            req, path, prev, client, cfg, frame_cache,
            ocr_prefetch=(ocr_text, ocr_lines),
            skip_vlm=skip_vlm,
        )
        analyses.append(result)
        completed += 1
        last_analyzed_path = path
        ocr_line_history.append(n_lines)
        log.info(
            f'[{completed}/{n}] ts={req.timestamp_ms}ms  '
            f'ocr_lines={n_lines}  vlm={"skip" if skip_vlm else "run"}'
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


def _save_analyses(analyses: list[FrameAnalysis], path: Path):
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
