"""core/vision — visual processing sub-package (phases 2a & 2b)

Modules
-------
frame_sampler   Build STT-driven frame schedule and extract frames via ffmpeg.
vlm_analyser    Analyse frames with VLM (scene / slide / diagram / delta) + OCR.
ocr_worker      Standalone subprocess runner for PaddleOCR (avoids CUDA conflict).
"""

from core.vision.frame_sampler import (
    FrameRequest,
    build_frame_schedule,
    extract_frames,
    frame_hash,
    save_schedule,
)
from core.vision.vlm_analyser import (
    FrameAnalysis,
    analyse_all_frames,
    analyse_frame,
    run_ocr,
)

__all__ = [
    "FrameRequest",
    "build_frame_schedule",
    "extract_frames",
    "frame_hash",
    "save_schedule",
    "FrameAnalysis",
    "analyse_all_frames",
    "analyse_frame",
    "run_ocr",
]
