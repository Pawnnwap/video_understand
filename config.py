"""config.py — all tunable parameters in one place.
Edit this file to match your local setup.

Three model backends are used:
  • VLM  (vision)  — via the local opencode server calling the built-in free
                     vision model "opencode/mimo-v2.5-free".  No base_url
                     or API key — opencode hosts the model internally.
  • LLM  (text)    — LOCAL, via LM Studio (e.g. qwen3.5-4b).  Used for phase-3
                     fusion text summaries and the query engine.
  • OCR            — LOCAL PaddleOCR (subprocess), unchanged.

Environment variables take precedence over defaults:
  VLM:   VLM_MODEL  (provider/model id; default opencode/mimo-v2.5-free)
  LLM:   LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LLM_MODEL
"""

import os
from pathlib import Path

# ── VLM (vision track) — opencode local server (free vision models) ──────────
# No endpoint or key: opencode hosts the model internally.  Set VLM_MODEL to
# a provider/model id that opencode exposes (see `opencode models opencode`).
VLM_MODEL = os.environ.get("VLM_MODEL", "opencode/mimo-v2.5-free")
# Optional reasoning variant for mimo (omit for default): low|medium|high|max
VLM_VARIANT = os.environ.get("VLM_VARIANT", "") or None
# Fixed port for the spawned opencode server (0 = random).
OPENCODE_SERVER_PORT = int(os.environ.get("OPENCODE_SERVER_PORT", "0") or 0)

# ── Phase-3 text LLM ("pure mode" — same opencode server, no image) ────────
# The fusion stage also runs through opencode, sending text-only messages
# (no file attachment).  Defaults to the same opencode model as the vision
# track; override VLM_LLM_MODEL / VLM_LLM_VARIANT to use a different model
# (e.g. a smaller reasoning variant) for summarisation.
VLM_LLM_MODEL = os.environ.get("VLM_LLM_MODEL", VLM_MODEL)
VLM_LLM_VARIANT = os.environ.get("VLM_LLM_VARIANT", "") or None

# ── LLM (text track) — local, via LM Studio ─────────────────────────────────
LLM_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1235/v1")
LLM_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-4b")

# Legacy aliases (kept for any caller still referencing LM_STUDIO_BASE_URL /
# LM_STUDIO_API_KEY directly — they now map to the local LLM endpoint).
LM_STUDIO_BASE_URL = LLM_BASE_URL
LM_STUDIO_API_KEY = LLM_API_KEY


# FunASR model ids — must be the full ModelScope ids (cached under
# ~/.cache/modelscope/hub/models/<id>).  The short aliases ("paraformer-zh" etc.)
# are not registered by funasr >= 1.3, so the full paths are required.
FUNASR_MODEL = os.environ.get(
    "FUNASR_MODEL",
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
FUNASR_VAD_MODEL = os.environ.get("FUNASR_VAD_MODEL", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
FUNASR_PUNC_MODEL = os.environ.get("FUNASR_PUNC_MODEL", "iic/punc_ct-transformer_cn-en-common-vocab471067-large")
FUNASR_DEVICE = os.environ.get("FUNASR_DEVICE", "cuda")
FUNASR_LANGUAGE = os.environ.get("FUNASR_LANGUAGE", "zh")
STT_SENTENCE_SPLIT_GAP_MS = int(os.environ.get("STT_SENTENCE_SPLIT_GAP_MS", "500"))


OCR_MODEL_NAME = "PP-OCRv5_mobile"
OCR_LANG = "ch"
OCR_USE_GPU = True
OCR_MIN_CONFIDENCE = 0.6
OCR_TIMEOUT_S = 60


SENTENCE_END_OFFSET_MS = 200
LONG_PAUSE_THRESHOLD_MS = 800
FALLBACK_FPS_FLOOR = 0.2


FRAME_MAX_DIM = 768
FRAME_QUALITY = 75


FRAME_SIMILARITY_THRESHOLD = 0.90


OCR_RICH_TEXT_MIN_LINES = 3


VLM_MAX_TOKENS = 512
VLM_TEMPERATURE = 0.1
VLM_CALL_TIMEOUT_S = 120
LLM_CALL_TIMEOUT_S = 60


FFMPEG_TIMEOUT_S = 300
FFMPEG_EXTRACTION_TIMEOUT_S = 60
FUNASR_TIMEOUT_S = 0


RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
RETRY_JITTER_FACTOR = 0.25


FUSION_SEGMENT_SIZE = 5
LLM_MAX_TOKENS_FUSION = 512
LLM_TEMPERATURE_FUSION = 0.2


DOWNLOAD_MAX_DURATION_SEC = 0


DB_DIR = str((Path(__file__).parent / "video_db").resolve())
CHROMA_COLLECTION = "segments"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
