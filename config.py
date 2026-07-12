"""Runtime configuration for the video-understanding pipeline.

Vision analysis, fusion, and querying use a local OpenCode server. OCR, speech
recognition, and embeddings remain local. Environment variables override the
model settings below.
"""

import os
from pathlib import Path


# OpenCode vision model. Frame analysis needs no deep reasoning, so thinking
# stays at the lowest effort the model exposes ("low" — opencode has no
# fully-off variant).
VLM_MODEL = os.environ.get("VLM_MODEL", "opencode/mimo-v2.5-free")
VLM_VARIANT = os.environ.get("VLM_VARIANT", "") or "low"
OPENCODE_SERVER_PORT = int(os.environ.get("OPENCODE_SERVER_PORT", "0") or 0)

# OpenCode text model used for fusion, queries, and web crosschecking.
# Text-LLM work defaults to the highest reasoning effort the opencode free
# models expose ("high"). Env vars / CLI flags override both variants.
VLM_LLM_MODEL = os.environ.get("VLM_LLM_MODEL", VLM_MODEL)
VLM_LLM_VARIANT = os.environ.get("VLM_LLM_VARIANT", "") or "high"


# FunASR model paths. Local model directories take precedence over model IDs.
_FUNASR_LOCAL_ROOT = str(Path(__file__).parent / "models" / "iic")


def _funasr_path(env_key: str, default_name: str) -> str:
    val = os.environ.get(env_key, "")
    if val and (Path(val).is_absolute() or "/" in val or os.sep in val):
        return val
    name = val or default_name
    local = Path(_FUNASR_LOCAL_ROOT) / name
    return str(local) if (local / "model.pt").exists() else name


FUNASR_MODEL = _funasr_path(
    "FUNASR_MODEL",
    "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
FUNASR_VAD_MODEL = _funasr_path("FUNASR_VAD_MODEL", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
FUNASR_PUNC_MODEL = _funasr_path("FUNASR_PUNC_MODEL", "punc_ct-transformer_cn-en-common-vocab471067-large")
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

FFMPEG_TIMEOUT_S = 300
FFMPEG_EXTRACTION_TIMEOUT_S = 60
FUNASR_TIMEOUT_S = 0

RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
RETRY_JITTER_FACTOR = 0.25

FUSION_SEGMENT_SIZE = 5

DOWNLOAD_MAX_DURATION_SEC = 0

DB_DIR = str((Path(__file__).parent / "video_db").resolve())
CHROMA_COLLECTION = "segments"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/paraphrase-multilingual-MiniLM-L12-v2")
