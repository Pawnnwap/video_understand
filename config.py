"""config.py — all tunable parameters in one place.
Edit this file to match your local setup.

Environment variables take precedence over defaults:
  LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, VLM_MODEL, LLM_MODEL
"""

import os
from pathlib import Path

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")

VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5-4b")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-4b")


FUNASR_MODEL = "paraformer-zh"
FUNASR_VAD_MODEL = "fsmn-vad"
FUNASR_PUNC_MODEL = "ct-punc"
FUNASR_DEVICE = "cuda"
FUNASR_LANGUAGE = "zh"
STT_SENTENCE_SPLIT_GAP_MS = 500


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
