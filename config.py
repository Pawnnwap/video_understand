"""
config.py — all tunable parameters in one place.
Edit this file to match your local setup.
"""

# ─────────────────────────────────────────────
#  LM Studio  (OpenAI-compatible endpoint)
# ─────────────────────────────────────────────
LM_STUDIO_BASE_URL  = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY   = "lm-studio"

# Model identifiers exactly as they appear in LM Studio
VLM_MODEL           = "qwen3.5-4b"
LLM_MODEL           = "qwen3.5-4b"

# ─────────────────────────────────────────────
#  STT  (FunASR — paraformer-zh, Chinese-native)
# ─────────────────────────────────────────────
# paraformer-zh: best offline Chinese ASR model
# SenseVoiceSmall: multilingual alternative (faster)
FUNASR_MODEL              = "paraformer-zh"
FUNASR_VAD_MODEL    = "fsmn-vad"
FUNASR_PUNC_MODEL   = "ct-punc"
FUNASR_DEVICE       = "cuda"        # "cuda" or "cpu"
FUNASR_LANGUAGE           = "zh"          # "zh" | "en" | "auto"
# Split threshold: gap between consecutive characters >= this → new sentence
# Primary split is on 。！？ punctuation; this is the secondary fallback.
STT_SENTENCE_SPLIT_GAP_MS = 500

# ─────────────────────────────────────────────
#  OCR  (PaddleOCR PP-OCRv5 — dedicated, fast)
# ─────────────────────────────────────────────
# PaddleOCR runs locally, handles Chinese+English mixed text natively,
# and is far lighter than asking a VLM to do OCR.
# "PP-OCRv5_mobile" = fastest (recommended); "PP-OCRv5_server" = highest accuracy
OCR_MODEL_NAME      = "PP-OCRv5_mobile"
OCR_LANG            = "ch"          # "ch" = simplified Chinese + English
OCR_USE_GPU         = True          # set False if no CUDA
OCR_MIN_CONFIDENCE  = 0.6           # discard low-confidence detections

# ─────────────────────────────────────────────
#  Frame sampling thresholds
# ─────────────────────────────────────────────
SENTENCE_END_OFFSET_MS   = 200
LONG_PAUSE_THRESHOLD_MS  = 800
FALLBACK_FPS_FLOOR       = 0.2   # 1 frame per 5 s — sentence triggers cover the rest

# ─────────────────────────────────────────────
#  Frame compression  (WebP with quality-based encoding)
# ─────────────────────────────────────────────
FRAME_MAX_DIM            = 768      # longest side cap before VLM (px); smaller = fewer context tokens
FRAME_QUALITY            = 75       # JPEG quality for VLM encoding

# ─────────────────────────────────────────────
#  Frame deduplication  (similarity skip)
# ─────────────────────────────────────────────
# Consecutive frames with perceptual similarity >= threshold skip OCR+VLM entirely.
# 0.90 filters out minor cursor movements / clock ticks.
FRAME_SIMILARITY_THRESHOLD = 0.90

# ─────────────────────────────────────────────
#  Adaptive VLM gate  (OCR-first logic)
# ─────────────────────────────────────────────
# After OCR runs, VLM is skipped for text-rich frames.
# A frame is "text-rich" when its OCR line count >= max(floor, running_avg).
# Floor prevents skipping VLM on frames with very few lines just because the
# video average happens to be low (e.g., mostly camera footage with 0-1 lines).
OCR_RICH_TEXT_MIN_LINES = 3

# ─────────────────────────────────────────────
#  VLM call settings
# ─────────────────────────────────────────────
VLM_MAX_TOKENS          = 512       # reduced: OCR is handled by PaddleOCR now
VLM_TEMPERATURE         = 0.1
VLM_PARALLEL_WORKERS    = 4         # concurrent VLM calls — tune to VRAM

# ─────────────────────────────────────────────
#  Retry settings
# ─────────────────────────────────────────────
RETRY_MAX_ATTEMPTS      = 4         # total attempts (1 original + 3 retries)
RETRY_BASE_DELAY_S      = 2.0       # initial back-off (doubles each attempt)
RETRY_MAX_DELAY_S       = 30.0      # cap on any single wait
RETRY_JITTER_FACTOR     = 0.25      # randomise ±25% of computed delay

# ─────────────────────────────────────────────
#  Fusion / summarisation
# ─────────────────────────────────────────────
FUSION_SEGMENT_SIZE     = 5
LLM_MAX_TOKENS_FUSION   = 512
LLM_TEMPERATURE_FUSION  = 0.2

# ─────────────────────────────────────────────
#  Downloader
# ─────────────────────────────────────────────
# Max video duration in seconds when downloading from URL (0 = no limit)
DOWNLOAD_MAX_DURATION_SEC = 0

# ─────────────────────────────────────────────
#  Database / output
# ─────────────────────────────────────────────
DB_DIR                  = "./video_db"
CHROMA_COLLECTION       = "segments"
EMBEDDING_MODEL         = "paraphrase-multilingual-MiniLM-L12-v2"
