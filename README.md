# Video Understanding System

Processes any local video file **or YouTube/Bilibili URL** into a queryable knowledge database using:

- **FunASR** (paraformer-zh + fsmn-vad + ct-punc) for Chinese/multilingual STT with timestamps
- **PaddleOCR** (PP-OCRv4, subprocess-isolated) for fast on-frame text extraction
- **Local VLM via LM Studio** for scene understanding, slide structure, diagram description, and visual delta
- **ChromaDB** for semantic vector search
- **Local LLM via LM Studio** for RAG-based Q&A and fusion summaries

All models run fully offline after the first download. No cloud APIs required.

---

## Setup

### 1. System dependency

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Python environment

Recommended: use a dedicated conda environment (Python 3.11).

```bash
pip install -r requirements.txt
```

Key packages: `funasr`, `paddleocr`, `paddlepaddle`, `openai`, `chromadb`, `sentence-transformers`, `yt-dlp`, `Pillow`

### 3. Configure `config.py`

```python
# LM Studio endpoint
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
VLM_MODEL          = "qwen3.5-4b"   # vision model loaded in LM Studio
LLM_MODEL          = "qwen3.5-4b"   # text model for fusion + Q&A

# STT
FUNASR_MODEL       = "paraformer-zh"
FUNASR_DEVICE      = "cuda"          # or "cpu"
FUNASR_LANGUAGE    = "zh"            # "zh" | "en" | "auto"

# OCR
OCR_LANG           = "ch"            # "ch" = simplified Chinese + English
OCR_USE_GPU        = True

# Frame deduplication — consecutive frames more similar than this are skipped
FRAME_SIMILARITY_THRESHOLD = 0.90
```

### 4. Start LM Studio

Load a vision-capable model (e.g. Qwen2.5-VL, LLaVA), enable the local server on port 1234.

---

## Usage

### Process a local video file

```bash
python pipeline.py my_lecture.mp4
```

### Process a YouTube or Bilibili video (auto-download)

Pass just the short code — no URL required:

```bash
# YouTube: 11-character video ID
python pipeline.py dQw4w9WgXcQ

# Bilibili: BV code
python pipeline.py BV1GE411T7Wv
```

Full URLs also work if you prefer:

```bash
python pipeline.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python pipeline.py "https://www.bilibili.com/video/BV1GE411T7Wv"
```

The video is downloaded automatically via **yt-dlp** to `./video_db/_downloads/` at up to 720p MP4, then processed through the normal pipeline. Re-running the same code skips re-downloading if the file already exists.

> **Note**: Bilibili downloads use your browser's Edge cookies automatically. If cookie extraction fails (Windows DPAPI error), the downloader retries without cookies.

### Process and immediately query

```bash
python pipeline.py my_lecture.mp4 --query
```

### Force full reprocessing

```bash
python pipeline.py my_lecture.mp4 --force
```

---

## Output files

Each processed video gets its own directory under `./video_db/<stem>/`:

| File | Contents |
|---|---|
| `audio.wav` | Extracted 16 kHz mono audio |
| `transcript.json` | FunASR sentence segments with ms timestamps |
| `frame_schedule.json` | Adaptive frame selection plan |
| `frames/` | Extracted JPEG frames (selected only) |
| `frame_analyses.json` | Per-frame OCR + VLM results |
| `fused_segments.json` | Merged transcript + visual per segment |
| `timeline.json` | Full structured timeline + slide index |
| `chroma/` | ChromaDB vector store |

---

## Querying

### Interactive REPL

```bash
python query.py ./video_db/my_lecture
```

### One-shot CLI queries

```bash
python query.py ./video_db/my_lecture --summary
python query.py ./video_db/my_lecture --outline
python query.py ./video_db/my_lecture --slides
python query.py ./video_db/my_lecture --transcript
python query.py ./video_db/my_lecture --ask "What is the attention mechanism?"
python query.py ./video_db/my_lecture --knowledge "transformer architecture"
python query.py ./video_db/my_lecture --at 14:32 --question "What slide is shown here?"
```

### REPL commands

```
/summary            full video summary
/outline            topic outline from slide titles
/slides             list all slide changes with timestamps
/transcript         full spoken transcript
/at 05:30 <q>       query what was happening at a specific time
/knowledge <topic>  deep extraction on a topic
/quit               exit
<any question>      semantic search + RAG answer
```

---

## Architecture

```
Input: local file  OR  YouTube / Bilibili URL
  │
  ├─ yt-dlp ──► MP4 (≤720p)          [URL only]
  │
  ├─ ffmpeg ──► audio.wav
  │
  ▼
Phase 1: Speech-to-Text  (FunASR paraformer-zh)
  │  sentence segments with ms timestamps + pause durations
  │
  ▼
Phase 2a: Adaptive Frame Sampler
  │  trigger: sentence_end  (t − 200 ms)
  │  trigger: long pause midpoint  (>800 ms)
  │  trigger: 1 fps floor fallback
  │  → sparse frame schedule (≈40–120 frames for 30-min video)
  │
  ▼
Phase 2b: Dual-track Frame Analysis
  │
  ├─ Track A: PaddleOCR (subprocess)
  │    Fast dedicated text extraction — slides, code, labels.
  │    Runs in a separate process to avoid PyTorch/Paddle CUDA conflict.
  │    Similarity pre-filter: consecutive frames with perceptual
  │    similarity ≥ 0.90 are skipped entirely (no OCR, no VLM call).
  │
  └─ Track B: VLM  (LM Studio, sequential)
       scene description → slide JSON → diagram description → visual delta
       Images compressed to JPEG (quality 85, max 1280 px) before encoding.
       Identical frames skipped via content hash cache.
  │
  ▼
Phase 3: Temporal Fusion  (LLM)
  │  group N sentences → find best visual frame → LLM synthesis
  │  resolves deictic references ("as shown here" → actual slide content)
  │
  ▼
Phase 4: Database
  ├─ ChromaDB (vector store)  — semantic search, RAG
  └─ timeline.json            — timestamp lookup, slide index, full export
```

---

## Performance notes

- **Similarity dedup** skips redundant frames before any AI inference runs. On a typical lecture recording, 80–95% of frames may be skipped.
- **Sequential VLM calls** — one frame at a time, four sub-prompts each — avoids overwhelming a local LM Studio server with concurrent requests.
- **OCR subprocess isolation** prevents PyTorch and PaddlePaddle from conflicting over shared CUDA device registrations.
- Re-runs are fully incremental: transcript, frames, and analyses are each cached separately and only missing work is re-done.
- Reduce `FALLBACK_FPS_FLOOR` (e.g. to `0.5`) to halve the floor frame count for very long videos.

---

## Extending the system

```python
from core.database import VideoDatabase
from query.query_engine import QueryEngine
from openai import OpenAI
import config as cfg

db     = VideoDatabase("./video_db/my_lecture", cfg)
client = OpenAI(base_url=cfg.LM_STUDIO_BASE_URL, api_key=cfg.LM_STUDIO_API_KEY)
engine = QueryEngine(db, client, cfg)

# Semantic search
hits = db.search("gradient descent", n_results=5)

# Ask anything
answer = engine.ask("Summarise the part about backpropagation")

# Full summary
summary = engine.summarize(style="comprehensive")  # "brief" | "comprehensive" | "bullet"

# Get all segments mentioning a topic
deep = engine.extract_knowledge("attention mechanism")
```
