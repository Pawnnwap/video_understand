# Video Understanding Pipeline

A comprehensive video analysis system that transforms lecture/recording videos into queryable knowledge bases using speech-to-text, visual analysis, and semantic search.

## Features

- **Speech-to-Text**: FunASR (paraformer-zh) for Chinese-native transcription with timestamps
- **Visual Analysis**: VLM frame analysis + PaddleOCR for slide content extraction
- **Semantic Search**: ChromaDB vector store for knowledge retrieval
- **Interactive CLI**: Query processed videos with natural language

## Requirements

- Python 3.10+
- ffmpeg (for audio extraction)
- LM Studio (for VLM/LLM inference)
- CUDA-capable GPU recommended

## Installation

```bash
# Clone repository
git clone <repo-url>
cd video_summarize

# Create conda environment
conda create -n video python=3.10
conda activate video

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Configuration is via `config.py` or environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `LM_STUDIO_API_KEY` | `lm-studio` | API key |
| `VLM_MODEL` | `qwen3.5-4b` | Vision model name |
| `LLM_MODEL` | `qwen3.5-4b` | Language model name |

### FunASR (Speech-to-Text)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FUNASR_MODEL` | `paraformer-zh` | ASR model |
| `FUNASR_VAD_MODEL` | `fsmn-vad` | Voice activity detection model |
| `FUNASR_PUNC_MODEL` | `ct-punc` | Punctuation model |
| `FUNASR_DEVICE` | `cuda` | Device (cuda/cpu) |
| `FUNASR_LANGUAGE` | `zh` | Language code |
| `FUNASR_TIMEOUT_S` | `0` | Timeout (0=unlimited) |
| `STT_SENTENCE_SPLIT_GAP_MS` | `500` | Sentence split gap threshold (ms) |

### OCR (PaddleOCR)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OCR_MODEL_NAME` | `PP-OCRv5_mobile` | OCR model |
| `OCR_LANG` | `ch` | OCR language |
| `OCR_USE_GPU` | `True` | Use GPU for OCR |
| `OCR_MIN_CONFIDENCE` | `0.6` | Minimum confidence threshold |
| `OCR_TIMEOUT_S` | `60` | OCR subprocess timeout |
| `OCR_RICH_TEXT_MIN_LINES` | `3` | Min lines for rich text detection |

### Frame Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SENTENCE_END_OFFSET_MS` | `200` | Offset before sentence end (ms) |
| `LONG_PAUSE_THRESHOLD_MS` | `800` | Long pause detection threshold (ms) |
| `FALLBACK_FPS_FLOOR` | `0.2` | Fallback frame rate for silent stretches |
| `FRAME_MAX_DIM` | `768` | Maximum frame dimension (pixels) |
| `FRAME_QUALITY` | `75` | JPEG quality |
| `FRAME_SIMILARITY_THRESHOLD` | `0.90` | Frame similarity threshold |

### VLM / LLM

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VLM_MAX_TOKENS` | `512` | VLM max output tokens |
| `VLM_TEMPERATURE` | `0.1` | VLM temperature |
| `VLM_CALL_TIMEOUT_S` | `120` | VLM HTTP call timeout |
| `LLM_CALL_TIMEOUT_S` | `60` | LLM fusion/query timeout |
| `LLM_MAX_TOKENS_FUSION` | `512` | LLM max tokens for fusion |
| `LLM_TEMPERATURE_FUSION` | `0.2` | LLM temperature for fusion |

### Retry Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRY_MAX_ATTEMPTS` | `4` | Max retry attempts |
| `RETRY_BASE_DELAY_S` | `2.0` | Base retry delay (seconds) |
| `RETRY_MAX_DELAY_S` | `30.0` | Max retry delay (seconds) |
| `RETRY_JITTER_FACTOR` | `0.25` | Jitter factor for retries |

### Database & Embedding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DB_DIR` | `./video_db` | Database storage directory |
| `CHROMA_COLLECTION` | `segments` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `FUSION_SEGMENT_SIZE` | `5` | Segments per fusion batch |

### Download & FFmpeg

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DOWNLOAD_MAX_DURATION_SEC` | `0` | Max download duration (0=unlimited) |
| `FFMPEG_TIMEOUT_S` | `300` | ffmpeg subprocess timeout |
| `FFMPEG_EXTRACTION_TIMEOUT_S` | `60` | Frame extraction timeout |
CLI overrides (highest priority):

```bash
python cli.py video.mp4 --base-url http://localhost:1234/v1 --vlm-model qwen3.5-4b
python pipeline.py URL --api-key your-key --llm-model qwen3.5-4b
python query.py ./video_db/project --model qwen3.5-4b
```

## Usage

### Process a Video

```bash
# Local file
python cli.py lecture.mp4

# YouTube/Bilibili URL
python cli.py https://www.youtube.com/watch?v=...
python cli.py BV1GE411T7Wv
python cli.py dQw4w9WgXcQ  # YouTube ID

# Direct pipeline
python pipeline.py video.mp4 --force  # Force reprocessing
```

### Query a Processed Video

```bash
# Interactive REPL
python cli.py
python query.py ./video_db/my_lecture

# One-shot queries
python query.py ./video_db/my_lecture --ask "What is the main topic?"
python query.py ./video_db/my_lecture --summary
python query.py ./video_db/my_lecture --at 05:30 --question "What slide is shown?"
```

### CLI Commands

Inside the interactive workspace:

| Command | Description |
|---------|-------------|
| `list` / `ls` | List processed projects |
| `open <name|#>` | Enter a project |
| `process <path>` | Process new video |
| `<BV/URL>` | Download, process, and open |

Inside a project:

| Command | Description |
|---------|-------------|
| `/summary` | Comprehensive summary |
| `/headline` | One-line headline |
| `/brief` | 3-5 sentence overview |
| `/outline` | Topic outline |
| `/slides` | List slide changes |
| `/transcript` | Full transcript |
| `/at MM:SS [q]` | Query at timestamp |
| `/knowledge <topic>` | Deep extraction |
| `<any text>` | Semantic search |

## Architecture

```
video.mp4
    │
    ▼ [Phase 1: STT]
transcript.json  (sentence segments with timestamps)
    │
    ▼ [Phase 2a: Frame Sampling]
frame_schedule.json  (adaptive triggers)
    │
    ▼ [Phase 2b: Visual Analysis]
frame_analyses.json  (OCR + VLM per frame)
    │
    ▼ [Phase 3: Fusion]
fused_segments.json  (speech + visual merged)
    │
    ▼ [Phase 4: Database]
chroma/  (vector store)
timeline.json  (structured timeline)
```

## Timeout Configuration

Timeouts configurable in `config.py` (see Configuration section above).
## Project Structure

```
video_summarize/
├── cli.py                    # Interactive workspace CLI
├── pipeline.py               # Main processing pipeline
├── query.py                  # Standalone query interface
├── config.py                 # Configuration (all tunable parameters)
├── downloader.py             # Video download (yt-dlp)
├── protocol.md               # Development protocol
├── requirements.txt          # Dependencies
├── core/
│   ├── lang.py               # Language detection
│   ├── stt.py                # Speech-to-text (FunASR)
│   ├── fusion.py             # Speech-visual fusion
│   ├── database.py           # ChromaDB + timeline
│   └── vision/
│       ├── __init__.py       # Vision module init
│       ├── frame_sampler.py  # Adaptive frame extraction
│       ├── vlm_analyser.py   # VLM frame analysis
│       └── ocr_worker.py     # OCR subprocess (PaddleOCR)
├── query/
│   ├── __init__.py           # Query module init
│   └ query_engine.py       # RAG query engine
├── utils/
│   ├── __init__.py           # Utils module init
│   ├── video.py              # Video utilities
│   └ retry.py              # Retry utilities
└── video_db/                 # Processed project storage
```

## Troubleshooting

**ffmpeg not found**: Install ffmpeg and add to PATH.

**CUDA out of memory**: Set `FUNASR_DEVICE=cpu` in config.py.

**LM Studio connection failed**: Verify LM Studio is running and model loaded.

**PaddleOCR fails**: Set `OCR_USE_GPU=False` if no CUDA.

## License

MIT
