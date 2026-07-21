# Video Understanding Pipeline

A comprehensive video analysis system that transforms lecture/recording videos into queryable knowledge bases using speech-to-text, visual analysis, and agent-driven querying.

## Features

- **Speech-to-Text**: FunASR (paraformer-zh) for Chinese-native transcription with timestamps
- **Visual Analysis**: VLM frame analysis + RapidOCR (ONNX) for slide content extraction
- **Agent-Driven Querying**: no vector database — short videos are fed to the model whole; long videos are searched by a read-only `video-qa` OpenCode agent that greps/reads the video's context files
- **Agentic Web Fact-Checking**: Crosscheck claims with an OpenCode agent that searches and reads web sources
- **Interactive CLI**: Query processed videos with natural language

## Requirements

- Python 3.10+
- ffmpeg (for audio extraction)
- [opencode](https://opencode.ai) CLI (for VLM/LLM inference, free built-in models)
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
| `VLM_MODEL` | `opencode/mimo-v2.5-free` | Vision model — the only vision job (frame analysis). agnes has no vision, so this stays on mimo |
| `VLM_VARIANT` | `low` | VLM reasoning variant (low/medium/high) — kept minimal for frame analysis |
| `OPENCODE_SERVER_PORT` | `0` (random) | opencode server port |
| `LLM_MODEL` | `agnes/agnes-2.0-flash` | Text model for ALL pure-text work (fusion, queries, summaries, claim extraction, read/web agents). Independent of the vision model |
| `LLM_VARIANT` | `none` (disabled) | Text-model reasoning variant. agnes exposes no thinking arg; set to low/medium/high only if you point `LLM_MODEL` at a thinking-capable model |

`/crosscheck` starts the project-local `web-crosscheck` OpenCode agent. It is
restricted to `websearch` and `webfetch`; the server enables Exa web search via
`OPENCODE_ENABLE_EXA=1` automatically. The selected OpenCode model must support
tool calling.

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

### OCR (RapidOCR, ONNX)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OCR_MODEL_NAME` | `PP-OCRv5_mobile` | OCR model name (informational) |
| `OCR_LANG` | `ch` | OCR language |
| `OCR_USE_GPU` | `True` | Use GPU for OCR (onnxruntime CUDA) |
| `OCR_GPU_MAX_WORKERS` | `1` | Max concurrent GPU OCR workers; keep low to avoid ONNXRuntime CUDA OOM |
| `OCR_MAX_DIM` | `1600` | Resize OCR input longest side; original frames are unchanged; set 0 to disable |
| `OCR_RESIZE_QUALITY` | `92` | JPEG quality for temporary resized OCR images |
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

### Retry Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRY_MAX_ATTEMPTS` | `4` | Max retry attempts |
| `RETRY_BASE_DELAY_S` | `2.0` | Base retry delay (seconds) |
| `RETRY_MAX_DELAY_S` | `30.0` | Max retry delay (seconds) |
| `RETRY_JITTER_FACTOR` | `0.25` | Jitter factor for retries |

### Storage & Fusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DB_DIR` | `./video_db` | Project storage directory (holds `timeline.json` + `context.md` per video) |
| `FUSION_SEGMENT_SIZE` | `5` | Segments per fusion batch |

### Agent-Driven Querying

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QA_AGENT` | `video-qa` | Read-only OpenCode agent used to search long videos |
| `QA_CONTEXT_TOKEN_FRACTION` | `0.10` | Feed the whole video inline when its estimated tokens are below this fraction of the text model's context window; otherwise let the agent grep/read |
| `QA_CONTEXT_LIMIT_FALLBACK` | `128000` | Context window assumed when neither `MODEL_CONTEXT_LIMITS` (agnes is pinned to 512k there) nor the opencode server reports one |
| `QA_IDLE_TIMEOUT_S` | `300` | Abort a `video-qa` session after this many seconds with no activity |

### Download & FFmpeg

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DOWNLOAD_MAX_DURATION_SEC` | `0` | Max download duration (0=unlimited) |
| `FFMPEG_TIMEOUT_S` | `300` | ffmpeg subprocess timeout |
| `FFMPEG_EXTRACTION_TIMEOUT_S` | `60` | Frame extraction timeout |
CLI overrides (highest priority):

```bash
python cli.py video.mp4 --vlm-model opencode/mimo-v2.5-free --text-model opencode/mimo-v2.5-free
python pipeline.py URL --text-model opencode/mimo-v2.5-free
python query.py ./video_db/project --text-model opencode/mimo-v2.5-free --ask "What is the main topic?"
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

### Batch Processing

```bash
# Process multiple videos sequentially and generate reports
python process_queue.py BV1iatTeGENk dQw4w9WgXcQ https://...

# From a queue file (one source per line; blank lines and #-comments ignored)
python process_queue.py --file queue.txt

# Custom output directory and crosscheck depth
python process_queue.py BV1... --out-dir reports --crosscheck-n 3

# Override model settings
python process_queue.py BV1... --text-model opencode/mimo-v2.5-free
```

Each video runs the full pipeline, then generates a markdown report with summary and web fact-checking results. Failed items are skipped and summarized at the end.

### Query a Processed Video

```bash
# Interactive project session
python cli.py
python cli.py ./video_db/my_lecture

# One-shot query wrapper
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
| `/summary [style]` | Whole-video summary: comprehensive (default), brief, headline |
| `/outline` | Topic outline |
| `/slides` | List slide changes |
| `/transcript` | Full transcript |
| `/at MM:SS [q]` | Query at timestamp |
| `/crosscheck [n]` | Fact-check top N claims (default 5) |
| `<any question>` | Ask anything in natural language (recommended) — direct answers or topic deep dives |

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
timeline.json  (structured timeline)
context.md     (grep-friendly whole-video context for querying)
```

## Timeout Configuration

Timeouts configurable in `config.py` (see Configuration section above).
## Project Structure

```
video_summarize/
├── cli.py                    # Interactive workspace CLI
├── pipeline.py               # Main processing pipeline
├── process_queue.py           # Batch queue runner with summary + crosscheck reports
├── query.py                  # Standalone query interface
├── config.py                 # Configuration (all tunable parameters)
├── downloader.py             # Video download (yt-dlp)
├── requirements.txt          # Dependencies
├── core/
│   ├── lang.py               # Language detection
│   ├── stt.py                # Speech-to-text (FunASR)
│   ├── fusion.py             # Speech-visual fusion
│   ├── database.py           # timeline.json + context.md writer/reader
│   └── vision/
│       ├── __init__.py       # Vision module init
│       ├── frame_sampler.py  # Adaptive frame extraction
│       ├── vlm_analyser.py   # VLM frame analysis + inline RapidOCR
│       └── opencode_vlm.py   # opencode serve subprocess + HTTP client
├── query/
│   ├── __init__.py           # Query module init
│   ├── query_engine.py       # agent-driven query engine (inline / video-qa search)
│   └── crosscheck.py         # Web fact-checking pipeline
├── utils/
│   ├── __init__.py           # Utils module init
│   ├── video.py              # Video utilities
│   ├── retry.py              # Retry utilities
│   └── logging_setup.py      # Logging configuration
├── models/                   # Bundled offline models
└── video_db/                 # Processed project storage
```

## Troubleshooting

**ffmpeg not found**: Install ffmpeg and add to PATH.

**CUDA out of memory**: Set `FUNASR_DEVICE=cpu` in config.py.

**LM Studio connection failed**: Verify LM Studio is running and model loaded.

**RapidOCR fails**: Set `OCR_USE_GPU=False` if no CUDA or onnxruntime-gpu issues. If GPU memory is tight, keep `OCR_GPU_MAX_WORKERS=1`, lower `OCR_MAX_DIM`, or lower `OCR_PARALLEL_FRACTION`; CUDA OOM falls back to CPU automatically.

## License

MIT
