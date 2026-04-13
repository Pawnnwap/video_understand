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

Timeouts are configurable in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VLM_CALL_TIMEOUT_S` | 120 | VLM HTTP call timeout |
| `LLM_CALL_TIMEOUT_S` | 60 | LLM fusion/query timeout |
| `OCR_TIMEOUT_S` | 60 | OCR subprocess timeout |
| `FFMPEG_TIMEOUT_S` | 300 | ffmpeg subprocess timeout |
| `FUNASR_TIMEOUT_S` | 0 | FunASR timeout (0=unlimited) |

## Project Structure

```
video_summarize/
├── cli.py              # Interactive workspace CLI
├── pipeline.py         # Main processing pipeline
├── query.py            # Standalone query interface
├── config.py           # Configuration
├── downloader.py       # Video download (yt-dlp)
├── retry.py            # Retry utilities
├── core/
│   ├── stt.py          # Speech-to-text (FunASR)
│   ├── frame_sampler.py # Adaptive frame extraction
│   ├── vlm_analyser.py # Visual analysis
│   ├── fusion.py       # Speech-visual fusion
│   ├── database.py     # ChromaDB + timeline
│   └── ocr_worker.py   # OCR subprocess
├── query/
│   └── query_engine.py # RAG query engine
├── utils/
│   └── retry.py        # Retry re-exports
└── video_db/           # Processed project storage
```

## Troubleshooting

**ffmpeg not found**: Install ffmpeg and add to PATH.

**CUDA out of memory**: Set `FUNASR_DEVICE=cpu` in config.py.

**LM Studio connection failed**: Verify LM Studio is running and model loaded.

**PaddleOCR fails**: Set `OCR_USE_GPU=False` if no CUDA.

## License

MIT
