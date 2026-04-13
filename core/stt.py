"""
core/stt.py — Phase 1
Runs FunASR (paraformer-zh + fsmn-vad + ct-punc) on the extracted audio
stream and returns sentence segments with millisecond timestamps.

Segmentation strategy
─────────────────────
FunASR with VAD often collapses the whole audio into one large segment.
We re-segment it ourselves using the character-level timestamps it provides:

  1. Primary split  — after sentence-ending punctuation (。！？\u3002\uff01\uff1f)
     ct-punc adds these WITH timestamps, so they appear in the word list.

  2. Secondary split — after inter-character gaps >= STT_SENTENCE_SPLIT_GAP_MS
     Catches natural pauses in segments where no sentence-end punct appears.

This typically yields 6-10 sentences/minute for Mandarin lecture content,
giving the frame sampler meaningful sentence_end and long_pause triggers.
"""

from __future__ import annotations
import concurrent.futures
import json
import subprocess
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from utils.retry import RetryConfig, retry_sync

log = logging.getLogger(__name__)

_AUDIO_RETRY = RetryConfig(max_attempts=4, base_delay_s=1.0)
_STT_RETRY   = RetryConfig(max_attempts=3, base_delay_s=2.0)

# Chinese sentence-ending punctuation (full-width and half-width)
_SENT_END_PUNCT = frozenset('。！？\u3002\uff01\uff1f')


@dataclass
class SentenceSegment:
    id:             int
    start_ms:       int
    end_ms:         int
    text:           str
    words:          List[dict] = field(default_factory=list)
    pause_after_ms: int        = 0
    trigger_frames: List[int]  = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str, out_dir: Path) -> Path:
    """Use ffmpeg to pull 16 kHz mono WAV.  Retries up to 4 times."""
    audio_path = out_dir / "audio.wav"
    if audio_path.exists():
        log.info("Audio already extracted, skipping.")
        return audio_path

    def _run():
        log.info(f"Extracting audio from {video_path} …")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=getattr(cfg, "FFMPEG_TIMEOUT_S", 300),
            )
        except subprocess.TimeoutExpired:
            raise OSError(f"ffmpeg audio extraction timed out after {getattr(cfg, 'FFMPEG_TIMEOUT_S', 300)}s")
        if result.returncode != 0:
            raise OSError(f"ffmpeg audio extraction failed:\n{result.stderr[-500:]}")
        return audio_path

    return retry_sync(_run, cfg=_AUDIO_RETRY, label="extract_audio")


def transcribe(audio_path: Path, cfg) -> List[SentenceSegment]:
    """
    Run FunASR and return a list of sentence-level segments.

    FunASR produces character-level timestamps; this function re-segments
    them into sentences using punctuation and pause gaps.
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise ImportError("funasr not installed. Run:  pip install funasr")

    model = _load_funasr_model(cfg)

    def _run_transcription():
        log.info("Transcribing with FunASR …")
        return model.generate(
            input        = str(audio_path),
            batch_size_s = 300,
            language     = cfg.FUNASR_LANGUAGE,
        )

    raw_result = retry_sync(_run_transcription, cfg=_STT_RETRY, label="funasr_transcribe")
    log.info(f"Raw FunASR segments: {len(raw_result)}")

    # ── Build flat word list from all raw segments ────────────────────────
    all_words: List[dict] = []
    for seg in raw_result:
        text      = seg.get("text", "")
        timestamp = seg.get("timestamp", [])   # [[start_ms, end_ms], ...] per char
        if not text or not timestamp:
            continue
        for ch, ts in zip(text, timestamp):
            if not ch.strip():          # skip pure whitespace
                continue
            all_words.append({
                "word":     ch,
                "start_ms": int(ts[0]),
                "end_ms":   int(ts[1]),
            })

    if not all_words:
        log.warning("FunASR produced no timestamped words.")
        return []

    # ── Re-segment into sentences ─────────────────────────────────────────
    gap_ms    = getattr(cfg, "STT_SENTENCE_SPLIT_GAP_MS", 500)
    word_chunks = _split_into_sentences(all_words, gap_ms)
    log.info(
        f"Re-segmented {len(all_words)} chars → {len(word_chunks)} sentences "
        f"(gap_threshold={gap_ms}ms)"
    )

    # ── Build SentenceSegment list ────────────────────────────────────────
    sentences: List[SentenceSegment] = []
    for chunk in word_chunks:
        text     = "".join(w["word"] for w in chunk)
        start_ms = chunk[0]["start_ms"]
        end_ms   = chunk[-1]["end_ms"]
        sentences.append(SentenceSegment(
            id       = len(sentences),
            start_ms = start_ms,
            end_ms   = end_ms,
            text     = text,
            words    = chunk,
        ))

    # ── Fill pause_after_ms between consecutive sentences ────────────────
    for i in range(len(sentences) - 1):
        sentences[i].pause_after_ms = max(
            0, sentences[i + 1].start_ms - sentences[i].end_ms
        )

    log.info(f"Produced {len(sentences)} sentence segments.")
    return sentences


def _split_into_sentences(words: List[dict], gap_ms: int) -> List[List[dict]]:
    """
    Split a flat word list into sentence chunks.

    Split triggers (in order of priority):
      1. Word is a sentence-ending punctuation mark (。！？ etc.)
      2. Gap between word[i].end_ms and word[i+1].start_ms >= gap_ms
    """
    if not words:
        return []

    chunks: List[List[dict]] = []
    current: List[dict] = []

    for i, w in enumerate(words):
        current.append(w)

        # Trigger 1: sentence-ending punctuation
        if w["word"] in _SENT_END_PUNCT:
            chunks.append(current)
            current = []
            continue

        # Trigger 2: gap before next word
        if i + 1 < len(words):
            gap = words[i + 1]["start_ms"] - w["end_ms"]
            if gap >= gap_ms:
                chunks.append(current)
                current = []

    if current:
        chunks.append(current)

    return chunks


def _load_funasr_model(cfg):
    """Load FunASR model, falling back gracefully to CPU if CUDA fails."""
    from funasr import AutoModel

    device = getattr(cfg, "FUNASR_DEVICE", "cuda")

    def _try_load(dev):
        log.info(f"Loading FunASR '{cfg.FUNASR_MODEL}' on {dev} …")
        return AutoModel(
            model      = cfg.FUNASR_MODEL,
            vad_model  = cfg.FUNASR_VAD_MODEL,
            punc_model = cfg.FUNASR_PUNC_MODEL,
            device     = dev,
        )

    try:
        return _try_load(device)
    except Exception as e:
        log.warning(f"Primary FunASR load failed ({e}), falling back to CPU.")
        try:
            return _try_load("cpu")
        except Exception as e2:
            raise RuntimeError(f"Could not load FunASR model: {e2}") from e2


# ─────────────────────────────────────────────────────────────────────────────

def save_transcript(sentences: List[SentenceSegment], out_dir: Path) -> Path:
    path = out_dir / "transcript.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in sentences], f, ensure_ascii=False, indent=2)
    log.info(f"Transcript saved → {path}")
    return path


def load_transcript(out_dir: Path) -> Optional[List[SentenceSegment]]:
    path = out_dir / "transcript.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [SentenceSegment(**d) for d in data]
