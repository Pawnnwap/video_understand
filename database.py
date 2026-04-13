"""
core/database.py — Phase 4
Persists fused segments into two complementary stores:
  1. ChromaDB vector store  — semantic search, RAG, similarity queries
  2. Structured JSON timeline — timestamp lookup, slide index, full export
"""

from __future__ import annotations
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding function (local, multilingual)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbeddingFunction:
    """
    Wraps sentence-transformers for ChromaDB.
    Fully local — no HuggingFace network calls (local_files_only=True).
    """
    def __init__(self, model_name: str):
        import os
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model (local): {model_name} …")
        self._model = SentenceTransformer(model_name, local_files_only=True)

    def name(self) -> str:
        return "local-sentence-transformer"

    def _encode(self, input: List[str]) -> List[List[float]]:
        return self._model.encode(input, show_progress_bar=False).tolist()

    # ChromaDB ≥1.5 calls embed_documents on add and embed_query on query
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._encode(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._encode(input)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._encode(input)


# ─────────────────────────────────────────────────────────────────────────────
#  VideoDatabase
# ─────────────────────────────────────────────────────────────────────────────

class VideoDatabase:
    """
    The complete queryable knowledge base for one processed video.

    Usage after building:
        db = VideoDatabase.load("./video_db/my_video")
        results = db.search("什么是注意力机制")
        timeline = db.get_timeline()
    """

    def __init__(self, db_dir: Path, cfg):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        # ChromaDB
        import chromadb
        self._chroma_client = chromadb.PersistentClient(path=str(self.db_dir / "chroma"))
        self._embed_fn = LocalEmbeddingFunction(cfg.EMBEDDING_MODEL)
        self._collection = self._chroma_client.get_or_create_collection(
            name            = cfg.CHROMA_COLLECTION,
            embedding_function = self._embed_fn,
            metadata        = {"hnsw:space": "cosine"},
        )

        # JSON timeline path
        self._timeline_path = self.db_dir / "timeline.json"
        self._meta_path     = self.db_dir / "meta.json"

    # ──────────────────────────────────────────────────────────────────────
    #  Ingestion
    # ──────────────────────────────────────────────────────────────────────

    def ingest(self, segments, video_path: str, video_duration_sec: float):
        """Write all segments into both stores."""
        self._write_chroma(segments)
        self._write_timeline(segments, video_path, video_duration_sec)
        log.info(f"Database ready at {self.db_dir}")

    def _write_chroma(self, segments):
        """Upsert all segments into ChromaDB."""
        existing_ids = set(self._collection.get(include=[])["ids"])

        ids, texts, metadatas = [], [], []
        for seg in segments:
            doc_id = f"seg_{seg.segment_id:05d}"
            if doc_id in existing_ids:
                continue

            ids.append(doc_id)
            texts.append(seg.embedding_text)
            metadatas.append({
                "segment_id":    seg.segment_id,
                "start_ms":      seg.start_ms,
                "end_ms":        seg.end_ms,
                "start_ts":      _fmt_ms(seg.start_ms),
                "end_ts":        _fmt_ms(seg.end_ms),
                "transcript":    seg.transcript[:500],
                "fused_summary": seg.fused_summary[:500],
                "slide_title":   seg.slide_title,
                "slide_type":    seg.slide_type,
                "is_slide_change": str(seg.is_slide_change),
                "frame_path":    seg.frame_path,
            })

        if ids:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
            log.info(f"Inserted {len(ids)} segments into ChromaDB.")
        else:
            log.info("ChromaDB already up to date.")

    def _write_timeline(self, segments, video_path: str, duration_sec: float):
        """Write the full structured timeline JSON."""
        seg_dicts = []
        for s in segments:
            d = asdict(s)
            d["start_ts"] = _fmt_ms(s.start_ms)
            d["end_ts"]   = _fmt_ms(s.end_ms)
            seg_dicts.append(d)
        timeline = {
            "video_path":    video_path,
            "duration_sec":  duration_sec,
            "total_segments":len(segments),
            "segments":      seg_dicts,
            "slide_index":   self._build_slide_index(segments),
        }
        with open(self._timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, ensure_ascii=False, indent=2)
        log.info(f"Timeline saved → {self._timeline_path}")

    def _build_slide_index(self, segments) -> List[Dict]:
        """Index of all slide changes for fast navigation."""
        index = []
        for seg in segments:
            if seg.is_slide_change or seg.segment_id == 0:
                index.append({
                    "timestamp_ms": seg.start_ms,
                    "timestamp":    _fmt_ms(seg.start_ms),
                    "slide_title":  seg.slide_title,
                    "slide_type":   seg.slide_type,
                    "segment_id":   seg.segment_id,
                    "frame_path":   seg.frame_path,
                })
        return index

    # ──────────────────────────────────────────────────────────────────────
    #  Query API
    # ──────────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Semantic search over all fused segment summaries.
        Returns the most relevant segments with metadata.
        """
        results = self._collection.query(
            query_texts = [query],
            n_results   = min(n_results, self._collection.count()),
            include     = ["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "rank":          i + 1,
                "score":         round(1 - results["distances"][0][i], 4),
                "segment_id":    results["metadatas"][0][i]["segment_id"],
                "timestamp":     results["metadatas"][0][i]["start_ts"],
                "transcript":    results["metadatas"][0][i]["transcript"],
                "fused_summary": results["metadatas"][0][i]["fused_summary"],
                "slide_title":   results["metadatas"][0][i]["slide_title"],
                "frame_path":    results["metadatas"][0][i]["frame_path"],
            })
        return hits

    def get_segment_by_time(self, timestamp_ms: int) -> Optional[Dict]:
        """Return the segment that covers a given millisecond timestamp."""
        timeline = self._load_timeline()
        if not timeline:
            return None
        for seg in timeline["segments"]:
            if seg["start_ms"] <= timestamp_ms <= seg["end_ms"]:
                return seg
        return None

    def get_timeline(self) -> Optional[Dict]:
        return self._load_timeline()

    def get_slide_index(self) -> List[Dict]:
        timeline = self._load_timeline()
        return timeline.get("slide_index", []) if timeline else []

    def get_all_segments(self) -> List[Dict]:
        timeline = self._load_timeline()
        return timeline.get("segments", []) if timeline else []

    def get_full_transcript(self) -> str:
        """Concatenated transcript of the entire video."""
        segs = self.get_all_segments()
        return "\n".join(
            f"[{s['start_ts']}] {s['transcript']}" for s in segs
        )

    def count(self) -> int:
        return self._collection.count()

    def _load_timeline(self) -> Optional[Dict]:
        if not self._timeline_path.exists():
            return None
        with open(self._timeline_path, encoding="utf-8") as f:
            return json.load(f)

    # ──────────────────────────────────────────────────────────────────────
    #  Static loader
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, db_dir: str, cfg) -> "VideoDatabase":
        """Load an existing database (no re-processing)."""
        db = cls(Path(db_dir), cfg)
        count = db.count()
        log.info(f"Loaded database from {db_dir} ({count} segments)")
        return db


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s//60:02d}:{s%60:02d}"
