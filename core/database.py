from __future__ import annotations

import json
import logging
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger(__name__)


class LocalEmbeddingFunction:
    """Embeds text with a local SentenceTransformer, loaded lazily.

    The model weights are only loaded on the first actual embed call, which
    ChromaDB triggers on ``add()``/``query()``. Query paths that read the
    timeline JSON (summary, crosscheck) never embed, so they skip the load
    entirely — no wasted weight-loading when only building a report.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        import os
        model_name = self._model_name
        local_dir = self._resolve_local(model_name)
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {local_dir or model_name} ...")
        if local_dir:
            saved = {}
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_ENDPOINT"):
                if k in os.environ:
                    saved[k] = os.environ.pop(k)
            saved.setdefault("HF_HUB_OFFLINE", None)
            saved.setdefault("TRANSFORMERS_OFFLINE", None)
            saved.setdefault("HF_DATASETS_OFFLINE", None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                self._model = SentenceTransformer(local_dir)
            finally:
                for k, v in saved.items():
                    if v is None: os.environ.pop(k, None)
                    else: os.environ[k] = v
        else:
            if self._looks_like_local_path(model_name):
                log.warning(
                    "Local embedding model not found at %s; using deterministic hash embeddings.",
                    model_name,
                )
                self._model = _HashEmbeddingModel()
            else:
                saved = {}
                for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                    if k in os.environ:
                        saved[k] = os.environ.pop(k)
                try:
                    self._model = SentenceTransformer(model_name)
                except Exception as e:
                    log.warning(
                        "SentenceTransformer load failed (%s); using deterministic hash embeddings.",
                        e,
                    )
                    self._model = _HashEmbeddingModel()
                finally:
                    os.environ.update(saved)
        log.info(f"Embedding model ready: {local_dir or model_name}")
        return self._model

    @staticmethod
    def _resolve_local(model_name: str):
        """Return a usable local directory for the model, else None."""
        from pathlib import Path
        cand = Path(model_name)
        if not cand.is_absolute():
            cand = (Path(__file__).resolve().parent.parent / cand).resolve()
        if (cand / "config.json").exists() and (cand / "modules.json").exists():
            return str(cand)
        return None

    @staticmethod
    def _looks_like_local_path(model_name: str) -> bool:
        from pathlib import Path
        p = Path(model_name)
        return p.is_absolute() or any(sep in model_name for sep in ("/", "\\"))

    def name(self) -> str:
        return "local-sentence-transformer"

    def _encode(self, input: list[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(input, show_progress_bar=False)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)


class _HashEmbeddingModel:
    """Small offline fallback with a SentenceTransformer-like encode method."""

    dim = 384

    def encode(self, input: list[str], show_progress_bar: bool = False) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        chars = [c for c in text.lower() if not c.isspace()]
        tokens = text.lower().split()
        features = tokens or chars
        if chars:
            features.extend("".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1)))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class VideoDatabase:
    def __init__(self, db_dir: Path, cfg):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        import chromadb
        self._chroma_client = chromadb.PersistentClient(path=str(self.db_dir / "chroma"))
        self._embed_fn = LocalEmbeddingFunction(cfg.EMBEDDING_MODEL)
        self._collection = self._chroma_client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._timeline_path = self.db_dir / "timeline.json"
        self._meta_path = self.db_dir / "meta.json"

    def ingest(self, segments, video_path: str, video_duration_sec: float):
        self._write_chroma(segments)
        self._write_timeline(segments, video_path, video_duration_sec)
        log.info(f"Database ready at {self.db_dir}")

    def _write_chroma(self, segments):
        existing_ids = set(self._collection.get(include=[])["ids"])
        ids, texts, metadatas = [], [], []
        for seg in segments:
            doc_id = f"seg_{seg.segment_id:05d}"
            if doc_id in existing_ids:
                continue
            ids.append(doc_id)
            texts.append(seg.embedding_text)
            metadatas.append({
                "segment_id": seg.segment_id,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "start_ts": _fmt_ms(seg.start_ms),
                "end_ts": _fmt_ms(seg.end_ms),
                "transcript": seg.transcript[:500],
                "fused_summary": seg.fused_summary[:500],
                "slide_title": seg.slide_title,
                "slide_type": seg.slide_type,
                "is_slide_change": str(seg.is_slide_change),
                "frame_path": seg.frame_path,
            })
        if ids:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
            log.info(f"Inserted {len(ids)} segments into ChromaDB.")
        else:
            log.info("ChromaDB already up to date.")

    def _write_timeline(self, segments, video_path: str, duration_sec: float):
        seg_dicts = []
        for s in segments:
            d = asdict(s)
            d["start_ts"] = _fmt_ms(s.start_ms)
            d["end_ts"] = _fmt_ms(s.end_ms)
            seg_dicts.append(d)
        timeline = {
            "video_path": video_path,
            "duration_sec": duration_sec,
            "total_segments": len(segments),
            "segments": seg_dicts,
            "slide_index": self._build_slide_index(segments),
        }
        with open(self._timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, ensure_ascii=False, indent=2)
        log.info(f"Timeline saved -> {self._timeline_path}")

    def _build_slide_index(self, segments) -> list[dict]:
        index = []
        for seg in segments:
            if seg.is_slide_change or seg.segment_id == 0:
                index.append({
                    "timestamp_ms": seg.start_ms,
                    "timestamp": _fmt_ms(seg.start_ms),
                    "slide_title": seg.slide_title,
                    "slide_type": seg.slide_type,
                    "segment_id": seg.segment_id,
                    "frame_path": seg.frame_path,
                })
        return index

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "rank": i + 1,
                "score": round(1 - results["distances"][0][i], 4),
                "segment_id": results["metadatas"][0][i]["segment_id"],
                "timestamp": results["metadatas"][0][i]["start_ts"],
                "transcript": results["metadatas"][0][i]["transcript"],
                "fused_summary": results["metadatas"][0][i]["fused_summary"],
                "slide_title": results["metadatas"][0][i]["slide_title"],
                "frame_path": results["metadatas"][0][i]["frame_path"],
            })
        return hits

    def get_segment_by_time(self, timestamp_ms: int) -> dict | None:
        timeline = self._load_timeline()
        if not timeline:
            return None
        for seg in timeline["segments"]:
            if seg["start_ms"] <= timestamp_ms <= seg["end_ms"]:
                return seg
        return None

    def get_timeline(self) -> dict | None:
        return self._load_timeline()

    def get_slide_index(self) -> list[dict]:
        timeline = self._load_timeline()
        return timeline.get("slide_index", []) if timeline else []

    def get_all_segments(self) -> list[dict]:
        timeline = self._load_timeline()
        return timeline.get("segments", []) if timeline else []

    def get_full_transcript(self) -> str:
        segs = self.get_all_segments()
        return "\n".join(
            f"[{s['start_ts']}] {s['transcript']}" for s in segs
        )

    def count(self) -> int:
        return self._collection.count()

    def _load_timeline(self) -> dict | None:
        if not self._timeline_path.exists():
            return None
        with open(self._timeline_path, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load(cls, db_dir: str, cfg) -> VideoDatabase:
        db = cls(Path(db_dir), cfg)
        count = db.count()
        log.info(f"Loaded database from {db_dir} ({count} segments)")
        return db


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"
