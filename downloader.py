"""downloader.py — video acquisition module
Accepts a local file path, a short video code, or a full URL.
Returns a local Path ready for the pipeline.

Supported short codes (no URL prefix needed):
  - YouTube  : 11-char video ID      e.g.  dQw4w9WgXcQ
  - Bilibili : BV-code               e.g.  BV1GE411T7Wv

Full URLs are also accepted:
  - YouTube  : https://www.youtube.com/watch?v=...  |  https://youtu.be/...
  - Bilibili : https://www.bilibili.com/video/BV...  |  https://b23.tv/...

Backend: yt-dlp
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# ── Short-code patterns ───────────────────────────────────────────────────────
# YouTube video IDs are exactly 11 URL-safe characters
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Bilibili BV codes
_BV_RE_SHORT = re.compile(r"^BV[A-Za-z0-9]{10}$", re.IGNORECASE)

# ── Full-URL patterns ─────────────────────────────────────────────────────────
_YT_URL_RE = re.compile(r"(youtube\.com/watch|youtu\.be/|youtube\.com/shorts)", re.IGNORECASE)
_BV_URL_RE = re.compile(r"(bilibili\.com/video|b23\.tv/)", re.IGNORECASE)


def _expand_short_code(source: str) -> str | None:
    """If source looks like a YouTube ID or Bilibili BV code, return the
    canonical URL.  Returns None if source is not a recognised short code.

    Heuristic: local files always have a file extension (e.g. '.mp4').
    Anything without an extension is treated as a potential short code.
    """
    s = source.strip()
    # Local files always have an extension — bail out early
    if Path(s).suffix:
        return None
    if _BV_RE_SHORT.match(s):
        return f"https://www.bilibili.com/video/{s}"
    if _YT_ID_RE.match(s):
        return f"https://www.youtube.com/watch?v={s}"
    return None


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def is_youtube(url: str) -> bool:
    return bool(_YT_URL_RE.search(url))


def is_bilibili(url: str) -> bool:
    return bool(_BV_URL_RE.search(url))


# ── Main entry point ──────────────────────────────────────────────────────────

def resolve_source(source: str, download_dir: str = "./downloads", max_duration_sec: int = 0) -> Path:
    """Given a local path, short video code, or full URL, return a local Path.
    Downloads if necessary. Raises ValueError for unsupported inputs.
    """
    # Expand short codes before anything else
    expanded = _expand_short_code(source)
    if expanded:
        log.info(f"Short code '{source}' → {expanded}")
        source = expanded

    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {source}")
        log.info(f"Local file: {path}")
        return path

    if not (is_youtube(source) or is_bilibili(source)):
        raise ValueError(
            f"Unsupported URL. Only YouTube and Bilibili are supported.\nGot: {source}",
        )

    return _download(source, Path(download_dir), max_duration_sec=max_duration_sec)


# ── Downloader ────────────────────────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """Fetch metadata (title, duration, etc.) without downloading."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _download(url: str, out_dir: Path, max_duration_sec: int = 0) -> Path:
    """Download video with yt-dlp. Picks best quality ≤720p as mp4.
    max_duration_sec: if > 0, refuse videos longer than this (0 = no limit).
    Returns path to downloaded file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp not installed. Run: pip install yt-dlp")

    # Duration pre-check (avoids downloading a 3-hour lecture by mistake)
    if max_duration_sec > 0:
        try:
            info = get_video_info(url)
            dur = info.get("duration", 0) or 0
            if dur > max_duration_sec:
                raise ValueError(
                    f"Video is {dur // 60}m{dur % 60}s — exceeds limit of "
                    f"{max_duration_sec // 60}m{max_duration_sec % 60}s. "
                    "Pass max_duration_sec=0 to disable this check.",
                )
            log.info(f"Duration OK: {dur // 60}m{dur % 60}s")
        except ValueError:
            raise
        except Exception as e:
            log.warning(f"Could not pre-check duration: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Output template — sanitise title, avoid collisions
    outtmpl = str(out_dir / "%(title).60s_%(id)s.%(ext)s")

    ydl_opts = {
        # Best mp4 ≤720p; fallback to any best single-file format
        "format": (
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4][height<=720]+bestaudio"
            "/best[ext=mp4][height<=720]"
            "/best[height<=720]"
            "/best"
        ),
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        },
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }

    log.info(f"Downloading: {url}")

    attempts = [ydl_opts]
    last_err = None

    for attempt_opts in attempts:
        downloaded_path: Path | None = None

        class _PathCapture(yt_dlp.YoutubeDL):
            def process_info(self, info_dict):
                nonlocal downloaded_path
                result = super().process_info(info_dict)
                fp = info_dict.get("requested_downloads", [{}])[0].get("filepath")
                if fp:
                    downloaded_path = Path(fp)
                return result

        try:
            with _PathCapture(attempt_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if downloaded_path is None:
                    fp = info.get("requested_downloads", [{}])[0].get("filepath") if info else None
                    if fp:
                        downloaded_path = Path(fp)
                    else:
                        files = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                        if files:
                            downloaded_path = files[-1]

            if downloaded_path and downloaded_path.exists():
                log.info(f"Downloaded → {downloaded_path}")
                return downloaded_path

        except Exception as e:
            msg = str(e)
            if "DPAPI" in msg or "decrypt" in msg.lower() or "cookie" in msg.lower():
                log.warning(f"Cookie extraction failed ({e}), retrying without cookies …")
                last_err = e
                continue
            raise  # non-cookie error — re-raise immediately

    raise RuntimeError(f"Download failed after all attempts. Last error: {last_err}")


def _progress_hook(d: dict):
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?%").strip()
        spd = d.get("_speed_str", "?/s").strip()
        eta = d.get("_eta_str", "?s").strip()
        log.info(f"  Downloading {pct}  speed={spd}  eta={eta}")
    elif d["status"] == "finished":
        log.info(f"  Download finished: {d.get('filename', '')}")
    elif d["status"] == "error":
        log.error(f"  Download error: {d.get('error', 'unknown')}")
