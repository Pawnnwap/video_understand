"""core/vision/opencode_vlm.py — VLM via local opencode server.

Spawns a persistent ``opencode serve`` subprocess and reuses one session for
all frame-analysis calls through the synchronous HTTP message endpoint
``POST /session/{id}/message``.  No base_url or API key is needed — opencode's
built-in ``opencode`` provider hosts free vision models (e.g. mimo-v2.5-free).
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_HEALTH_RETRIES = 60
_HEALTH_INTERVAL_S = 0.5
_REQUEST_TIMEOUT_S = 180


def _find_opencode_binary() -> str:
    """Locate the opencode executable (npm global .cmd on Windows)."""
    for name in ("opencode", "opencode.cmd", "opencode.exe"):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError(
        "opencode binary not found in PATH. Install via `npm i -g opencode-ai` "
        "or set OPENCODE_BIN to the full path."
    )


class OpencodeVLM:
    """Persistent opencode server + session for VLM frame analysis.

    Usage::

        with OpencodeVLM(model="opencode/mimo-v2.5-free") as vlm:
            text = vlm.call(image_b64, "Describe this image.")
    """

    def __init__(
        self,
        model: str = "opencode/mimo-v2.5-free",
        port: int = 0,
        variant: str | None = None,
        text_model: str | None = None,
        text_variant: str | None = None,
    ):
        if "/" in model:
            self._provider, self._model = model.split("/", 1)
        else:
            self._provider, self._model = "opencode", model
        self._port = port
        self._variant = variant
        # Text-LLM model — defaults to the same as vision model.
        if text_model:
            if "/" in text_model:
                self._text_provider, self._text_model = text_model.split("/", 1)
            else:
                self._text_provider, self._text_model = "opencode", text_model
        else:
            self._text_provider, self._text_model = self._provider, self._model
        self._text_variant = text_variant
        self._base_url: str | None = None
        self._proc: subprocess.Popen | None = None
        self._session_id: str | None = None
        self._client = httpx.Client(timeout=_REQUEST_TIMEOUT_S)
        self._start()

    # ── server lifecycle ──────────────────────────────────────────────

    def _start(self) -> None:
        bin_path = _find_opencode_binary()
        args = [bin_path, "serve"]
        if self._port:
            args += ["--port", str(self._port)]
        log.info(f"Starting opencode server: {' '.join(args)}")
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        self._base_url = self._wait_health()
        log.info(f"opencode server ready at {self._base_url}")
        self._session_id = self._create_session()
        log.info(f"opencode session: {self._session_id}")

    def _wait_health(self) -> str:
        """Poll until server responds, parsing the port from stdout if random."""
        port = self._port or self._read_port_from_stdout()
        base = f"http://127.0.0.1:{port}"
        for _ in range(_HEALTH_RETRIES):
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("opencode server exited unexpectedly")
            try:
                r = httpx.get(f"{base}/api/health", timeout=2)
                if r.status_code == 200:
                    return base
            except Exception:
                pass
            time.sleep(_HEALTH_INTERVAL_S)
        raise RuntimeError("opencode server did not become healthy in time")

    def _read_port_from_stdout(self) -> int:
        """Parse the chosen port from early serve stdout."""
        deadline = time.time() + 20
        while time.time() < deadline:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError("opencode serve exited before listening")
                time.sleep(0.1)
                continue
            log.debug(f"opencode: {line.strip()}")
            if "listening on" in line:
                tail = line.rsplit(":", 1)[-1].strip()
                return int("".join(c for c in tail if c.isdigit()))
        raise RuntimeError("Could not determine opencode server port")

    def _create_session(self) -> str:
        r = self._client.post(f"{self._base_url}/session", json={})
        r.raise_for_status()
        body = r.json()
        return (body.get("data") or body)["id"]

    # ── VLM call ──────────────────────────────────────────────────────

    def call(self, image_b64: str, prompt: str, mime: str = "image/jpeg") -> str:
        """Send one image + prompt; return the assistant's text response."""
        body: dict = {
            "model": {"providerID": self._provider, "modelID": self._model},
            "parts": [
                {
                    "type": "file",
                    "mime": mime,
                    "url": f"data:{mime};base64,{image_b64}",
                },
                {"type": "text", "text": prompt},
            ],
        }
        if self._variant:
            body["variant"] = self._variant
        r = self._client.post(
            f"{self._base_url}/session/{self._session_id}/message",
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        texts = [
            p["text"]
            for p in data.get("parts", [])
            if p.get("type") == "text"
        ]
        return "\n".join(texts).strip() if texts else ""

    def call_image_file(self, image_path: Path, prompt: str) -> str:
        """Compress a frame image and send it with the prompt."""
        from utils.retry import compress_frame_for_vlm
        import importlib
        m = importlib.import_module("config")
        b64, mime, _w, _h = compress_frame_for_vlm(image_path, m)
        return self.call(b64, prompt, mime)

    # ── text-only LLM call (pure mode, no image) ───────────────────────

    def call_text(self, prompt: str, variant: str | None = None) -> str:
        """Send a text-only message — no image attachment — for "pure mode"
        summarisation.  A fresh session is created per call so each segment
        fusion runs with a clean context (no prior prompt bleeding in).
        """
        sid = self._create_session()
        body: dict = {
            "model": {"providerID": self._text_provider, "modelID": self._text_model},
            "parts": [{"type": "text", "text": prompt}],
        }
        v = variant if variant is not None else self._text_variant
        if v:
            body["variant"] = v
        r = self._client.post(
            f"{self._base_url}/session/{sid}/message",
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        texts = [
            p["text"]
            for p in data.get("parts", [])
            if p.get("type") == "text"
        ]
        return "\n".join(texts).strip() if texts else ""

    # ── cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

