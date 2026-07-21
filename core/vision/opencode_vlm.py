"""core/vision/opencode_vlm.py — VLM via local opencode server.

Spawns a persistent ``opencode serve`` subprocess and reuses one session for
all frame-analysis calls through the synchronous HTTP message endpoint
``POST /session/{id}/message``.  No base_url or API key is needed — opencode's
built-in ``opencode`` provider hosts free vision models (e.g. mimo-v2.5-free).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_HEALTH_RETRIES = 60
_HEALTH_INTERVAL_S = 0.5
# Every model-generating call is idle-monitored — aborted only after a stretch of
# NO activity, never by a total wall-clock cap on work that is still progressing.
# Control-plane calls (session create, providers, stats, abort) keep short
# explicit timeouts since those are not "work in progress".
_IDLE_TIMEOUT_S = 300
# An identical failed tool call repeated MORE than this many times force-stops the
# session (loop guard), independent of the idle timer — the agent is thrashing.
_LOOP_TOLERANCE = 3
_CONTROL_TIMEOUT_S = 30


class AgentTimeout(TimeoutError):
    """Raised when a monitored session is force-stopped (idle or loop guard).

    Carries the agent's partial output so callers can salvage a best-effort
    result instead of discarding the work already done. ``reason`` is ``"idle"``
    or ``"loop"``.
    """

    def __init__(self, message: str, partial_text: str = "", last_tool: str = "", reason: str = "idle"):
        super().__init__(message)
        self.partial_text = partial_text
        self.last_tool = last_tool
        self.reason = reason


def _find_opencode_binary() -> str:
    """Locate the actual OpenCode executable, avoiding an npm shell shim."""
    env_bin = os.environ.get("OPENCODE_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise FileNotFoundError(f"OPENCODE_BIN is set but not executable: {env_bin}")

    if os.name == "nt":
        # ``opencode`` resolves to npm's extensionless shell shim on Windows.
        # Killing that shim leaves its ``opencode.exe serve`` child orphaned,
        # so prefer the executable installed alongside the global npm command.
        shim = shutil.which("opencode.cmd")
        if shim:
            bundled = Path(shim).parent / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
            if bundled.is_file():
                return str(bundled)

    for name in ("opencode.exe", "opencode.cmd", "opencode"):
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
        # The client only talks to the local opencode server. Ignore proxy
        # environment variables so VPN toggles or unsupported proxy schemes
        # (for example socks://...) cannot break localhost requests.
        # No total read timeout — long-but-progressing generations must not be
        # killed by a wall-clock cap; idleness is enforced per call instead.
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10, read=None, write=30, pool=30),
            trust_env=False,
        )
        self._start()

    # ── server lifecycle ──────────────────────────────────────────────

    def _start(self) -> None:
        bin_path = _find_opencode_binary()
        args = [bin_path, "serve"]
        if self._port:
            args += ["--port", str(self._port)]
        log.info(f"Starting opencode server: {' '.join(args)}")
        # Exa-backed web search is opt-in for non-OpenCode providers.  Set it
        # on the server process so the web-crosscheck agent always receives
        # both built-in websearch and webfetch tools.
        env = os.environ.copy()
        env["OPENCODE_ENABLE_EXA"] = "1"
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
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
                r = httpx.get(f"{base}/api/health", timeout=2, trust_env=False)
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
        r = self._client.post(f"{self._base_url}/session", json={}, timeout=_CONTROL_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
        return (body.get("data") or body)["id"]

    def _abort(self, sid: str) -> None:
        try:
            self._client.post(f"{self._base_url}/session/{sid}/abort", json={}, timeout=_CONTROL_TIMEOUT_S)
        except Exception:
            pass

    def text_model_id(self) -> str:
        """``provider/model`` id of the configured text model."""
        return f"{self._text_provider}/{self._text_model}"

    def text_context_limit(self, fallback: int = 128000, overrides: dict | None = None) -> int:
        """Context-window size (in tokens) of the configured text model.

        Resolution order: an explicit ``overrides`` entry for this model wins
        (some providers, e.g. agnes, report ``limit 0`` in the catalogue), then
        the opencode server's ``/config/providers`` catalogue (``limit.context``
        from models.dev), then ``fallback``. Used to size the short/long
        threshold so it tracks whichever model is selected.
        """
        if overrides and (override := overrides.get(self.text_model_id())):
            return int(override)
        try:
            r = self._client.get(f"{self._base_url}/config/providers", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            log.warning("could not read model context limit (%s); using %d", exc, fallback)
            return fallback
        providers = data.get("providers", data) if isinstance(data, dict) else data
        for provider in providers or []:
            if provider.get("id") not in (self._text_provider, None):
                continue
            model = (provider.get("models") or {}).get(self._text_model)
            context = ((model or {}).get("limit") or {}).get("context")
            if context:
                return int(context)
        log.warning(
            "model %s/%s reported no context limit; using %d",
            self._text_provider, self._text_model, fallback,
        )
        return fallback

    # ── VLM call ──────────────────────────────────────────────────────

    def call(
        self,
        image_b64: str,
        prompt: str,
        mime: str = "image/jpeg",
        fresh_session: bool = False,
    ) -> str:
        """Send one image + prompt; return the assistant's text response.

        ``fresh_session=True`` isolates the call in its own session — required
        when calls run concurrently, since parallel posts to the shared
        session would interleave chat history.
        """
        sid = self._create_session() if fresh_session else self._session_id
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
        return self._post_and_wait(sid, body)

    def call_image_file(self, image_path: Path, prompt: str) -> str:
        """Compress a frame image and send it with the prompt."""
        from utils.retry import compress_frame_for_vlm
        import importlib
        m = importlib.import_module("config")
        b64, mime, _w, _h = compress_frame_for_vlm(image_path, m)
        return self.call(b64, prompt, mime)

    # ── text-only LLM call (pure mode, no image) ───────────────────────

    def call_text(
        self,
        prompt: str,
        variant: str | None = None,
        agent: str | None = None,
    ) -> str:
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
        if agent:
            body["agent"] = agent
        return self._post_and_wait(sid, body)

    def call_text_monitored(
        self,
        prompt: str,
        variant: str | None = None,
        agent: str | None = None,
        on_progress=None,
        idle_timeout_s: float = _IDLE_TIMEOUT_S,
        poll_interval_s: float = 2.0,
        loop_tolerance: int = _LOOP_TOLERANCE,
    ) -> str:
        """``call_text`` with an ``on_progress`` callback for long agent runs.

        Behaviour matches every other call — idle-monitored, loop-guarded, no
        total timeout — this variant additionally streams progress stats to the
        caller and lets it tune the idle/loop thresholds.
        """
        sid = self._create_session()
        body: dict = {
            "model": {"providerID": self._text_provider, "modelID": self._text_model},
            "parts": [{"type": "text", "text": prompt}],
        }
        v = variant if variant is not None else self._text_variant
        if v:
            body["variant"] = v
        if agent:
            body["agent"] = agent
        return self._post_and_wait(
            sid, body,
            idle_timeout_s=idle_timeout_s,
            on_progress=on_progress,
            poll_interval_s=poll_interval_s,
            loop_tolerance=loop_tolerance,
        )

    def _post_and_wait(
        self,
        sid: str,
        body: dict,
        idle_timeout_s: float = _IDLE_TIMEOUT_S,
        on_progress=None,
        poll_interval_s: float = 2.0,
        loop_tolerance: int = _LOOP_TOLERANCE,
    ) -> str:
        """POST one message and wait for the reply under idle + loop guards.

        The POST runs in a worker thread with no read timeout, so a long-but-
        progressing generation is never killed by a wall-clock cap. This thread
        polls ``GET /session/{id}/message``; any growth in messages, parts, tool
        calls, or streamed text counts as progress and resets the idle timer.
        The session is force-stopped (``POST /session/{id}/abort``, raising
        :class:`AgentTimeout` carrying the partial output) when either guard
        trips:

          * idle — no activity at all for ``idle_timeout_s`` seconds; or
          * loop — an identical failed tool call repeated more than
            ``loop_tolerance`` times. Thrashing keeps the idle timer alive but
            makes no real progress, so it needs its own guard.
        """
        outcome: dict = {}

        def _post():
            try:
                outcome["response"] = self._client.post(
                    f"{self._base_url}/session/{sid}/message",
                    json=body,
                    timeout=httpx.Timeout(connect=10, read=None, write=30, pool=30),
                )
            except Exception as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=_post, daemon=True)
        worker.start()

        started = last_activity = time.time()
        signature = None
        stats = {"messages": 0, "parts": 0, "tools": 0, "text_chars": 0}
        while worker.is_alive():
            worker.join(poll_interval_s)
            now = time.time()
            current = self._session_stats(sid)
            if current is not None:
                if (sig := tuple(sorted(current.items()))) != signature:
                    signature = sig
                    last_activity = now
                    stats = current
                if loop_tolerance and current.get("max_repeated_failure", 0) > loop_tolerance:
                    self._abort(sid)
                    worker.join(5)
                    raise AgentTimeout(
                        f"repeated a failing tool call more than {loop_tolerance} times",
                        partial_text=current.get("text_tail", ""),
                        last_tool=current.get("last_tool", ""),
                        reason="loop",
                    )
            stats["elapsed_s"] = now - started
            stats["idle_s"] = now - last_activity
            if on_progress:
                on_progress(dict(stats))
            if now - last_activity > idle_timeout_s:
                self._abort(sid)
                worker.join(5)
                raise AgentTimeout(
                    f"made no progress for {idle_timeout_s:.0f}s",
                    partial_text=stats.get("text_tail", ""),
                    last_tool=stats.get("last_tool", ""),
                    reason="idle",
                )

        if "error" in outcome:
            raise outcome["error"]
        r = outcome["response"]
        r.raise_for_status()
        data = r.json()
        texts = [p["text"] for p in data.get("parts", []) if p.get("type") == "text"]
        return "\n".join(texts).strip() if texts else ""

    def _session_stats(self, sid: str) -> dict | None:
        """Snapshot activity counters for a session; None when the poll fails."""
        try:
            r = self._client.get(
                f"{self._base_url}/session/{sid}/message", timeout=10
            )
            r.raise_for_status()
            messages = r.json()
        except Exception:
            return None
        if not isinstance(messages, list):
            return None
        parts = tools = text_chars = 0
        last_tool = ""
        text_accum: list[str] = []
        failed_calls: dict[str, int] = {}  # identical failed tool call -> count
        for message in messages:
            role = (message.get("info") or {}).get("role") or message.get("role")
            if role == "user":
                continue  # the prompt echo must not count as agent progress
            for part in (message.get("parts") or []):
                parts += 1
                kind = part.get("type")
                if kind == "tool":
                    tools += 1
                    state = part.get("state") or {}
                    status = state.get("status", "?")
                    inputs = state.get("input") or {}
                    target = inputs.get("url") or inputs.get("query") or ""
                    tool_name = part.get("tool", "?")
                    last_tool = f"{tool_name}[{status}] {target}"[:70]
                    if status == "error":
                        key = tool_name + "|" + json.dumps(inputs, sort_keys=True, default=str)
                        failed_calls[key] = failed_calls.get(key, 0) + 1
                elif kind == "reasoning":
                    text_chars += len(part.get("text", "") or "")
                elif kind == "text":
                    text = part.get("text", "") or ""
                    text_chars += len(text)
                    text_accum.append(text)
        return {
            "messages": len(messages),
            "parts": parts,
            "tools": tools,
            "text_chars": text_chars,
            "last_tool": last_tool,
            "text_tail": "".join(text_accum)[-4000:],
            "max_repeated_failure": max(failed_calls.values(), default=0),
        }

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
