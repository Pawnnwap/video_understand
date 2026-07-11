"""utils/logging_setup.py — centralised logging with coloured console
output and tqdm-aware routing so active progress bars are never corrupted
by interleaved log lines.

Usage::
    from utils.logging_setup import setup_logging
    setup_logging(level=logging.INFO, log_file="run.log")

Any handler installed here routes records through ``tqdm.write`` when a
progress bar is active, so bars stay intact while log messages still show.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"

_DATEFMT = "%H:%M:%S"
_CONSOLE_FMT = "%(asctime)s %(levelname)s %(name)s \u2014 %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-8s %(name)s \u2014 %(message)s"


class _ColorFormatter(logging.Formatter):
    """Paint the level name with ANSI colours; fall back to plain on non-TTY."""

    def __init__(self, fmt: str, datefmt: str, *, use_color: bool = True):
        super().__init__(fmt, datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        try:
            if self._use_color:
                color = _LEVEL_COLORS.get(original, "")
                record.levelname = f"{color}{original:<8}{_RESET}"
            else:
                record.levelname = f"{original:<8}"
            return super().format(record)
        finally:
            record.levelname = original


class TqdmAwareHandler(logging.Handler):
    """Stream handler that renders through tqdm.write when a bar is active.

    Falls back to a plain stream.write if tqdm is unavailable or not
    installed yet (e.g. during very early import).
    """

    def __init__(self, stream=None, formatter: logging.Formatter | None = None):
        super().__init__()
        if formatter:
            self.setFormatter(formatter)
        self._stream = stream or sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg, file=self._stream)
            except Exception:
                self._stream.write(msg + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)


# Loggers from third-party libs that are chatty at INFO and drown out the
# pipeline signal.  They are forced to WARNING once setup_logging runs.
_NOISY_LIBS = (
    "httpx", "httpcore", "urllib3", "openai", "paddleocr", "funasr",
    "matplotlib", "PIL", "Pillow", "chardet", "asyncio", "filelock",
    "sentence_transformers", "torch", "torch.distributed", "tensorflow",
)

_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    *,
    use_color: bool | None = None,
) -> logging.RootLogger:
    """Configure root logging: one tqdm-aware console handler + optional file.

    Idempotent: calling again replaces existing handlers (safe for re-entry
    from CLI wrappers that reconfigure level).
    """
    global _CONFIGURED
    root = logging.getLogger()

    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    if use_color is None:
        try:
            use_color = sys.stderr.isatty()
        except Exception:
            use_color = False

    console = TqdmAwareHandler(sys.stderr, _ColorFormatter(_CONSOLE_FMT, _DATEFMT, use_color=use_color))
    root.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATEFMT))
        root.addHandler(fh)

    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def is_configured() -> bool:
    return _CONFIGURED

