"""core/lang.py — lightweight language detector for STT output.

Runs on a small character sample from the transcribed sentences to decide
whether the primary language is Chinese or English.  No external deps —
uses only Unicode code-point ranges.

Returns
-------
"zh"  Chinese (CJK characters dominate)
"en"  English / other latin-script language

"""

from __future__ import annotations

_SAMPLE_CHARS = 400   # characters to inspect (usually 2-5 sentences)
_CJK_THRESHOLD = 0.25  # if CJK fraction of alpha chars exceeds this → "zh"


def detect_language(sentences, sample_chars: int = _SAMPLE_CHARS) -> str:
    """Return 'zh' or 'en' from a small sample of SentenceSegment objects."""
    if not sentences:
        return "zh"

    sample = ""
    for s in sentences:
        sample += s.text
        if len(sample) >= sample_chars:
            break
    sample = sample[:sample_chars]

    cjk = sum(1 for ch in sample if _is_cjk(ch))
    ascii_ = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    total = cjk + ascii_

    if total == 0:
        return "zh"  # no recognisable alpha chars → default to Chinese

    return "zh" if (cjk / total) >= _CJK_THRESHOLD else "en"


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0xF900 <= cp <= 0xFAFF   # CJK Compatibility Ideographs
    )
