"""query/crosscheck.py — web fact-checking for video claims.

Pipeline
--------
1. LLM extracts the top-N (claim, evidence) pairs from the video timeline.
2. DDGS text search per claim, enforcing ≥ 3 unique domains per query.
3. Keyword-relevance scoring compresses each search snippet to one sentence
   (no extra LLM call — fast and deterministic).
4. Single LLM call compares all pairs against web findings → structured report.

Language
--------
All LLM prompts are bilingual (zh / en).  Language is auto-detected from the
video transcript stored in the database, using the same CJK-ratio heuristic
as core/lang.py.  Claims sent to the web search are always phrased in English
(requested from the LLM) to maximise source diversity.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# ── Bilingual prompt templates ────────────────────────────────────────────────

_EXTRACT_PROMPT_ZH = """\
以下是视频内容时间线：

{timeline}

请从中识别 {n} 个最重要的事实性声明。
每个声明包含：
  - claim:    具体的事实断言（用中文表达，以便在中文网络中搜索）
  - evidence: 视频中为该声明提供的依据或论述

仅输出 JSON 数组，不含任何其他文字：
[{{"claim": "...", "evidence": "..."}}, ...]"""

_EXTRACT_PROMPT_EN = """\
Below is the video content timeline:

{timeline}

Identify the {n} most important factual claims made in this video.
For each claim provide:
  - claim:    the specific factual assertion (in English, for effective web search)
  - evidence: the supporting evidence or reasoning given in the video

Output ONLY a JSON array, no other text:
[{{"claim": "...", "evidence": "..."}}, ...]"""

# Retry prompts used when the first pass fails to parse — no timeline resent,
# the LLM is asked to reformat the claims it already produced.
_RETRY_PROMPT_ZH = """\
你刚才的回答无法被解析为 JSON。请仅输出一个合法的 JSON 数组，格式如下，\
不要包含任何说明文字、代码块标记或注释：

[
  {{"claim": "中文事实断言", "evidence": "视频依据"}},
  ...（共 {n} 条）
]"""

_RETRY_PROMPT_EN = """\
Your previous response could not be parsed as JSON. \
Output ONLY a valid JSON array — no prose, no code fences, no comments:

[
  {{"claim": "factual assertion in English", "evidence": "evidence from video"}},
  ...({n} items total)
]"""

# ─────────────────────────────────────────────────────────────────────────────

_REPORT_PROMPT_ZH = """\
{context}

针对上方每个声明，写一份简洁的事实核查条目，格式如下：

  判定：SUPPORTED（得到支持）| PARTIALLY SUPPORTED（部分支持）\
 | UNVERIFIED（无法核实）| CONTRADICTED（与事实矛盾）
  置信度：HIGH | MEDIUM | LOW
  分析：1-2句话，引用具体来源域名说明判断依据。

最后附上一段"综合可信度评估"，对视频整体可信度给出结论。
保持客观中立，引用域名而非完整 URL。"""

_REPORT_PROMPT_EN = """\
{context}

For each claim above, write a concise fact-check entry:

  Verdict   : SUPPORTED | PARTIALLY SUPPORTED | UNVERIFIED | CONTRADICTED
  Confidence: HIGH | MEDIUM | LOW
  Analysis  : 1-2 sentences referencing specific source domains.

End with a one-paragraph **Overall Reliability** assessment of the video.
Be precise and neutral. Cite domain names, not full URLs."""


# ── Language detection ────────────────────────────────────────────────────────

def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


def _detect_lang(engine) -> str:
    """Infer 'zh' or 'en' from the video transcript stored in the database."""
    segs = engine.db.get_all_segments()
    sample = "".join(
        (s.get("transcript", "") or "") + (s.get("fused_summary", "") or "")
        for s in segs[:20]
    )[:600]
    if not sample:
        return "zh"
    cjk   = sum(1 for ch in sample if _is_cjk(ch))
    ascii_ = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    total  = cjk + ascii_
    if total == 0:
        return "zh"
    return "zh" if (cjk / total) >= 0.25 else "en"


# ── Step 1: claim extraction ──────────────────────────────────────────────────

def _sample_segments(segments: list[dict], max_tokens: int = 5000) -> list[dict]:
    if not segments:
        return []
    avg = 80
    cap = max_tokens // avg
    if len(segments) <= cap:
        return segments
    step = len(segments) / cap
    return [segments[int(i * step)] for i in range(cap)]


def _as_list(obj) -> list | None:
    """Return obj if it is a list, or the first list-valued entry if it is a dict."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
    return None


def _parse_claim_json(raw: str) -> list[dict] | None:
    """Multi-strategy JSON extractor — robust against all common LLM quirks.

    Strategies tried in order:
      1. Direct json.loads after stripping markdown fences.
      2. Find outermost [...] or {...}, fix trailing commas, json.loads.
      3. ast.literal_eval on the same bracket-bounded chunk (handles single quotes).
      4. Scrape every {...} object that contains a "claim" key.
      5. json_repair library (optional, used if installed).
    """
    # Normalise: strip markdown code fences
    text = re.sub(r"```(?:json|JSON|python)?\s*|\s*```", "", raw).strip()

    # 1 — plain parse
    try:
        result = _as_list(json.loads(text))
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 2 — bracket-bounded chunk + trailing-comma fix
    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, text)
        if not m:
            continue
        chunk = re.sub(r",\s*(?=[}\]])", "", m.group())   # trailing commas
        try:
            result = _as_list(json.loads(chunk))
            if result is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 3 — ast.literal_eval (single-quoted dicts / Python-style output)
    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            result = _as_list(ast.literal_eval(m.group()))
            if result is not None:
                return result
        except (ValueError, SyntaxError):
            pass

    # 4 — scrape individual flat {...} objects that contain "claim"
    scraped = []
    for obj_str in re.findall(r"\{[^{}]+\}", text, re.DOTALL):
        obj_str = re.sub(r",\s*(?=[}\]])", "", obj_str)
        try:
            obj = json.loads(obj_str)
            if "claim" in obj:
                scraped.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    if scraped:
        return scraped

    # 5 — json_repair (optional dependency)
    try:
        import json_repair  # type: ignore
        result = _as_list(json_repair.loads(text))
        if result is not None:
            return result
    except (ImportError, Exception):
        pass

    return None


def _extract_claim_pairs(engine, n: int, lang: str) -> list[dict]:
    """LLM pass 1 (+1 retry): return a list of {claim, evidence} dicts."""
    segs     = engine.db.get_all_segments()
    sampled  = _sample_segments(segs, max_tokens=5000)
    timeline = "\n\n".join(
        f"[{s['start_ts']}] {s['fused_summary']}" for s in sampled
    )

    template = _EXTRACT_PROMPT_ZH if lang == "zh" else _EXTRACT_PROMPT_EN
    prompt   = template.format(timeline=timeline, n=n)

    raw    = engine._llm(prompt, max_tokens=900)
    result = _parse_claim_json(raw)
    if result:
        return [p for p in result if p.get("claim")][:n]

    # Re-prompt: ask the model to reformat without resending the whole timeline
    log.warning("crosscheck: first extraction pass failed to parse — retrying with strict JSON prompt")
    retry_tmpl = _RETRY_PROMPT_ZH if lang == "zh" else _RETRY_PROMPT_EN
    raw2       = engine._llm(retry_tmpl.format(n=n), max_tokens=700)
    result2    = _parse_claim_json(raw2)
    if result2:
        return [p for p in result2 if p.get("claim")][:n]

    log.error("crosscheck: both extraction passes failed to produce valid JSON")
    return []


# ── Step 2: query expansion + multi-search + LLM title reranking ─────────────

_EXPAND_PROMPT_ZH = """\
声明：{claim}

请生成 3 条角度各异的搜索查询，用于对该声明进行网络事实核查。要求：
1. 核心关键词（简短精确，适合直接搜索）
2. 换一种表述或使用该事件的别称/学术名词
3. 聚焦关键数据、机构名称或信息来源

每行一条查询，不含编号或多余解释。"""

_EXPAND_PROMPT_EN = """\
Claim: {claim}

Generate 3 diverse search queries to fact-check this claim from different angles:
1. Core keywords (short and precise)
2. Alternative phrasing or related terminology
3. Key data point, institution, or source keyword

One query per line, no numbering or extra text."""

_RERANK_PROMPT_ZH = """\
声明：{claim}

候选搜索结果：

{candidates}

从中选出最相关且来源最可信的 {k} 条，仅输出编号，以英文逗号分隔（例如：2,5,7）。\
不要输出其他任何内容。"""

_RERANK_PROMPT_EN = """\
Claim: {claim}

Candidate search results:

{candidates}

Select the {k} most relevant and trustworthy results. \
Output only their numbers separated by commas (e.g. 2,5,7), nothing else."""


def _ddgs_search(query: str, max_results: int = 6) -> list[dict]:
    """Raw DDGS text search; returns up to max_results raw result dicts."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("  [crosscheck] ddgs not installed — run: pip install ddgs")
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        print(f"  [search error: {exc}]")
        return []


def _expand_queries(engine, claim: str, lang: str) -> list[str]:
    """LLM generates 2–3 diverse search queries from a single claim."""
    template = _EXPAND_PROMPT_ZH if lang == "zh" else _EXPAND_PROMPT_EN
    raw = engine._llm(template.format(claim=claim), max_tokens=120)
    queries = [
        # strip leading list markers (1. / - / •) that some models add
        re.sub(r"^[\s\d\.\-\•\*]+", "", line).strip()
        for line in raw.strip().splitlines()
        if line.strip() and len(line.strip()) > 3
    ]
    # Always include the claim itself as a reliable fallback
    if claim not in queries:
        queries.insert(0, claim)
    return queries[:4]


def _pool_candidates(queries: list[str], per_query: int = 6) -> list[dict]:
    """Search every query variant; deduplicate by domain; return full pool."""
    seen:    set[str]  = set()
    pool: list[dict] = []
    for q in queries:
        for r in _ddgs_search(q, max_results=per_query):
            domain = urlparse(r.get("href", "")).netloc.removeprefix("www.")
            if not domain or domain in seen:
                continue
            seen.add(domain)
            pool.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "domain":  domain,
                "snippet": r.get("body", ""),
            })
    return pool


def _rerank_by_title(engine, claim: str, pool: list[dict],
                     lang: str, top_k: int = 3) -> list[dict]:
    """LLM picks the top_k most relevant + trustworthy results from the pool.

    Each candidate is shown as:  N. [domain] title — first 120 chars of snippet
    The LLM outputs only a comma-separated list of numbers (e.g. '1,4,7').
    """
    if len(pool) <= top_k:
        return pool

    candidates_text = "\n".join(
        f"{i + 1}. [{r['domain']}] {r['title']} — {r['snippet'][:120]}"
        for i, r in enumerate(pool)
    )
    template = _RERANK_PROMPT_ZH if lang == "zh" else _RERANK_PROMPT_EN
    raw = engine._llm(
        template.format(claim=claim, candidates=candidates_text, k=top_k),
        max_tokens=24,
    )

    # Parse indices robustly: extract all digit runs
    indices = []
    for tok in re.findall(r"\d+", raw):
        idx = int(tok) - 1
        if 0 <= idx < len(pool) and idx not in indices:
            indices.append(idx)
        if len(indices) >= top_k:
            break

    return [pool[i] for i in indices] if indices else pool[:top_k]


def _search_and_rerank(engine, claim: str, lang: str, top_k: int = 3) -> list[dict]:
    """Full search pipeline: expand → pool → rerank."""
    queries = _expand_queries(engine, claim, lang)
    print(f"        Queries: {queries}")

    pool = _pool_candidates(queries, per_query=6)
    if not pool:
        return []
    print(f"        Pool: {len(pool)} candidate(s) from {len(pool)} domain(s)")

    selected = _rerank_by_title(engine, claim, pool, lang, top_k=top_k)
    print(f"        Selected: {', '.join(s['domain'] for s in selected)}")
    return selected


# ── Step 3: snippet compression ───────────────────────────────────────────────

def _compress_snippet(snippet: str, claim: str) -> str:
    """Return the single most claim-relevant sentence from a search snippet.

    Scoring is keyword-overlap based — deterministic, zero LLM cost.
    Falls back to the raw snippet (truncated) when it is already short.
    """
    if not snippet:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", snippet.strip())
    if len(sentences) <= 2:
        return snippet[:300]

    claim_kw  = {w.lower() for w in claim.split() if len(w) > 3}
    best, best_score = sentences[0], -1
    for sent in sentences:
        words = {re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", w.lower()) for w in sent.split()}
        score = len(claim_kw & words)
        if score > best_score:
            best_score, best = score, sent
    return best


# ── Step 4: comparison report ─────────────────────────────────────────────────

def _build_report(engine, enriched: list[dict], lang: str) -> str:
    """Single LLM call that compares every (claim, evidence) pair vs web sources."""
    blocks: list[str] = []
    for i, e in enumerate(enriched, 1):
        src_lines = "\n".join(
            f'  [{j}] {s["domain"]}: "{s["key_sentence"]}"  <{s["url"]}>'
            for j, s in enumerate(e["web"], 1)
        ) or "  (no web sources found)"
        blocks.append(
            f"CLAIM {i}: {e['claim']}\n"
            f"VIDEO EVIDENCE: {e['evidence']}\n"
            f"WEB SOURCES:\n{src_lines}"
        )

    context  = "\n\n".join(blocks)
    template = _REPORT_PROMPT_ZH if lang == "zh" else _REPORT_PROMPT_EN
    prompt   = template.format(context=context)
    return engine._llm(prompt, max_tokens=1600)


# ── Source appendix ──────────────────────────────────────────────────────────

_SOURCES_HEADER_ZH = "参考来源"
_SOURCES_HEADER_EN = "Sources Checked"
_SOURCES_NONE_ZH   = "（未找到来源）"
_SOURCES_NONE_EN   = "(no sources found)"


def _format_sources(enriched: list[dict], lang: str) -> str:
    """Build a hard-attached reference block listing every URL that was checked."""
    header    = _SOURCES_HEADER_ZH if lang == "zh" else _SOURCES_HEADER_EN
    none_text = _SOURCES_NONE_ZH   if lang == "zh" else _SOURCES_NONE_EN
    rule      = "─" * 60

    lines = [f"\n\n{rule}", f"  {header}", rule]
    for i, e in enumerate(enriched, 1):
        label = e["claim"][:80] + ("…" if len(e["claim"]) > 80 else "")
        lines.append(f"\n  [{i}] {label}")
        if e["web"]:
            for s in e["web"]:
                title = s["title"][:70] + ("…" if len(s["title"]) > 70 else "")
                lines.append(f"      • {s['domain']} — {title}")
                lines.append(f"        {s['url']}")
        else:
            lines.append(f"      {none_text}")
    lines.append(rule)
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────

def crosscheck(engine, n: int = 5) -> str:
    """Fact-check the top *n* claims from the video against the open web.

    Parameters
    ----------
    engine : QueryEngine
        A loaded QueryEngine instance (provides _llm() and db access).
    n : int
        Number of claims to extract and verify (1-10).

    Returns
    -------
    str
        A structured fact-check report in the video's detected language.
    """
    lang = _detect_lang(engine)
    log.info(f"crosscheck: detected language={lang!r}")

    print(f"\n  [1/3] Extracting top {n} claim-evidence pairs from video …")
    pairs = _extract_claim_pairs(engine, n, lang)
    if not pairs:
        msg = {
            "zh": "  无法从视频内容中提取声明，请检查视频是否已完整处理。",
            "en": "  Could not extract any claims from video content.",
        }
        return msg.get(lang, msg["en"])
    print(f"        {len(pairs)} pair(s) extracted.")

    enriched: list[dict] = []
    for i, pair in enumerate(pairs, 1):
        claim    = (pair.get("claim")    or "").strip()
        evidence = (pair.get("evidence") or "").strip()
        if not claim:
            continue

        label = claim[:70] + ("…" if len(claim) > 70 else "")
        print(f"\n  [2/3] Searching & reranking claim {i}/{len(pairs)}: {label}")

        sources = _search_and_rerank(engine, claim, lang, top_k=3)

        compressed = [
            {
                "domain":       s["domain"],
                "url":          s["url"],
                "title":        s["title"],
                "key_sentence": _compress_snippet(s["snippet"], claim),
            }
            for s in sources
        ] if sources else []

        if not sources:
            print("        No search results found for this claim.")

        enriched.append({"claim": claim, "evidence": evidence, "web": compressed})

    if not enriched:
        return "  All extracted claims were empty after filtering."

    print("\n  [3/3] Generating fact-check report …\n")
    report = _build_report(engine, enriched, lang)
    return report + _format_sources(enriched, lang)
