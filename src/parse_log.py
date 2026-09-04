"""Step 1 — parse the probe log into probes, claims and source URLs.

    python -m src.parse_log

Reads data/probe_log.csv (never writes to it) and emits data/probes.jsonl,
data/claims.jsonl and data/source_urls.json per SPEC.md section 3, applying the
methodology in sections 4.1-4.3.

Shape of the export, as found (not assumed):
  - Row 1 is a title banner; row 2 is the real header.
  - Cited URLs are not a column. They live inside the response text, in a
    trailing block introduced by "Source:" or "Sources:" (sometimes
    parenthesised, e.g. "Sources (3rd only):"), numbered "1. <url>" when there
    is more than one.
  - Inline markers (section 4.2) appear as a bare numeral on its own line
    immediately after the sentence they attach to.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.config import CLAIMS_JSONL, PROBE_LOG_CSV, PROBES_JSONL, SOURCE_URLS_JSON, ensure_dirs

# --- URL normalisation (SPEC.md section 6, step 1) ---------------------------
# Query parameters stripped before deduplication. Compared case-insensitively,
# which covers both "mv" and "MV2".
_DROP_PARAMS = {"instant", "adobe_brand_concierge_source", "promoid", "mv", "mv2"}

_URL_RE = re.compile(r"https?://[^\s<>\"]+")

# --- response structure ------------------------------------------------------
# "Source:", "Sources:", "Sources (3rd only):" — everything from here down is
# the citation block, not answer text.
_SOURCE_BLOCK_RE = re.compile(r"^[ \t]*Sources?\b[^:\n]{0,40}:", re.MULTILINE)
# A link the UI surfaces on a product card, not a citation. See module notes.
_PRICING_LINK_RE = re.compile(r"^[ \t]*Pricing link[^:\n]*:\s*(\S+)", re.MULTILINE)
# "1. https://..." inside the citation block.
_NUMBERED_SOURCE_RE = re.compile(r"^[ \t]*(\d{1,2})[.)]\s*(https?://\S+)", re.MULTILINE)
# A bare numeral alone on a line: an inline citation marker.
_MARKER_LINE_RE = re.compile(r"^[ \t]*(\d{1,2})[ \t]*$")
# Human run annotations the export carries inside the response cell.
_RUN_PREFIX_RE = re.compile(
    r"^\s*(?:\d(?:st|nd|rd|th)(?:\s+and\s+\d(?:st|nd|rd|th))?\s+response|Turn\s*\d+)\s*:\s*",
    re.IGNORECASE,
)

# --- sentence splitting (SPEC.md section 6: a simple regex splitter, no nltk) -
_ABBREVIATIONS = r"(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\bInc)(?<!\bNo)"
# The terminator may sit inside a closing quote or bracket — Adobe writes
# 'a plan named "Creative Cloud Essentials." You might be...' — so accept both
# a bare terminator and one followed by a closing mark.
_TERMINATOR = r"(?:(?<=[.!?])|(?<=[.!?][\"'’”)\]]))"
_SENTENCE_SPLIT_RE = re.compile(_ABBREVIATIONS + _TERMINATOR + r"\s+(?=[\"'(\[“]?[A-Z0-9])")

# --- claim filtering (SPEC.md section 4.3) -----------------------------------
# Openers that mark a sentence as conversational rather than factual. Deliberately
# short: section 4.3 says keep when unsure, so this only catches clear cases.
_NON_FACTUAL_RE = re.compile(
    r"^\s*(?:"
    r"hi\b|hello\b|thanks\b|thank you\b|"
    r"hope (?:this|that) helps|"
    r"i'?m here to help|"
    r"i'?d be happy to|"
    r"would you like|"
    r"do you want|"
    r"are you looking for|"
    r"let me know|"
    r"is there (?:something|anything)|"
    r"what (?:type|kind) of|"
    r"that'?s outside my scope|"
    r"i can(?:'?t| not) help"
    r")",
    re.IGNORECASE,
)
_MIN_CLAIM_WORDS = 4


def normalize_url(url: str) -> str:
    """Strip tracking parameters and fragments so URLs deduplicate correctly."""
    url = url.strip().rstrip(".,;)]’'\"")
    parts = urlsplit(url)
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in _DROP_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))


def split_response(text: str) -> tuple[str, str]:
    """Split a response into (answer body, citation block)."""
    match = _SOURCE_BLOCK_RE.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.start() :]


def parse_sources(block: str) -> tuple[dict[int, str], list[str], list[str]]:
    """Parse the citation block.

    Returns (numbered sources, ordered cited URLs, pricing-card links).

    The pricing link is kept out of cited_urls: the response presents it as a
    click target on a product card, not as a source for its claims. It is still
    fetched, so the corpus keeps the page.
    """
    pricing = [normalize_url(u) for u in _PRICING_LINK_RE.findall(block)]

    numbered: dict[int, str] = {}
    for raw_n, raw_url in _NUMBERED_SOURCE_RE.findall(block):
        url = normalize_url(raw_url)
        if url not in pricing:
            numbered.setdefault(int(raw_n), url)

    if numbered:
        ordered = list(dict.fromkeys(numbered[n] for n in sorted(numbered)))
    else:
        # Unnumbered form: "Source: <url>". Treat it as source 1.
        found = [normalize_url(u) for u in _URL_RE.findall(block)]
        ordered = list(dict.fromkeys(u for u in found if u not in pricing))
        numbered = {i: u for i, u in enumerate(ordered, start=1)}

    return numbered, ordered, pricing


def split_sentences(line: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip()]


def extract_sentences(body: str) -> list[tuple[str, int | None]]:
    """Split the answer body into (sentence, marker) pairs.

    A bare numeral on its own line attaches to the sentence just before it
    (SPEC.md section 4.2). Splitting happens per line first, so bulleted lines
    with no terminal punctuation stay separate sentences.
    """
    pairs: list[tuple[str, int | None]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        marker = _MARKER_LINE_RE.match(raw_line)
        if marker:
            if pairs:
                sentence, existing = pairs[-1]
                # First marker wins if a sentence somehow carries two.
                pairs[-1] = (sentence, existing if existing is not None else int(marker.group(1)))
            continue
        line = _RUN_PREFIX_RE.sub("", line)
        if not line:
            continue
        for sentence in split_sentences(line):
            pairs.append((sentence, None))
    return pairs


def is_factual(sentence: str) -> bool:
    """SPEC.md section 4.3. Drop greetings, questions back to the user and
    pleasantries; keep anything asserting a fact, number, capability, price or
    condition. When unsure, keep."""
    text = sentence.strip()
    if len(text.split()) < _MIN_CLAIM_WORDS:
        return False
    if text.endswith("?"):
        return False
    if _NON_FACTUAL_RE.match(text):
        return False
    return True


def build_probe(row: dict[str, str]) -> tuple[dict, list[dict], list[str]]:
    """Turn one CSV row into a probe record, its claims, and every URL it surfaced."""
    response_text = (row.get("Response summary") or "").strip()
    body, block = split_response(response_text)
    numbered, cited_urls, pricing = parse_sources(block)

    pairs = extract_sentences(body)
    has_inline_markers = any(marker is not None for _, marker in pairs)

    probe_id = row["ID"].strip()
    probe = {
        "id": probe_id,
        "category": _null_if_blank(row.get("Category")),
        "prompt": _null_if_blank(row.get("Prompt")),
        "response_text": response_text or None,
        "cited_urls": cited_urls,
        "has_inline_markers": has_inline_markers,
        "verdict": _null_if_blank(row.get("Verdict")),
        "failure_mode": _null_if_blank(row.get("Failure mode")),
        "persona": _null_if_blank(row.get("Persona affected")),
        "notes": _null_if_blank(row.get("Notes / next action")),
    }

    claims = []
    for sentence, marker in pairs:
        if not is_factual(sentence):
            continue
        # Section 4.2: a resolvable marker overrides; anything ambiguous falls
        # back to all of the response's URLs.
        if marker is not None and marker in numbered:
            claim_urls = [numbered[marker]]
            marker_derived = True
        else:
            claim_urls = list(cited_urls)
            marker_derived = False
        claims.append(
            {
                "claim_id": f"{probe_id}-c{len(claims) + 1:02d}",
                "probe_id": probe_id,
                "system": "baseline",
                "text": sentence,
                "cited_urls": claim_urls,
                "marker_derived": marker_derived,
            }
        )

    return probe, claims, cited_urls + pricing


def _null_if_blank(value: str | None) -> str | None:
    """SPEC.md section 1: a missing field stays null rather than being filled in."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def read_rows() -> list[dict[str, str]]:
    """Read the export. Row 1 is a title banner; row 2 is the header."""
    with PROBE_LOG_CSV.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[1]
    return [dict(zip(header, r)) for r in rows[2:] if any(c.strip() for c in r)]


def write_jsonl(path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ensure_dirs()
    if not PROBE_LOG_CSV.exists():
        print(f"FAIL: {PROBE_LOG_CSV} not found.")
        return 1

    probes, claims, all_urls = [], [], []
    for row in read_rows():
        probe, probe_claims, urls = build_probe(row)
        probes.append(probe)
        claims.extend(probe_claims)
        all_urls.extend(urls)

    unique_urls = sorted(set(all_urls))
    write_jsonl(PROBES_JSONL, probes)
    write_jsonl(CLAIMS_JSONL, claims)
    SOURCE_URLS_JSON.write_text(json.dumps(unique_urls, indent=2) + "\n", encoding="utf-8")

    # --- acceptance check (SPEC.md section 6, step 1) ------------------------
    probes_with_citations = sum(1 for p in probes if p["cited_urls"])
    marker_derived = sum(1 for c in claims if c["marker_derived"])
    marker_derived_pct = round(100 * marker_derived / len(claims), 1) if claims else 0.0
    uncited_claims = sum(1 for c in claims if not c["cited_urls"])

    print(f"probes={len(probes)}")
    print(f"probes_with_citations={probes_with_citations}")
    print(f"unique_urls={len(unique_urls)}")
    print(f"claims={len(claims)}")
    print(f"marker_derived_pct={marker_derived_pct}")
    print()
    print(f"  claims with no cited source (unjudgeable in step 3): {uncited_claims}")
    print(f"  probes with inline markers: {sum(1 for p in probes if p['has_inline_markers'])}")
    print(f"  verdicts: {dict(Counter(p['verdict'] for p in probes))}")

    failures = []
    if len(probes) != 40:
        failures.append(f"probes={len(probes)}, expected 40")
    if not 30 <= len(unique_urls) <= 50:
        failures.append(f"unique_urls={len(unique_urls)}, expected 30-50")
    if len(claims) < 80:
        failures.append(f"claims={len(claims)} — under 80 means the sentence splitter is broken")
    elif not 100 <= len(claims) <= 200:
        failures.append(f"claims={len(claims)}, expected 100-200")

    print()
    if failures:
        print("ACCEPTANCE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
