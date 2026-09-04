"""Step 4 — turn the page cache into a retrievable corpus.

    python -m src.chunk

Chunks by heading, splits anything over ~1800 characters on paragraph
boundaries, drops chunks under 120 characters, and writes corpus/chunks.jsonl
(SPEC.md section 3).

Boilerplate stripping matters here. The Adobe global footer appears on nearly
every page; if it survives it appears in almost every chunk, and BM25 will
happily rank it above real content for any query that shares a word with it.
The same goes for cookie banners and the "Shop for / For business / Experience
Cloud" link blocks.

Small sections are MERGED rather than emitted individually. Adobe's technical
requirements pages use one-word headings ("Memory", "GPU", "Storage") whose
sections are far under the 120-character floor; emitting them separately would
drop exactly the specs the audit cares about.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter

from src.config import CHUNKS_JSONL, PAGES_CACHE_DIR, ensure_dirs

TARGET_CHARS = 1800   # split sections longer than this
MIN_CHARS = 120       # drop chunks shorter than this
MAX_CHUNKS_PER_PAGE = 60  # above this the splitter is broken (section 6, step 4)

SAMPLE_SEED = 42
SAMPLE_SIZE = 3

# --- boilerplate ------------------------------------------------------------
# Whole lines matching any of these are dropped before chunking. Anchored and
# specific: the goal is to remove chrome, never to remove a price, a spec value
# or a product claim.
_BOILERPLATE_LINE = re.compile(
    r"""^\s*(?:
        # global nav / footer link blocks named in SPEC.md section 6 step 4
        Shop\s+for\b .* |
        For\s+business\b\s* |
        Experience\s+Cloud\b\s* |
        Creative\s+Cloud\b\s* |
        Document\s+Cloud\b\s* |
        (?:Learn|Community|Support|Company|Adobe\s+Home|Adobe\s+Live|Behance)\s* |
        View\s+all\s+products\s* |
        # helpx article furniture
        Was\s+this\s+page\s+helpful.*  |
        (?:Yes,\s*thanks|Not\s+really)\s* |
        (?:Previous|Next)\s* |
        Last\s+updated\s+on\b.* |
        # cookie / consent banners
        .*(?:we\s+use\s+cookies|cookie\s+preferences|accept\s+all\s+cookies).* |
        # call-to-action stubs, no informational content
        (?:Buy\s+now|Free\s+trial|Save\s+today|Watch\s+video|Watch\s+overview|
           Watch\s+tutorial|Learn\s+more|See\s+terms|Contact\s+sales|Open\s+the\s+app|
           See\s+all\s+plans|Compare\s+plans|Sign\s+in\s+to\s+Admin\s+Console|
           Explore\s+Knowledge\s+Base|Get\s+inspired|See\s+what's\s+new)\s*[.›>]?\s* |
        # legal / trademark / AI disclaimers repeated across pages
        .*(?:registered\s+trademarks?|trademarks\s+of\s+Adobe).* |
        Use\s+of\s+this\s+beta\s+AI\s+chatbot\b.* |
        .*Generative\s+AI\s+Terms.*
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# Blocks that run from a marker line to the next blank line, e.g. the list of
# language versions at the foot of every helpx article.
_BOILERPLATE_BLOCK_START = re.compile(
    r"^\s*(?:Language\s+versions\s+available|Adobe,\s+the\s+Adobe\s+logo)\b", re.IGNORECASE
)

# --- headings ---------------------------------------------------------------
_SENTENCE_END = (".", ",", ";", ":", "!", "?")


def is_heading(line: str, nxt: str | None) -> bool:
    """A short label line introducing the text beneath it."""
    text = line.strip()
    if not text or len(text) > 80:
        return False
    if text.isupper() and len(text) > 2:
        return True
    if len(text.split()) > 10:
        return False
    # A short line with no terminal punctuation, followed by something.
    return not text.endswith(_SENTENCE_END) and bool(nxt)


def strip_boilerplate(text: str) -> tuple[str, int]:
    """Remove nav, footer, cookie and CTA lines. Returns (text, lines_removed)."""
    kept, removed, skipping = [], 0, False
    lines = text.split("\n")
    for line in lines:
        if skipping:
            if not line.strip():
                skipping = False
            else:
                removed += 1
            continue
        if _BOILERPLATE_BLOCK_START.match(line):
            skipping = True
            removed += 1
            continue
        if line.strip() and _BOILERPLATE_LINE.match(line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, body) sections."""
    lines = [l.rstrip() for l in text.split("\n")]
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current: list[str] = []

    for i, line in enumerate(lines):
        nxt = next((l for l in lines[i + 1:] if l.strip()), None)
        if line.strip() and is_heading(line, nxt):
            if any(l.strip() for l in current):
                sections.append((current_heading, current))
            current_heading, current = line.strip(), []
        else:
            current.append(line)
    if any(l.strip() for l in current):
        sections.append((current_heading, current))

    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def split_long(body: str) -> list[str]:
    """Split an over-long section on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts, buffer = [], ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) + 2 > TARGET_CHARS:
            parts.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer:
        parts.append(buffer)

    # A single paragraph longer than the target still has to be broken up.
    final = []
    for part in parts:
        while len(part) > TARGET_CHARS * 1.5:
            cut = part.rfind("\n", 0, TARGET_CHARS)
            if cut < MIN_CHARS:
                cut = TARGET_CHARS
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def merge_sections(sections: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    """Merge consecutive sections up to the target size.

    Keeps the first heading of the merged run. Without this, one-word headings
    like "Memory" produce sub-120-character chunks that the floor then deletes,
    taking the system requirements with them.
    """
    merged: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer = ""

    for h, body in sections:
        piece = f"{h}\n{body}" if h else body
        if buffer and len(buffer) + len(piece) + 1 > TARGET_CHARS:
            merged.append((heading, buffer))
            heading, buffer = h, piece
        else:
            if not buffer:
                heading = h
            buffer = f"{buffer}\n{piece}" if buffer else piece
    if buffer:
        merged.append((heading, buffer))
    return merged


def chunk_page(page: dict, counter: list[int]) -> list[dict]:
    text, _ = strip_boilerplate(page["text"])
    sections = merge_sections(split_sections(text))

    chunks = []
    for heading, body in sections:
        for part in (split_long(body) if len(body) > TARGET_CHARS else [body]):
            part = part.strip()
            if len(part) < MIN_CHARS:
                continue
            counter[0] += 1
            chunks.append(
                {
                    "chunk_id": f"c{counter[0]:04d}",
                    "url": page["url_normalized"],
                    "page_title": page.get("title"),
                    "heading": heading,
                    "text": part,
                    "char_count": len(part),
                }
            )
    return chunks


def main() -> int:
    ensure_dirs()
    pages = sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in PAGES_CACHE_DIR.glob("*.json")),
        key=lambda r: r["url_normalized"],
    )
    usable = [p for p in pages if p.get("status") == "ok" and p.get("text")]
    print(f"pages in cache: {len(pages)}  usable: {len(usable)}")

    counter = [0]
    all_chunks, removed_total, dropped_short = [], 0, 0
    per_page = {}

    for page in usable:
        _, removed = strip_boilerplate(page["text"])
        removed_total += removed
        chunks = chunk_page(page, counter)
        per_page[page["url_normalized"]] = len(chunks)
        all_chunks.extend(chunks)

    # count what the floor discarded, for reporting
    for page in usable:
        text, _ = strip_boilerplate(page["text"])
        for heading, body in merge_sections(split_sections(text)):
            for part in (split_long(body) if len(body) > TARGET_CHARS else [body]):
                if 0 < len(part.strip()) < MIN_CHARS:
                    dropped_short += 1

    with CHUNKS_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- acceptance check (SPEC.md section 6, step 4) ------------------------
    counts = sorted(per_page.values())
    pages_represented = sum(1 for n in counts if n)
    print()
    print(f"chunks={len(all_chunks)}")
    print(f"pages_represented={pages_represented}")
    print(
        f"chunks_per_page min={min(counts)} median={statistics.median(counts):.0f} max={max(counts)}"
    )
    print()
    print(f"boilerplate lines stripped: {removed_total}")
    print(f"chunks dropped for being under {MIN_CHARS} chars: {dropped_short}")
    sizes = sorted(c["char_count"] for c in all_chunks)
    print(
        f"chunk size min={sizes[0]} median={statistics.median(sizes):.0f} max={sizes[-1]}"
    )
    with_heading = sum(1 for c in all_chunks if c["heading"])
    print(f"chunks with a heading: {with_heading}/{len(all_chunks)}")

    print()
    print(f"--- {SAMPLE_SIZE} random chunks (seed {SAMPLE_SEED}) for eyeball inspection ---")
    for c in random.Random(SAMPLE_SEED).sample(all_chunks, min(SAMPLE_SIZE, len(all_chunks))):
        print()
        print(f"[{c['chunk_id']}] {c['char_count']} chars")
        print(f"  page:    {c['page_title']}")
        print(f"  url:     {c['url']}")
        print(f"  heading: {c['heading']!r}")
        print("  text:")
        for line in c["text"].split("\n"):
            print(f"    {line}")

    failures = []
    if max(counts) > MAX_CHUNKS_PER_PAGE:
        failures.append(f"chunks_per_page max={max(counts)} > {MAX_CHUNKS_PER_PAGE} — splitter is broken")
    if pages_represented < len(usable):
        missing = [u for u, n in per_page.items() if not n]
        failures.append(f"{len(missing)} usable page(s) produced no chunks: {missing[:3]}")

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
