"""Paths for the citation grounding audit.

Every path in SPEC.md section 2 is defined here exactly once. No module builds
its own path strings; a renamed file changes this file and nothing else.
"""

from pathlib import Path

# Repo root — this file lives at <root>/src/config.py
ROOT = Path(__file__).resolve().parent.parent

# --- data/ -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
PROBE_LOG_CSV = DATA_DIR / "probe_log.csv"      # human-provided export, do not modify
EXTRA_URLS_TXT = DATA_DIR / "extra_urls.txt"    # human-provided, one URL per line
PROBES_JSONL = DATA_DIR / "probes.jsonl"        # step 1
CLAIMS_JSONL = DATA_DIR / "claims.jsonl"        # step 1
SOURCE_URLS_JSON = DATA_DIR / "source_urls.json"  # step 1

# Pages captured through a real browser because adobe.com refuses automated
# HTTP clients from this network. One <sha256-of-url>.txt per page:
#   line 1 = url, line 2 = title, line 3+ = extracted text.
# src.fetch imports these into the page cache before fetching anything else.
MANUAL_PAGES_DIR = DATA_DIR / "manual_pages"

# --- cache/ ------------------------------------------------------------------
# Gitignored. Never re-fetch a page that is already here (SPEC.md section 1).
CACHE_DIR = ROOT / "cache"
PAGES_CACHE_DIR = CACHE_DIR / "pages"           # <sha256>.json, step 2
LLM_CACHE_DIR = CACHE_DIR / "llm"               # <sha256>.json, every LLM call
PAGES_SQLITE = CACHE_DIR / "pages.sqlite"       # mirror of pages/, step 2

# --- corpus/ -----------------------------------------------------------------
CORPUS_DIR = ROOT / "corpus"
CHUNKS_JSONL = CORPUS_DIR / "chunks.jsonl"      # step 4

# --- out/ --------------------------------------------------------------------
OUT_DIR = ROOT / "out"
AUDIT_BASELINE_JSON = OUT_DIR / "audit_baseline.json"   # step 3
HUMAN_LABELS_CSV = OUT_DIR / "human_labels.csv"         # step 3 emits blank; a HUMAN fills it
AGREEMENT_JSON = OUT_DIR / "agreement.json"             # step 3
ANSWERS_EBODA_JSON = OUT_DIR / "answers_eboda.json"     # step 6
AUDIT_EBODA_JSON = OUT_DIR / "audit_eboda.json"         # step 7
COMPARISON_JSON = OUT_DIR / "comparison.json"           # step 7

SITE_DIR = OUT_DIR / "site"                             # step 8
SITE_INDEX_HTML = SITE_DIR / "index.html"
SITE_CSS = SITE_DIR / "eboda-web-toolkit.css"
DOCS_DIR = ROOT / "docs"                                # published by step 8

# When the probes were run against the live Ask beta. This is the one fact on
# the page that no artefact carries — the log records latency and verdicts but
# not the sitting — so it is declared once here and rendered everywhere. It is
# NOT the judging date; that is derived and shown separately as `measured_at`.
PROBED_ON = "8/29/2026"
PROBED_ON_LONG = "8/29/2026, ~2PM MDT"

# --- templates/ --------------------------------------------------------------
TEMPLATES_DIR = ROOT / "templates"
INDEX_TEMPLATE = TEMPLATES_DIR / "index.html.j2"

# Directories created on demand by the steps that write into them.
WRITABLE_DIRS = (
    DATA_DIR,
    CACHE_DIR,
    PAGES_CACHE_DIR,
    LLM_CACHE_DIR,
    CORPUS_DIR,
    OUT_DIR,
    SITE_DIR,
)


def ensure_dirs() -> None:
    """Create every directory this project writes into. Safe to call repeatedly."""
    for d in WRITABLE_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"root = {ROOT}")
    for name, value in sorted(globals().items()):
        if name.isupper() and isinstance(value, Path):
            exists = "ok " if value.exists() else "   "
            print(f"  {exists} {name:24s} {value.relative_to(ROOT)}")
