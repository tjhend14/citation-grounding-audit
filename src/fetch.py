"""Step 2 — fetch every cited page into the on-disk cache.

    python -m src.fetch

Reads data/source_urls.json and data/extra_urls.txt, fetches anything not
already cached, and writes one cache/pages/<sha256>.json per URL (SPEC.md
section 3). Mirrors the result into cache/pages.sqlite.

A URL whose cache file already exists is never re-fetched (SPEC.md section 1).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import httpx
import trafilatura

from src.config import (
    EXTRA_URLS_TXT,
    MANUAL_PAGES_DIR,
    PAGES_CACHE_DIR,
    PAGES_SQLITE,
    SOURCE_URLS_JSON,
    ensure_dirs,
)
from src.parse_log import normalize_url

TIMEOUT = 20.0          # seconds
SLEEP_BETWEEN = 1.5     # seconds between requests
RETRIES_5XX = 2         # retries on a 5xx, on top of the first attempt

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# SPEC.md section 3: status is one of these five.
STATUSES = ("ok", "http_error", "empty", "timeout", "blocked")

SUCCESS_THRESHOLD = 0.60  # below this, stop and hand back to the human

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def url_sha(url_normalized: str) -> str:
    """Cache key for a page. Step 3 reuses this as the source_sha."""
    return hashlib.sha256(url_normalized.encode("utf-8")).hexdigest()


def cache_path(url_normalized: str):
    return PAGES_CACHE_DIR / (url_sha(url_normalized) + ".json")


def load_urls() -> tuple[list[str], int, int]:
    """Every URL to fetch: the cited ones plus data/extra_urls.txt."""
    cited = [normalize_url(u) for u in json.loads(SOURCE_URLS_JSON.read_text(encoding="utf-8"))]

    extra: list[str] = []
    if EXTRA_URLS_TXT.exists():
        for line in EXTRA_URLS_TXT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra.append(normalize_url(line))

    combined = list(dict.fromkeys(cited + extra))
    return combined, len(cited), len(extra)


def extract_title(html: str) -> str | None:
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            return meta.title.strip()
    except Exception:
        pass
    match = _TITLE_RE.search(html)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip() or None
    return None


def fetch_one(client: httpx.Client, url: str) -> dict:
    """Fetch one URL and build its cache record. Never raises."""
    record = {
        "url": url,
        "url_normalized": url,
        "sha": url_sha(url),
        "status": None,
        "http_status": None,
        "title": None,
        "text": None,
        "char_count": 0,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    response = None
    for attempt in range(RETRIES_5XX + 1):
        try:
            response = client.get(url)
        except httpx.TimeoutException:
            record["status"] = "timeout"
            return record
        except httpx.HTTPError:
            # DNS, TLS, connection reset. Not one of the named buckets; the
            # closest honest label is http_error with no status code.
            record["status"] = "http_error"
            return record

        if response.status_code < 500:
            break
        if attempt < RETRIES_5XX:
            time.sleep(SLEEP_BETWEEN * (attempt + 1))

    record["http_status"] = response.status_code

    if response.status_code in (401, 403, 429):
        record["status"] = "blocked"
        return record
    if not response.is_success:
        record["status"] = "http_error"
        return record

    text = trafilatura.extract(response.text, include_comments=False, include_tables=True)
    record["title"] = extract_title(response.text)
    if not text or not text.strip():
        record["status"] = "empty"
        return record

    record["text"] = text.strip()
    record["char_count"] = len(record["text"])
    record["status"] = "ok"
    return record


def import_manual_pages() -> int:
    """Turn browser-captured pages into normal cache records.

    adobe.com accepts the TCP connection from an automated HTTP client and then
    never answers, so the pages are captured through a real browser instead and
    dropped into data/manual_pages/. The text is the same text; only the
    transport differs. Records are marked http_status 200 / status ok because
    that is what the browser received.
    """
    if not MANUAL_PAGES_DIR.exists():
        return 0

    imported = 0
    for path in sorted(MANUAL_PAGES_DIR.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").split("\n")
        if len(lines) < 3:
            print(f"  skipping malformed capture: {path.name}")
            continue

        url = normalize_url(lines[0].strip())
        title = lines[1].strip() or None
        text = "\n".join(lines[2:]).strip()

        cache_file = cache_path(url)
        if cache_file.exists():
            continue

        record = {
            "url": url,
            "url_normalized": url,
            "sha": url_sha(url),
            "status": "ok" if text else "empty",
            "http_status": 200,
            "title": title,
            "text": text or None,
            "char_count": len(text),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_cache(record)
        imported += 1

    return imported


def write_cache(record: dict) -> None:
    path = cache_path(record["url_normalized"])
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def mirror_sqlite(records: list[dict]) -> None:
    """Rebuild the SQLite mirror from the cache files (SPEC.md section 6, step 2)."""
    conn = sqlite3.connect(PAGES_SQLITE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            url_normalized TEXT PRIMARY KEY,
            sha            TEXT,
            status         TEXT,
            http_status    INTEGER,
            title          TEXT,
            text           TEXT,
            char_count     INTEGER,
            fetched_at     TEXT
        )
        """
    )
    conn.executemany(
        "INSERT OR REPLACE INTO pages "
        "(url_normalized, sha, status, http_status, title, text, char_count, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["url_normalized"],
                r["sha"],
                r["status"],
                r["http_status"],
                r["title"],
                r["text"],
                r["char_count"],
                r["fetched_at"],
            )
            for r in records
        ],
    )
    conn.commit()
    conn.close()


def main() -> int:
    ensure_dirs()
    if not SOURCE_URLS_JSON.exists():
        print(f"FAIL: {SOURCE_URLS_JSON} not found. Run `python -m src.parse_log` first.")
        return 1

    urls, n_cited, n_extra = load_urls()
    print(f"{len(urls)} urls to cover ({n_cited} cited, {n_extra} from extra_urls.txt)")
    if not EXTRA_URLS_TXT.exists():
        print(f"note: {EXTRA_URLS_TXT.name} does not exist — treating it as empty, nothing invented")

    imported = import_manual_pages()
    if imported:
        print(f"imported {imported} browser-captured pages from {MANUAL_PAGES_DIR.name}/")
    print()

    records: list[dict] = []
    fetched = skipped = 0

    with httpx.Client(
        timeout=TIMEOUT, headers=HEADERS, follow_redirects=True, http2=False
    ) as client:
        for i, url in enumerate(urls, start=1):
            path = cache_path(url)
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
                skipped += 1
                continue

            if fetched:
                time.sleep(SLEEP_BETWEEN)
            record = fetch_one(client, url)
            write_cache(record)
            records.append(record)
            fetched += 1
            print(
                f"  [{i:2d}/{len(urls)}] {record['status']:10s} "
                f"{str(record['http_status'] or '—'):>4s}  {record['char_count']:6d} chars  {url[:78]}"
            )

    mirror_sqlite(records)

    # --- acceptance check (SPEC.md section 6, step 2) ------------------------
    counts = Counter(r["status"] for r in records)
    ok = counts.get("ok", 0)
    total = len(records)
    success_rate = ok / total if total else 0.0

    print()
    print(f"fetched={fetched}  skipped_already_cached={skipped}")
    print("status breakdown:")
    for status in STATUSES:
        if counts.get(status):
            print(f"  {status:12s} {counts[status]:3d}")
    for status, n in sorted(counts.items()):
        if status not in STATUSES:
            print(f"  {status:12s} {n:3d}   <-- not in the SPEC section 3 vocabulary")
    print(f"success_rate={success_rate:.2f}  ({ok}/{total})")
    print()

    if success_rate < SUCCESS_THRESHOLD:
        print("ACCEPTANCE: FAIL — success_rate is below 0.60.")
        print()
        print("STOPPING. The concierge would have nothing to retrieve from, so step 4")
        print("onward cannot produce a defensible number. SPEC.md section 6 step 2 says the")
        print("fallback — auditing six probes by hand — is a human decision, not mine.")
        return 1

    print("ACCEPTANCE: PASS — success_rate >= 0.60, safe to continue to step 3/4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
