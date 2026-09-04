"""Step 3 — judge every baseline (claim, source) pair for grounding.

    python -m src.judge

Reads data/claims.jsonl and the page cache, judges each (claim, source) pair
with the SPEC.md section 5.1 prompt, and writes out/audit_baseline.json plus a
blank out/human_labels.csv for a human to fill in.

Two rules from the spec govern what the judge sees:
  - Section 4.4: the judge never sees the URL, the question, or the brand. Only
    the claim text and the source text. Do not add context to the prompt.
  - Section 4.5: a source that could not be fetched is not judged against
    nothing. It is flagged in its own field and excluded from the headline rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone

from src.config import (
    AUDIT_BASELINE_JSON,
    CLAIMS_JSONL,
    HUMAN_LABELS_CSV,
    PAGES_CACHE_DIR,
    ensure_dirs,
)
from src.llm import MODEL_JUDGE, call

# --- prompt (SPEC.md section 5.1) --------------------------------------------
# VERBATIM. Do not reword, do not add few-shot examples, do not add the URL,
# the question, or the brand. This prompt is the instrument.
JUDGE_SYSTEM = "You are a careful evaluator. You respond with JSON only."

JUDGE_USER_TEMPLATE = """You are evaluating whether a piece of source text supports a specific factual claim.

<source>
{source_text}
</source>

<claim>
{claim_text}
</claim>

Decide which single label applies:

- "Supported" — the source states the claim, or states something that entails it. Every specific
  detail in the claim (numbers, product names, conditions) appears in or follows from the source.
- "Partially supported" — the source supports the general shape of the claim but not all of its
  specifics. Use this when a number, product name, condition, or qualifier in the claim is absent
  from the source or differs from it.
- "Unsupported" — the source is about the same subject area but does not state or entail the claim.
- "Source unrelated" — the source is about a different subject entirely. It could not support this
  claim regardless of what the claim said.

Rules:
- Judge only against the source text above. Do not use outside knowledge. If you know the claim is
  true but the source does not say so, the label is "Unsupported".
- evidence_span must be copied verbatim from the source, or null. Never paraphrase it.
- If the source text is empty, or is navigation and boilerplate only, use "Source unrelated".

Respond with JSON only, no preamble:
{{"label": "...", "evidence_span": "..." or null, "reason": "20 words or fewer"}}"""

SOURCE_TRUNCATE_CHARS = 12_000

# The label vocabulary, best first. Section 4.1: a claim's final label is the
# BEST label across its sources — the charitable reading. Defined once here and
# imported by src.compare so the two audits can never diverge.
LABEL_ORDER = ["Supported", "Partially supported", "Unsupported", "Source unrelated"]
UNFETCHABLE = "source_unfetchable"

# The judge occasionally abbreviates a label ("Unrelated" for "Source
# unrelated"). These are unambiguous shorthands for the four labels above, so
# they are normalised on the way out rather than by editing the section 5.1
# prompt, which is the instrument and stays verbatim. Every normalisation is
# counted and reported so it is never a silent rewrite.
_LABEL_ALIASES = {
    "supported": "Supported",
    "partially supported": "Partially supported",
    "partial": "Partially supported",
    "partially": "Partially supported",
    "unsupported": "Unsupported",
    "not supported": "Unsupported",
    "source unrelated": "Source unrelated",
    "unrelated": "Source unrelated",
}
NORMALISED_LABELS: list[tuple[str, str, str]] = []  # (claim_id, raw, canonical)


def canonical_label(raw, claim_id: str) -> str | None:
    """Map the judge's label onto the four-label vocabulary, or return None."""
    if not isinstance(raw, str):
        return None
    if raw in LABEL_ORDER:
        return raw
    canonical = _LABEL_ALIASES.get(" ".join(raw.strip().split()).lower())
    if canonical:
        NORMALISED_LABELS.append((claim_id, raw, canonical))
    return canonical

HUMAN_SAMPLE_SIZE = 15
HUMAN_SAMPLE_SEED = 42


def best_label(labels) -> str | None:
    """Section 4.1: the claim's label is the best one across its sources."""
    ranked = [l for l in LABEL_ORDER if l in set(labels)]
    return ranked[0] if ranked else None


def _next_round() -> int:
    """Next free suffix for an archived labelling sheet."""
    existing = list(HUMAN_LABELS_CSV.parent.glob(f"{HUMAN_LABELS_CSV.stem}_round*.csv"))
    return len(existing) + 1


def judge_cache_key(claim_text: str, source_sha: str) -> str:
    """SPEC.md section 6 step 3: cache on sha256(claim_text + source_sha)."""
    return hashlib.sha256((claim_text + source_sha).encode("utf-8")).hexdigest()


def load_pages() -> dict[str, dict]:
    pages = {}
    for path in PAGES_CACHE_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        pages[record["url_normalized"]] = record
    return pages


def load_claims(system: str = "baseline") -> list[dict]:
    claims = [json.loads(l) for l in CLAIMS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [c for c in claims if c["system"] == system]


def judge_pair(claim_text: str, source_text: str) -> dict:
    """One judgement. Returns the parsed JSON from the model."""
    truncated = source_text[:SOURCE_TRUNCATE_CHARS]
    user = JUDGE_USER_TEMPLATE.format(source_text=truncated, claim_text=claim_text)
    return call(JUDGE_SYSTEM, user, MODEL_JUDGE)


def build_record(claim: dict, source_ref: str, page: dict | None) -> dict:
    """Judge one (claim, source) pair and return an audit record (section 3)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "claim_id": claim["claim_id"],
        "probe_id": claim["probe_id"],
        "system": claim["system"],
        "source_ref": source_ref,
        "label": None,
        "evidence_span": None,
        "reason": None,
        "judge_model": MODEL_JUDGE,
        "judged_at": now,
        # Section 4.5: its own field, excluded from the headline rate.
        "source_unfetchable": False,
    }

    if page is None or page.get("status") != "ok" or not page.get("text"):
        record["source_unfetchable"] = True
        record["reason"] = f"source not retrievable (status={page['status'] if page else 'missing'})"
        record["judge_model"] = None
        return record

    parsed = judge_pair(claim["text"], page["text"])
    label = canonical_label(parsed.get("label"), claim["claim_id"])
    if label is None:
        raise ValueError(
            f"{claim['claim_id']}: judge returned unknown label {parsed.get('label')!r}"
        )

    record["label"] = label
    record["evidence_span"] = parsed.get("evidence_span")
    record["reason"] = parsed.get("reason")
    return record


def write_human_labels(records: list[dict], pages: dict[str, dict], seed: int = HUMAN_SAMPLE_SEED) -> int:
    """Emit the blank sheet a human fills in (SPEC.md section 6 step 3, section 8).

    model_label is left BLANK on purpose so the labeller is not anchored. The
    true model labels stay in out/audit_baseline.json.
    """
    claims = {c["claim_id"]: c for c in load_claims()}
    judgeable = [r for r in records if not r["source_unfetchable"]]

    rng = random.Random(seed)
    sample = rng.sample(judgeable, min(HUMAN_SAMPLE_SIZE, len(judgeable)))

    # Never silently overwrite a sheet someone has already filled in.
    if HUMAN_LABELS_CSV.exists():
        with HUMAN_LABELS_CSV.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
        if any(r.get("human_label", "").strip() for r in existing):
            backup = HUMAN_LABELS_CSV.with_name(
                f"{HUMAN_LABELS_CSV.stem}_round{_next_round():02d}.csv"
            )
            HUMAN_LABELS_CSV.rename(backup)
            print(f"  preserved previous labelled sheet as {backup.name}")

    with HUMAN_LABELS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["claim_id", "source_ref", "claim_text", "source_excerpt", "model_label", "human_label", "note"]
        )
        for r in sample:
            page = pages.get(r["source_ref"], {})
            writer.writerow(
                [
                    r["claim_id"],
                    r["source_ref"],
                    claims[r["claim_id"]]["text"],
                    (page.get("text") or "")[:SOURCE_TRUNCATE_CHARS],
                    "",  # model_label — deliberately blank, a human fills human_label
                    "",
                    "",
                ]
            )
    return len(sample)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge baseline (claim, source) pairs.")
    parser.add_argument(
        "--seed",
        type=int,
        default=HUMAN_SAMPLE_SEED,
        help="sampling seed for human_labels.csv (SPEC.md specifies 42; change only to draw "
             "a fresh independent sample, and record which seed produced which sheet)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    claims = load_claims("baseline")
    pages = load_pages()

    pairs = [(c, url) for c in claims for url in c["cited_urls"]]
    print(f"claims={len(claims)}  pairs={len(pairs)}  judge_model={MODEL_JUDGE}")
    print(f"claims with no cited source (no pairs, unjudgeable): "
          f"{sum(1 for c in claims if not c['cited_urls'])}")
    print()

    truncated = 0
    records = []
    for i, (claim, url) in enumerate(pairs, start=1):
        page = pages.get(url)
        if page and page.get("text") and len(page["text"]) > SOURCE_TRUNCATE_CHARS:
            truncated += 1
            print(f"  [truncate] {claim['claim_id']} source {len(page['text'])} chars -> {SOURCE_TRUNCATE_CHARS}")
        records.append(build_record(claim, url, page))
        if i % 25 == 0 or i == len(pairs):
            print(f"  judged {i}/{len(pairs)}")

    AUDIT_BASELINE_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_sampled = write_human_labels(records, pages, seed=args.seed)

    # --- acceptance check (SPEC.md section 6, step 3) ------------------------
    counts = Counter(r["label"] for r in records if not r["source_unfetchable"])
    unfetchable = sum(1 for r in records if r["source_unfetchable"])

    print()
    print(f"audit_baseline.json: {len(records)} records for {len(pairs)} (claim, source) pairs")
    print(f"sources truncated to {SOURCE_TRUNCATE_CHARS} chars: {truncated}")
    print(f"labels normalised from a shorthand spelling: {len(NORMALISED_LABELS)}")
    for claim_id, raw, canonical in NORMALISED_LABELS:
        print(f"  {claim_id}: {raw!r} -> {canonical!r}")
    print()
    print("label distribution (per claim-source pair):")
    for label in LABEL_ORDER:
        n = counts.get(label, 0)
        pct = 100 * n / max(1, sum(counts.values()))
        print(f"  {label:22s} {n:4d}  {pct:5.1f}%")
    print(f"  {UNFETCHABLE:22s} {unfetchable:4d}   (excluded from the headline rate)")

    # Section 4.1: per claim, the best label across its sources.
    per_claim = {}
    for r in records:
        if not r["source_unfetchable"]:
            per_claim.setdefault(r["claim_id"], []).append(r["label"])
    claim_labels = Counter(best_label(v) for v in per_claim.values())
    print()
    print(f"per-claim best label (section 4.1), {len(per_claim)} judged claims:")
    for label in LABEL_ORDER:
        n = claim_labels.get(label, 0)
        print(f"  {label:22s} {n:4d}  {100 * n / max(1, len(per_claim)):5.1f}%")

    print()
    print(f"human_labels.csv: {n_sampled} rows, model_label blank (seed {args.seed})")

    failures = []
    if len(records) != len(pairs):
        failures.append(f"{len(records)} records for {len(pairs)} pairs")
    if n_sampled != HUMAN_SAMPLE_SIZE:
        failures.append(f"human_labels.csv has {n_sampled} rows, expected {HUMAN_SAMPLE_SIZE}")

    print()
    if failures:
        print("ACCEPTANCE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("ACCEPTANCE: PASS")
    print()
    print("STOP — handing off to the human (SPEC.md section 8).")
    print(f"  1. Fill in the human_label column of {HUMAN_LABELS_CSV}")
    print(f"     Use exactly one of: {', '.join(LABEL_ORDER)}")
    print("  2. Run `python -m src.agreement`")
    print("  3. Step 4 does not start until agreement.json shows >= 0.80.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
