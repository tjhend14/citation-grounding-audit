"""Step 7 — audit the concierge's output and compare it to the baseline.

    python -m src.compare

Judges every sentence the concierge generated against every chunk it cited,
using the SPEC.md section 5.1 judge — the same prompt, model and cache that
scored Adobe. Writes out/audit_eboda.json and out/comparison.json.

Why every generated sentence, and not just the ones the gate kept: the gate
keeps precisely the sentences the judge labelled Supported or Partially
supported, so scoring only those would return 1.00 by construction and measure
nothing. Judging the generator's raw output makes the two systems directly
comparable — both are unfiltered model output scored by the same judge — and
the gate's contribution is reported separately in the "gate" block.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone

from src.config import (
    AGREEMENT_JSON,
    ANSWERS_EBODA_JSON,
    AUDIT_BASELINE_JSON,
    AUDIT_EBODA_JSON,
    CHUNKS_JSONL,
    COMPARISON_JSON,
    ensure_dirs,
)
from src.judge import LABEL_ORDER, best_label, judge_pair
from src.llm import MODEL_JUDGE

# The two labels that count as grounded.
GROUNDED_LABELS = ("Supported", "Partially supported")


def grounding_rate(labels: list[str]) -> float:
    """(Supported + Partially supported) / judged claims.

    THE definition, used for both systems. SPEC.md section 7 requires this be
    written once and only once — do not recompute it anywhere else, import it.

    `labels` is one label per CLAIM (or per generated sentence for the
    concierge), already rolled up across that claim's sources by section 4.1's
    best-label rule. Claims with no judgeable source are not in this list and so
    are excluded from both numerator and denominator; they are reported
    alongside as a separate count (section 4.5).
    """
    if not labels:
        return 0.0
    return sum(1 for l in labels if l in GROUNDED_LABELS) / len(labels)


def load_chunks() -> dict[str, dict]:
    return {
        c["chunk_id"]: c
        for c in (json.loads(l) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip())
    }


def audit_eboda(chunks: dict[str, dict]) -> list[dict]:
    """One judgement record per (generated sentence, cited chunk) pair."""
    answers = json.loads(ANSWERS_EBODA_JSON.read_text(encoding="utf-8"))
    records: list[dict] = []

    for entry in answers:
        probe_id = entry["probe_id"]
        sentences = entry["kept"] + entry["dropped"]
        for i, item in enumerate(sentences, start=1):
            claim_id = f"{probe_id}-e{i:02d}"
            cited = [cid for cid in (item.get("chunk_ids") or []) if cid in chunks]
            if not cited:
                continue  # no source to judge against, like an uncited baseline claim
            for chunk_id in cited:
                judgement = judge_pair(item["sentence"], chunks[chunk_id]["text"])
                records.append(
                    {
                        "claim_id": claim_id,
                        "probe_id": probe_id,
                        "system": "eboda",
                        "source_ref": chunk_id,
                        "label": judgement.get("label"),
                        "evidence_span": judgement.get("evidence_span"),
                        "reason": judgement.get("reason"),
                        "judge_model": MODEL_JUDGE,
                        "judged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "source_unfetchable": False,
                        "served": item in entry["kept"] and not entry["abstained"],
                    }
                )
    return records


def rollup(records: list[dict]) -> tuple[list[str], int]:
    """Section 4.1: per claim, the BEST label across its sources.

    Returns (labels, unfetchable_claim_count).
    """
    per_claim: dict[str, list[str]] = {}
    unfetchable: set[str] = set()
    for r in records:
        if r.get("source_unfetchable"):
            unfetchable.add(r["claim_id"])
            continue
        per_claim.setdefault(r["claim_id"], []).append(r["label"])
    unfetchable -= set(per_claim)
    return [best_label(v) for v in per_claim.values()], len(unfetchable)


def block(records: list[dict], include_unfetchable: bool) -> dict:
    labels, unfetchable = rollup(records)
    counts = Counter(labels)
    out = {
        "claims": len(labels),
        "supported": counts.get("Supported", 0),
        "partially": counts.get("Partially supported", 0),
        "unsupported": counts.get("Unsupported", 0),
        "unrelated": counts.get("Source unrelated", 0),
    }
    if include_unfetchable:
        out["unfetchable"] = unfetchable
    out["grounding_rate"] = round(grounding_rate(labels), 4)
    return out


def main() -> int:
    ensure_dirs()
    for path in (AUDIT_BASELINE_JSON, ANSWERS_EBODA_JSON, CHUNKS_JSONL):
        if not path.exists():
            print(f"FAIL: {path} not found.")
            return 1

    chunks = load_chunks()
    baseline = json.loads(AUDIT_BASELINE_JSON.read_text(encoding="utf-8"))

    print(f"judging concierge output with {MODEL_JUDGE} ...")
    eboda = audit_eboda(chunks)
    AUDIT_EBODA_JSON.write_text(json.dumps(eboda, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {AUDIT_EBODA_JSON} ({len(eboda)} records)")

    answers = json.loads(ANSWERS_EBODA_JSON.read_text(encoding="utf-8"))
    gate = {
        "sentences_dropped": sum(len(a["dropped"]) for a in answers),
        "answers_abstained": sum(1 for a in answers if a["abstained"]),
    }

    human_agreement = 0.0
    if AGREEMENT_JSON.exists():
        human_agreement = json.loads(AGREEMENT_JSON.read_text(encoding="utf-8"))["exact_match_agreement"]

    comparison = {
        "baseline": block(baseline, include_unfetchable=True),
        "eboda": block(eboda, include_unfetchable=False),
        "gate": gate,
        "human_agreement": human_agreement,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(json.dumps(comparison, indent=2))

    # --- context the exact-shape file cannot carry --------------------------
    served = [r for r in eboda if r["served"]]
    served_labels, _ = rollup(served)
    print()
    print("context (not part of comparison.json):")
    print(f"  eboda grounding_rate over ALL generated sentences : {comparison['eboda']['grounding_rate']:.2f}")
    print(f"  eboda grounding_rate over only what it SERVED     : {grounding_rate(served_labels):.2f}"
          f"  ({len(served_labels)} sentences — near 1.00 by construction, the gate keeps what the judge passes)")
    print(f"  answers abstained                                 : {gate['answers_abstained']}/{len(answers)}")

    # --- acceptance check (SPEC.md section 6, step 7) -----------------------
    vocab_baseline = {r["label"] for r in baseline if r["label"]}
    vocab_eboda = {r["label"] for r in eboda if r["label"]}
    failures = []
    if not vocab_baseline <= set(LABEL_ORDER) or not vocab_eboda <= set(LABEL_ORDER):
        failures.append(f"label vocabulary drift: baseline={vocab_baseline} eboda={vocab_eboda}")
    for name in ("baseline", "eboda"):
        rate = comparison[name]["grounding_rate"]
        if not 0.0 <= rate <= 1.0:
            failures.append(f"{name} grounding_rate={rate} outside [0, 1]")

    print()
    print(f"label vocabulary — baseline: {sorted(vocab_baseline)}")
    print(f"label vocabulary — eboda:    {sorted(vocab_eboda)}")
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
