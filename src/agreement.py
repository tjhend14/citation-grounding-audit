"""Step 3 (second half) — human/model agreement on the 15-row sample.

    python -m src.agreement

Reads the human-filled out/human_labels.csv, joins each row to the model's
label in out/audit_baseline.json, prints exact-match agreement and a confusion
matrix, and writes out/agreement.json.

The CSV is filled in by a HUMAN (SPEC.md section 8). This module only reads it.
Step 4 does not start until agreement is >= 0.80; if it is lower, the rubric
needs a human fix, not a code fix.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone

from src.config import AGREEMENT_JSON, AUDIT_BASELINE_JSON, HUMAN_LABELS_CSV, ensure_dirs
from src.judge import LABEL_ORDER

AGREEMENT_THRESHOLD = 0.80


def normalize_label(raw: str) -> str | None:
    """Accept a human label case-insensitively; return the canonical spelling."""
    if not raw or not raw.strip():
        return None
    cleaned = " ".join(raw.strip().split()).lower()
    for label in LABEL_ORDER:
        if cleaned == label.lower():
            return label
    return raw.strip()  # unrecognised — reported, not silently dropped


def main() -> int:
    ensure_dirs()
    if not HUMAN_LABELS_CSV.exists():
        print(f"FAIL: {HUMAN_LABELS_CSV} not found. Run `python -m src.judge` first.")
        return 1
    if not AUDIT_BASELINE_JSON.exists():
        print(f"FAIL: {AUDIT_BASELINE_JSON} not found. Run `python -m src.judge` first.")
        return 1

    audit = json.loads(AUDIT_BASELINE_JSON.read_text(encoding="utf-8"))
    model_labels = {
        (r["claim_id"], r["source_ref"]): r["label"]
        for r in audit
        if not r.get("source_unfetchable")
    }

    with HUMAN_LABELS_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    pairs, unlabeled, unmatched, invalid = [], [], [], []
    for row in rows:
        key = (row["claim_id"], row["source_ref"])
        human = normalize_label(row.get("human_label", ""))
        if human is None:
            unlabeled.append(row["claim_id"])
            continue
        if human not in LABEL_ORDER:
            invalid.append((row["claim_id"], human))
            continue
        if key not in model_labels:
            unmatched.append(row["claim_id"])
            continue
        pairs.append((row["claim_id"], human, model_labels[key]))

    print(f"human_labels.csv: {len(rows)} rows, {len(pairs)} labeled and joined")
    if unlabeled:
        print(f"  not yet labeled: {len(unlabeled)}  ({', '.join(unlabeled[:8])}{'...' if len(unlabeled) > 8 else ''})")
    if invalid:
        print(f"  unrecognised labels: {invalid}")
        print(f"  use exactly one of: {', '.join(LABEL_ORDER)}")
    if unmatched:
        print(f"  rows with no matching model judgement: {unmatched}")

    if not pairs:
        print()
        print("Nothing to compare yet — the human_label column is empty.")
        print(f"Fill it in at {HUMAN_LABELS_CSV}, then re-run.")
        return 1

    matches = sum(1 for _, h, m in pairs if h == m)
    agreement = matches / len(pairs)

    # confusion matrix: rows human, columns model
    confusion = Counter((h, m) for _, h, m in pairs)

    print()
    print(f"exact-match agreement: {agreement:.2f}  ({matches}/{len(pairs)})")
    print()
    print("confusion matrix (rows = human, columns = model):")
    width = max(len(l) for l in LABEL_ORDER) + 2
    print(" " * width + "".join(f"{l[:10]:>12s}" for l in LABEL_ORDER))
    for h in LABEL_ORDER:
        cells = "".join(f"{confusion.get((h, m), 0):>12d}" for m in LABEL_ORDER)
        print(f"{h:<{width}}{cells}")

    print()
    print("disagreements:")
    any_disagreement = False
    for claim_id, h, m in pairs:
        if h != m:
            any_disagreement = True
            print(f"  {claim_id:12s} human={h:22s} model={m}")
    if not any_disagreement:
        print("  none")

    payload = {
        "n_rows": len(rows),
        "n_labeled": len(pairs),
        "exact_match_agreement": round(agreement, 4),
        "matches": matches,
        "confusion": {f"{h} -> {m}": n for (h, m), n in sorted(confusion.items())},
        "disagreements": [
            {"claim_id": c, "human_label": h, "model_label": m} for c, h, m in pairs if h != m
        ],
        "threshold": AGREEMENT_THRESHOLD,
        "meets_threshold": agreement >= AGREEMENT_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    AGREEMENT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"wrote {AGREEMENT_JSON}")

    print()
    if agreement < AGREEMENT_THRESHOLD:
        print(f"ACCEPTANCE: FAIL — agreement {agreement:.2f} is below {AGREEMENT_THRESHOLD:.2f}.")
        print("Per SPEC.md section 6 step 3, the rubric needs a human fix, not a code fix.")
        print("Do not proceed to step 4.")
        return 1

    print(f"ACCEPTANCE: PASS — agreement {agreement:.2f} >= {AGREEMENT_THRESHOLD:.2f}. Step 4 is unblocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
