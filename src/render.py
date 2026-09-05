"""Step 8 — render the static report page.

    python -m src.render

Reads every artefact produced by steps 1-7 and renders one self-contained
out/site/index.html via Jinja. All data is inlined at render time: no fetch, no
JS framework, no server. Citation expansion is <details>/<summary> only.

Nothing numeric lives in the template. Every figure on the page traces back to
out/comparison.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import (
    AGREEMENT_JSON,
    ANSWERS_EBODA_JSON,
    AUDIT_BASELINE_JSON,
    AUDIT_EBODA_JSON,
    CHUNKS_JSONL,
    CLAIMS_JSONL,
    COMPARISON_JSON,
    COMPARISON_JSON as _CMP,
    PAGES_CACHE_DIR,
    PROBES_JSONL,
    SITE_CSS,
    SITE_DIR,
    SITE_INDEX_HTML,
    TEMPLATES_DIR,
    ensure_dirs,
)
from src.judge import LABEL_ORDER, best_label
from src.llm import MODEL_GENERATION, MODEL_JUDGE

# SPEC.md section 6 step 8: these six probes, side by side.
SIDE_BY_SIDE = ["A2", "A7", "B5", "H3", "A9", "B2"]

# Where the toolkit CSS is looked for, in order.
CSS_SOURCES = [
    Path("assets/eboda-web-toolkit.css"),
    Path(".mockups/toolkit/eboda-web-toolkit.css"),
    Path.home() / "Downloads" / "files" / "eboda-web-toolkit.css",
]

LABEL_SLUG = {
    "Supported": "supported",
    "Partially supported": "partially",
    "Unsupported": "unsupported",
    "Source unrelated": "unrelated",
}


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def page_titles() -> dict[str, str]:
    titles = {}
    for p in PAGES_CACHE_DIR.glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        titles[rec["url_normalized"]] = rec.get("title") or rec["url_normalized"]
    return titles


def eboda_sentences_for(probe_id, entry, eboda_by_claim, chunks) -> list[dict]:
    """One record per generated sentence, kept or dropped, with its judgement."""
    out = []
    for i, item in enumerate(entry["kept"] + entry["dropped"], start=1):
        claim_id = f"{probe_id}-e{i:02d}"
        kept = item in entry["kept"]
        judgements = eboda_by_claim.get(claim_id, [])
        out.append(
            {
                "n": i,
                "text": item["sentence"],
                "kept": kept,
                "label": item.get("label"),
                "slug": LABEL_SLUG.get(item.get("label"), "none"),
                "reason": item.get("reason"),
                "evidence_span": item.get("evidence_span"),
                "chunk_ids": [c for c in (item.get("chunk_ids") or []) if c in chunks],
                "sources": [
                    {
                        "chunk_id": j["source_ref"],
                        "title": chunks.get(j["source_ref"], {}).get("page_title") or j["source_ref"],
                        "url": chunks.get(j["source_ref"], {}).get("url", ""),
                        "label": j["label"],
                        "slug": LABEL_SLUG.get(j["label"], "none"),
                        "reason": j["reason"],
                        "evidence_span": j["evidence_span"],
                    }
                    for j in judgements
                ],
            }
        )
    return out


def build_context() -> dict:
    comparison = load_json(COMPARISON_JSON)
    probes = {p["id"]: p for p in load_jsonl(PROBES_JSONL)}
    claims = {c["claim_id"]: c for c in load_jsonl(CLAIMS_JSONL)}
    baseline = load_json(AUDIT_BASELINE_JSON)
    eboda_audit = load_json(AUDIT_EBODA_JSON) if AUDIT_EBODA_JSON.exists() else []
    answers = {a["probe_id"]: a for a in load_json(ANSWERS_EBODA_JSON)}
    chunks = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_JSONL)}
    titles = page_titles()

    # --- baseline claims, grouped by probe, with per-source judgements -------
    by_claim = defaultdict(list)
    for r in baseline:
        by_claim[r["claim_id"]].append(r)

    baseline_by_probe = defaultdict(list)
    for claim_id, records in by_claim.items():
        claim = claims[claim_id]
        labels = [r["label"] for r in records if not r["source_unfetchable"]]
        baseline_by_probe[claim["probe_id"]].append(
            {
                "claim_id": claim_id,
                "text": claim["text"],
                "marker_derived": claim["marker_derived"],
                "best": best_label(labels),
                "best_slug": LABEL_SLUG.get(best_label(labels), "none"),
                "sources": [
                    {
                        "url": r["source_ref"],
                        "title": titles.get(r["source_ref"], r["source_ref"]),
                        "label": r["label"],
                        "slug": LABEL_SLUG.get(r["label"], "none"),
                        "reason": r["reason"],
                        "evidence_span": r["evidence_span"],
                    }
                    for r in records
                ],
            }
        )
    for v in baseline_by_probe.values():
        v.sort(key=lambda c: c["claim_id"])

    # --- eboda sentences, grouped by probe ----------------------------------
    eboda_by_claim = defaultdict(list)
    for r in eboda_audit:
        eboda_by_claim[r["claim_id"]].append(r)

    def eboda_sentences(probe_id: str) -> list[dict]:
        entry = answers.get(probe_id)
        if not entry:
            return []
        out = []
        for i, item in enumerate(entry["kept"] + entry["dropped"], start=1):
            claim_id = f"{probe_id}-e{i:02d}"
            kept = item in entry["kept"]
            judgements = eboda_by_claim.get(claim_id, [])
            out.append(
                {
                    "text": item["sentence"],
                    "kept": kept,
                    "label": item.get("label"),
                    "slug": LABEL_SLUG.get(item.get("label"), "none"),
                    "reason": item.get("reason"),
                    "sources": [
                        {
                            "chunk_id": j["source_ref"],
                            "title": chunks.get(j["source_ref"], {}).get("page_title")
                            or j["source_ref"],
                            "url": chunks.get(j["source_ref"], {}).get("url", ""),
                            "label": j["label"],
                            "slug": LABEL_SLUG.get(j["label"], "none"),
                            "reason": j["reason"],
                            "evidence_span": j["evidence_span"],
                        }
                        for j in judgements
                    ],
                }
            )
        return out

    # --- chat threads for the Ask modal -------------------------------------
    def segment_response(text: str, probe_claims: list[dict]) -> list[dict]:
        """Split Adobe's transcript into claim / non-claim spans.

        Claims were extracted verbatim from this text in step 1, so each one can
        be found and wrapped. Anything between them (greetings, questions back to
        the user, the Sources block) renders as plain, unhighlightable text.
        """
        segments: list[dict] = []
        cursor = 0
        for claim in probe_claims:
            idx = text.find(claim["text"], cursor)
            if idx == -1:
                continue
            if idx > cursor:
                segments.append({"text": text[cursor:idx], "claim": None})
            segments.append({"text": claim["text"], "claim": claim})
            cursor = idx + len(claim["text"])
        if cursor < len(text):
            segments.append({"text": text[cursor:], "claim": None})
        return segments

    threads = []
    for pid in SIDE_BY_SIDE:
        probe = probes.get(pid)
        if not probe:
            continue

        # --- Adobe side: numbered sources, claims tagged with their numbers ---
        src_no = {u: i + 1 for i, u in enumerate(probe["cited_urls"])}
        adobe_sources = [
            {"n": n, "url": u, "title": titles.get(u, u)} for u, n in src_no.items()
        ]
        probe_claims = baseline_by_probe.get(pid, [])
        for c in probe_claims:
            c["marks"] = sorted(src_no[s["url"]] for s in c["sources"] if s["url"] in src_no)
            c["best_source"] = next(
                (s for s in c["sources"] if s["label"] == c["best"]), None
            )
        adobe_segments = segment_response(probe["response_text"] or "", probe_claims)

        # --- Eboda side: numbered chunks, one entry per generated sentence ----
        entry = answers.get(pid)
        thread_sentences, eboda_sources = [], []
        if entry:
            cited_ids: list[str] = []
            for item in entry["kept"] + entry["dropped"]:
                for cid in item.get("chunk_ids") or []:
                    if cid in chunks and cid not in cited_ids:
                        cited_ids.append(cid)
            chunk_no = {cid: i + 1 for i, cid in enumerate(cited_ids)}
            eboda_sources = [
                {
                    "n": n,
                    "chunk_id": cid,
                    "title": chunks[cid].get("page_title") or cid,
                    "url": chunks[cid].get("url", ""),
                    "heading": chunks[cid].get("heading"),
                }
                for cid, n in chunk_no.items()
            ]
            for s in eboda_sentences_for(pid, entry, eboda_by_claim, chunks):
                s["marks"] = sorted(chunk_no[c] for c in s["chunk_ids"] if c in chunk_no)
                thread_sentences.append(s)

        threads.append(
            {
                "id": pid,
                "category": probe["category"],
                "prompt": probe["prompt"],
                "verdict": probe["verdict"],
                "failure_mode": probe["failure_mode"],
                "adobe_segments": adobe_segments,
                "adobe_sources": adobe_sources,
                "adobe_claims": probe_claims,
                "eboda_present": entry is not None,
                "eboda_display_only": bool(entry and entry.get("display_only")),
                "eboda_abstained": entry["abstained"] if entry else None,
                "eboda_final": entry["final_text"] if entry else None,
                "eboda_reason": entry["reason"] if entry else None,
                "eboda_sentences": thread_sentences,
                "eboda_sources": eboda_sources,
            }
        )

    # --- the six side-by-side probes ----------------------------------------
    pairs = []
    for pid in SIDE_BY_SIDE:
        probe = probes.get(pid)
        if not probe:
            continue
        entry = answers.get(pid)
        pairs.append(
            {
                "id": pid,
                "category": probe["category"],
                "prompt": probe["prompt"],
                "verdict": probe["verdict"],
                "failure_mode": probe["failure_mode"],
                "persona": probe["persona"],
                "baseline_text": probe["response_text"],
                "baseline_claims": baseline_by_probe.get(pid, []),
                "baseline_cited": [
                    {"url": u, "title": titles.get(u, u)} for u in probe["cited_urls"]
                ],
                "eboda_present": entry is not None,
                "eboda_abstained": entry["abstained"] if entry else None,
                "eboda_reason": entry["reason"] if entry else None,
                "eboda_final": entry["final_text"] if entry else None,
                "eboda_sentences": eboda_sentences(pid) if entry else [],
            }
        )

    # --- full audit table ----------------------------------------------------
    table = []
    for r in baseline:
        table.append(
            {
                "claim_id": r["claim_id"],
                "probe_id": r["probe_id"],
                "system": "Adobe",
                "claim": claims[r["claim_id"]]["text"],
                "source": titles.get(r["source_ref"], r["source_ref"]),
                "source_url": r["source_ref"],
                "label": r["label"] or "source_unfetchable",
                "slug": LABEL_SLUG.get(r["label"], "none"),
                "reason": r["reason"],
            }
        )
    for r in eboda_audit:
        chunk = chunks.get(r["source_ref"], {})
        table.append(
            {
                "claim_id": r["claim_id"],
                "probe_id": r["probe_id"],
                "system": "Eboda",
                "claim": next(
                    (s["text"] for s in eboda_sentences(r["probe_id"])
                     if s["sources"] and s["sources"][0]["chunk_id"] == r["source_ref"]),
                    "",
                ),
                "source": chunk.get("page_title") or r["source_ref"],
                "source_url": chunk.get("url", ""),
                "label": r["label"],
                "slug": LABEL_SLUG.get(r["label"], "none"),
                "reason": r["reason"],
            }
        )

    # --- derived figures, all from comparison.json --------------------------
    b, e = comparison["baseline"], comparison["eboda"]
    dates = [p.get("judged_at") for p in baseline if p.get("judged_at")]

    # Sample size comes from agreement.json, never from a literal in the template.
    agreement = load_json(AGREEMENT_JSON) if AGREEMENT_JSON.exists() else {}
    n_human_labels = agreement.get("n_labeled", 0)

    return {
        "comparison": comparison,
        "baseline": b,
        "eboda": e,
        "gate": comparison["gate"],
        "pct": lambda x: f"{x * 100:.0f}%",
        "pairs": pairs,
        "table": table,
        "label_order": LABEL_ORDER,
        "label_slug": LABEL_SLUG,
        "n_probes": len(probes),
        "n_probes_cited": sum(1 for p in probes.values() if p["cited_urls"]),
        "n_pages": len(titles),
        "n_chunks": len(chunks),
        "judge_model": MODEL_JUDGE,
        "generation_model": MODEL_GENERATION,
        "measured_at": (min(dates)[:10] if dates else comparison["generated_at"][:10]),
        "generated_at": comparison["generated_at"],
        "marker_derived_pct": round(
            100 * sum(1 for c in claims.values() if c["marker_derived"]) / max(1, len(claims)), 1
        ),
        "n_claims_uncited": sum(1 for c in claims.values() if not c["cited_urls"]),
        "n_claims_total": len(claims),
        "threads": threads,
        "n_human_labels": n_human_labels,
        "human_agreement_display": f"{comparison['human_agreement']:.2f}",
    }


def copy_css() -> Path | None:
    for candidate in CSS_SOURCES:
        if candidate.exists():
            shutil.copyfile(candidate, SITE_CSS)
            return candidate
    return None


def main() -> int:
    ensure_dirs()
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    for path in (COMPARISON_JSON, AUDIT_BASELINE_JSON, ANSWERS_EBODA_JSON):
        if not path.exists():
            print(f"FAIL: {path} not found. Run the earlier steps first.")
            return 1

    context = build_context()

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    html = env.get_template("index.html.j2").render(**context)
    SITE_INDEX_HTML.write_text(html, encoding="utf-8")

    css_from = copy_css()
    print(f"wrote {SITE_INDEX_HTML}  ({len(html):,} bytes)")
    if css_from:
        print(f"copied CSS from {css_from} -> {SITE_CSS}")
    else:
        print(f"WARNING: eboda-web-toolkit.css not found in {[str(c) for c in CSS_SOURCES]}")

    # --- acceptance check (SPEC.md section 6, step 8) -----------------------
    failures = []
    if not SITE_CSS.exists():
        failures.append("eboda-web-toolkit.css missing from out/site/")
    for banned in ("<script", "fetch(", "XMLHttpRequest", "cdn.", "https://unpkg"):
        if banned in html:
            failures.append(f"page contains {banned!r} — must be static")
    if "index.html.j2" in html:
        failures.append("template path leaked into output")

    b = context["baseline"]
    e = context["eboda"]
    print()
    print(f"probes={context['n_probes']}  side-by-side={len(context['pairs'])}  "
          f"audit rows={len(context['table'])}  chunks={context['n_chunks']}")
    print(f"baseline grounding_rate={b['grounding_rate']}  eboda={e['grounding_rate']}  "
          f"human_agreement={context['comparison']['human_agreement']}")
    print()
    if failures:
        print("ACCEPTANCE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ACCEPTANCE: PASS — static, no script tags, no network calls, CSS copied.")
    print(f"Open: file:///{SITE_INDEX_HTML.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
