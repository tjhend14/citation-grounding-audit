"""Step 5 — retrieval-grounded generation with sentence-level citations.

    python -m src.concierge            # runs the three acceptance queries
    python -m src.concierge "question"

    answer(prompt) -> dict

Retrieves with BM25, formats the numbered sources block, calls the SPEC.md
section 5.2 generation prompt, and returns the parsed JSON plus the chunks it
retrieved. The step 6 verification gate is layered on top of this.
"""

from __future__ import annotations

import json
import re
import sys

from src.config import ANSWERS_EBODA_JSON, PROBES_JSONL, ensure_dirs
from src.judge import judge_pair
from src.llm import MODEL_GENERATION, MODEL_JUDGE, call
from src.retrieve import DEFAULT_K, Chunk, search

# --- prompt (SPEC.md section 5.2) --------------------------------------------
# VERBATIM. Do not paraphrase, do not add few-shot examples.
GENERATION_SYSTEM = "You are a product assistant. You respond with JSON only."

GENERATION_USER_TEMPLATE = """Answer the question using only the numbered sources below.

<sources>
{sources}
</sources>

<question>
{prompt}
</question>

Rules:
- Every sentence stating a fact must cite at least one source ID.
- Do not state anything the sources do not say. You have no knowledge outside these sources.
- If the sources do not answer the question, set abstained to true, say plainly that you don't have
  it documented, name the closest thing you do have, and offer a handoff. Do not guess.
- Prices, terms, cancellation rules, and fees must come from a source or be omitted entirely.
  Never approximate them.
- Two to five sentences. No marketing language.

Respond with JSON only, no preamble:
{{"answer": [{{"sentence": "...", "chunk_ids": ["c0000"]}}], "abstained": true or false}}"""

# The three questions SPEC.md section 6 step 5 requires be run by hand.
ACCEPTANCE_QUERIES = [
    "How much is Photoshop per month?",
    "What are the minimum system requirements to run Premiere Pro on Windows?",
    "Does Eboda offer a discount for nonprofit organisations in Canada?",
]


def format_sources(chunks: list[Chunk]) -> str:
    """One line per chunk: [id] heading — text."""
    return "\n".join(f"[{c.chunk_id}] {c.label()} — {c.text}" for c in chunks)


def answer(prompt: str, k: int = DEFAULT_K) -> dict:
    """Retrieve, generate, and return the parsed answer plus its chunks."""
    chunks = search(prompt, k=k)

    if not chunks:
        # Nothing retrieved at all — abstain without spending a call.
        return {
            "prompt": prompt,
            "answer": [],
            "abstained": True,
            "chunks": [],
            "no_retrieval": True,
        }

    user = GENERATION_USER_TEMPLATE.format(sources=format_sources(chunks), prompt=prompt)
    parsed = call(GENERATION_SYSTEM, user, MODEL_GENERATION)

    sentences = parsed.get("answer") or []
    return {
        "prompt": prompt,
        "answer": sentences,
        "abstained": bool(parsed.get("abstained")),
        "chunks": chunks,
        "no_retrieval": False,
    }


# --- step 6: the verification gate -------------------------------------------
# Labels the judge returns that mean "keep this sentence".
KEEP_LABELS = {"Supported", "Partially supported"}

# A dropped sentence carrying any of these means the whole answer is abstained.
# A partial pricing answer is worse than no pricing answer — this rule is the
# product (SPEC.md section 6, step 6).
_NUMERIC_RE = re.compile(r"[$£€¥₹]|%|\d")


# --- abstain copy ------------------------------------------------------------
# This is the A9/A10 side-by-side on the final page, so the wording carries
# weight. Three jobs, in order: say plainly it isn't documented, name the
# closest thing actually retrieved, offer a handoff. No apology spiral, no
# marketing, no implied promise about what support will say.
ABSTAIN_WITH_PAGE = (
    "I don't have that documented in the material I can cite, so I'm not going to guess. "
    "The closest page I found is “{title}”, which covers related ground but doesn't "
    "answer this. "
    "If it's useful, I can put you in touch with someone who can confirm it directly."
)

ABSTAIN_NO_PAGE = (
    "I don't have anything documented that answers this, so I'm not going to guess. "
    "Nothing in the material I can cite covers it. "
    "If it's useful, I can put you in touch with someone who can confirm it directly."
)


def contains_number(sentence: str) -> bool:
    """A currency symbol, a percentage, or any digit."""
    return bool(_NUMERIC_RE.search(sentence))


def abstain_copy(chunks: list[Chunk]) -> str:
    """The fallback answer. Names the closest retrieved page by title."""
    if not chunks:
        return ABSTAIN_NO_PAGE
    title = chunks[0].page_title or chunks[0].label() or "an untitled page"
    return ABSTAIN_WITH_PAGE.format(title=title)


def verify(result: dict) -> dict:
    """Judge each generated sentence against the text of the chunks it cites.

    Uses the SPEC.md section 5.1 judge unchanged — the same prompt, the same
    model, the same cache that scored the baseline. The concierge is held to
    exactly the standard Adobe was held to.
    """
    by_id = {c.chunk_id: c for c in result["chunks"]}
    kept, dropped = [], []

    for item in result["answer"]:
        sentence = (item.get("sentence") or "").strip()
        if not sentence:
            continue
        chunk_ids = [cid for cid in (item.get("chunk_ids") or []) if cid in by_id]

        if not chunk_ids:
            # Section 5.2 requires every factual sentence to cite a source. One
            # that cites nothing (or cites an id we did not retrieve) cannot be
            # verified, so it cannot be kept.
            dropped.append({
                "sentence": sentence,
                "chunk_ids": item.get("chunk_ids") or [],
                "label": None,
                "reason": "no citable source id",
            })
            continue

        source_text = "\n\n".join(by_id[cid].text for cid in chunk_ids)
        judgement = judge_pair(sentence, source_text)
        label = judgement.get("label")
        record = {
            "sentence": sentence,
            "chunk_ids": chunk_ids,
            "label": label,
            "reason": judgement.get("reason"),
            "evidence_span": judgement.get("evidence_span"),
        }
        (kept if label in KEEP_LABELS else dropped).append(record)

    # --- the abstain rules --------------------------------------------------
    abstained, reason = False, "answer passed the gate"

    numeric_drops = [d for d in dropped if contains_number(d["sentence"])]
    if numeric_drops:
        abstained = True
        reason = (
            f"abstained: {len(numeric_drops)} dropped sentence(s) contained a number, "
            "currency symbol or percentage"
        )
    elif dropped and len(dropped) > len(kept):
        abstained = True
        reason = f"abstained: {len(dropped)} of {len(kept) + len(dropped)} sentences dropped"
    elif not kept:
        abstained = True
        reason = "abstained: no sentence survived verification"

    if result["abstained"]:
        abstained = True
        reason = "abstained: the generator declined to answer from these sources"

    return {
        **result,
        "kept": kept,
        "dropped": dropped,
        "abstained": abstained,
        "reason": reason,
        "final_text": (
            abstain_copy(result["chunks"]) if abstained
            else " ".join(k["sentence"] for k in kept)
        ),
        "judge_model": MODEL_JUDGE,
    }


def answer_verified(prompt: str, k: int = DEFAULT_K) -> dict:
    """Generate, then run the step 6 verification gate."""
    return verify(answer(prompt, k=k))


def print_answer(result: dict) -> None:
    print(f'Q: {result["prompt"]}')
    print(f'   abstained={result["abstained"]}')
    if result.get("no_retrieval"):
        print("   (BM25 returned nothing above zero score — abstained without a model call)")
    print("   retrieved:")
    for c in result["chunks"]:
        print(f"     [{c.chunk_id}] {c.score:6.2f}  {c.page_title}")
    print("   answer:")
    for s in result["answer"]:
        cites = ", ".join(s.get("chunk_ids") or []) or "NO CITATION"
        print(f"     - {s.get('sentence', '')}")
        print(f"       cites: {cites}")
    if not result["answer"]:
        print("     (none)")


def run_gate_over_probes() -> int:
    """Step 6 — run every cited probe through generation + the gate."""
    ensure_dirs()
    probes = [json.loads(l) for l in PROBES_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    cited = [p for p in probes if p["cited_urls"]]
    print(f"generation={MODEL_GENERATION}  gate judge={MODEL_JUDGE}  probes with citations={len(cited)}")
    print()

    records, generated, dropped_total, abstained_total = [], 0, 0, 0
    for probe in cited:
        result = answer_verified(probe["prompt"])
        generated += len(result["kept"]) + len(result["dropped"])
        dropped_total += len(result["dropped"])
        abstained_total += bool(result["abstained"])

        flag = "ABSTAIN" if result["abstained"] else "answer "
        print(f"  {probe['id']:4s} {flag} kept={len(result['kept'])} dropped={len(result['dropped'])}"
              f"  {result['reason']}")
        for d in result["dropped"]:
            num = " [NUMERIC]" if contains_number(d["sentence"]) else ""
            print(f"        dropped{num}: {d['sentence'][:88]}")
            print(f"          label={d['label']} — {d['reason']}")

        records.append({
            "probe_id": probe["id"],
            "prompt": probe["prompt"],
            "kept": result["kept"],
            "dropped": result["dropped"],
            "abstained": result["abstained"],
            "reason": result["reason"],
            "final_text": result["final_text"],
            "retrieved_chunk_ids": [c.chunk_id for c in result["chunks"]],
            "judge_model": result["judge_model"],
        })

    ANSWERS_EBODA_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print()
    print(f"sentences_generated={generated}")
    print(f"sentences_dropped={dropped_total}")
    print(f"answers_abstained={abstained_total}")
    print()
    if dropped_total == 0:
        print("WARNING: sentences_dropped == 0 across all probes.")
        print("SPEC.md section 6 step 6 says to verify the gate is actually running.")
        print("Run `python -m src.concierge --prove-gate` for a direct demonstration.")
    print(f"wrote {ANSWERS_EBODA_JSON}")
    return 0


def prove_gate() -> int:
    """Demonstrate the gate rejecting text, independent of what the model happens
    to generate. Feeds fabricated sentences through the same verify() path."""
    chunks = search("photoshop price per month", k=2)
    fake = {
        "prompt": "(gate proof)",
        "chunks": chunks,
        "abstained": False,
        "no_retrieval": False,
        "answer": [
            {"sentence": "Photoshop costs US$22.99/mo when billed monthly on an annual plan.",
             "chunk_ids": [chunks[0].chunk_id]},
            {"sentence": "Photoshop costs US$4.00 per month for nonprofits in Canada.",
             "chunk_ids": [chunks[0].chunk_id]},
            {"sentence": "Adobe was founded in a submarine.", "chunk_ids": [chunks[0].chunk_id]},
            {"sentence": "This sentence cites nothing at all.", "chunk_ids": []},
        ],
    }
    print("Feeding 4 sentences through the SAME verify() the probes use:")
    print(f"  source chunk: [{chunks[0].chunk_id}] {chunks[0].page_title}")
    print()
    out = verify(fake)
    for k in out["kept"]:
        print(f"  KEPT    label={k['label']:20s} {k['sentence'][:70]}")
    for d in out["dropped"]:
        num = " [NUMERIC]" if contains_number(d["sentence"]) else ""
        print(f"  DROPPED label={str(d['label']):20s}{num} {d['sentence'][:70]}")
    print()
    print(f"  abstained = {out['abstained']}")
    print(f"  reason    = {out['reason']}")
    print()
    print("  final_text served to the user:")
    print(f"    {out['final_text']}")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--prove-gate":
        return prove_gate()
    if argv and argv[0] == "--acceptance":
        print(f"generation model = {MODEL_GENERATION}")
        print("=" * 78)
        for query in ACCEPTANCE_QUERIES:
            print()
            print_answer(answer(query))
            print("-" * 78)
        return 0
    if argv:
        result = answer_verified(" ".join(argv))
        print_answer(result)
        print(f"   gate: {result['reason']}")
        print(f"   final: {result['final_text']}")
        return 0
    return run_gate_over_probes()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
