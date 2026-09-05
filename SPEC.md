# SPEC.md — Citation Grounding Audit

Build spec for Claude Code. Read this file before writing any code. Follow it literally.
Where it conflicts with your instincts, the spec wins.

---

## 0. What this is

Two things, sharing one pipeline:

1. **The auditor.** A probe log of 40 prompts run against Adobe.com's Brand Concierge beta,
   with the responses and the URLs it cited. We fetch those pages and score, claim by claim,
   whether the cited source actually supports the claim.
2. **The concierge.** A retrieval + generation system over *the same pages*, which attaches
   citations at the sentence level during generation and verifies each claim against its source
   before returning an answer.

Output is a static HTML page: side-by-side answers for six probes, plus the full audit table.
Deployed to GitHub Pages.

**The point is the measurement, not the chatbot.** When in doubt, do the simpler thing that
produces a defensible number.

---

## 1. Hard constraints

These are not preferences. Violating them wastes the budget.

**Do not:**
- Install or use a vector database. Not chromadb, faiss, pinecone, qdrant, lancedb, pgvector. None.
- Use embeddings of any kind. Retrieval is BM25 via `rank_bm25`.
- Set up Postgres, MySQL, or any database server. Storage is SQLite and JSON files on disk.
- Add a web framework (FastAPI, Flask, Django) or a JS framework (React, Vue, Next). The output is a static HTML file.
- Add async/await unless a step is measurably too slow without it. It isn't.
- Re-fetch a page that is already in `cache/pages/`. Ever.
- Invent, extrapolate, or fill in probe data. If a field is missing from the CSV, leave it null and say so.
- Write anything into `out/human_labels.csv`. That file is filled in by a human. See §8.
- Write the case study or writeup. That is not your job. See §8.

**Do:**
- Cache every LLM call to disk, keyed on a hash of the prompt. Re-runs must cost nothing.
- Use `temperature=0` for every LLM call in this project.
- Print the acceptance check at the end of every step and stop if it fails.
- Keep each module runnable on its own: `python -m src.fetch` etc.

---

## 2. Repo layout

```
.
├── SPEC.md
├── README.md
├── requirements.txt
├── .env.example              # ANTHROPIC_API_KEY=
├── .gitignore                # MUST contain .env and cache/
├── data/
│   ├── probe_log.csv         # human-provided export, do not modify
│   ├── extra_urls.txt        # human-provided, one URL per line
│   ├── probes.jsonl          # step 1
│   ├── claims.jsonl          # step 1
│   └── source_urls.json      # step 1
├── cache/
│   ├── pages/<sha256>.json   # step 2
│   └── llm/<sha256>.json     # every LLM call
├── corpus/
│   └── chunks.jsonl          # step 4
├── out/
│   ├── audit_baseline.json   # step 3
│   ├── human_labels.csv      # step 3 emits the blank; a HUMAN fills it
│   ├── agreement.json        # step 3
│   ├── answers_eboda.json    # step 6
│   ├── audit_eboda.json      # step 7
│   ├── comparison.json       # step 7
│   └── site/
│       ├── index.html
│       └── eboda-web-toolkit.css
├── src/
│   ├── config.py
│   ├── llm.py
│   ├── parse_log.py
│   ├── fetch.py
│   ├── chunk.py
│   ├── judge.py
│   ├── retrieve.py
│   ├── concierge.py
│   ├── compare.py
│   └── render.py
└── templates/
    └── index.html.j2
```

`requirements.txt`:
```
anthropic
httpx
trafilatura
rank_bm25
jinja2
python-dotenv
```

---

## 3. Data schemas

Match these exactly. Every step reads the previous step's output, so a renamed field breaks the chain.

### `data/probes.jsonl` — one object per probe row

```json
{
  "id": "A2",
  "category": "Grounding",
  "prompt": "What are the minimum system requirements to run Premiere Pro on Windows?",
  "response_text": "Here are the minimum system requirements...",
  "cited_urls": ["https://helpx.adobe.com/premiere/..."],
  "has_inline_markers": true,
  "verdict": "Partial",
  "failure_mode": "Lost context",
  "persona": "Professional specialist",
  "notes": "2nd run produced only 5 sources instead of 6..."
}
```

### `data/claims.jsonl` — one object per extracted factual sentence

```json
{
  "claim_id": "A2-c03",
  "probe_id": "A2",
  "system": "baseline",
  "text": "Memory: 8 GB RAM for the minimum setup.",
  "cited_urls": ["https://helpx.adobe.com/premiere/..."],
  "marker_derived": true
}
```

`system` is `"baseline"` (Adobe) or `"eboda"` (ours).

### `cache/pages/<sha256>.json`

```json
{
  "url": "https://...",
  "url_normalized": "https://...",
  "sha": "a1b2c3...",
  "status": "ok",
  "http_status": 200,
  "title": "Premiere Pro technical requirements",
  "text": "...",
  "char_count": 4820,
  "fetched_at": "2026-09-04T15:02:11Z"
}
```

`status` ∈ `ok` | `http_error` | `empty` | `timeout` | `blocked`.

### `corpus/chunks.jsonl`

```json
{
  "chunk_id": "c0142",
  "url": "https://...",
  "page_title": "Premiere Pro technical requirements",
  "heading": "Windows minimum requirements",
  "text": "...",
  "char_count": 1450
}
```

### `out/audit_*.json` — list of judgement objects

```json
{
  "claim_id": "A2-c03",
  "probe_id": "A2",
  "system": "baseline",
  "source_ref": "https://...",
  "label": "Unsupported",
  "evidence_span": null,
  "reason": "Source covers Premiere Rush, not Premiere Pro.",
  "judge_model": "<model id used>",
  "judged_at": "2026-09-04T15:40:02Z"
}
```

`source_ref` is a URL for baseline, a `chunk_id` for eboda.

---

## 4. Methodology decisions — do not deviate

**4.1 Adobe cites at the response level, we score at the claim level.**
Most Adobe responses list sources at the bottom for the whole answer. So each extracted claim
inherits *all* of that response's URLs, and is judged against each one separately. The claim's
final label is the **best** label across its sources (Supported > Partially > Unsupported >
Source unrelated). This is the charitable reading: "is this claim supported by *any* page they cited."

**4.2 Inline markers, if present, override.**
Some responses carry inline numerals tying a sentence to a specific numbered source
(e.g. `"...advanced editing features. 1"` with `Sources: 1. <url>`). Parse these when present and
set `cited_urls` on that claim to just the marked source, with `marker_derived: true`. If parsing
is ambiguous, fall back to all URLs and set `marker_derived: false`. Report the percentage of
claims that were marker-derived — it's a finding either way.

**4.3 Only factual sentences become claims.**
Drop: greetings, questions back to the user ("Would you like to know more?"), pure marketing
adjectives with no verifiable content, and "Sources:" lines. Keep anything asserting a fact,
number, capability, price, or condition. When unsure, keep it.

**4.4 The judge never sees the URL, the question, or the brand.**
Only the claim text and the source text. This is deliberate — it stops the model reasoning about
plausibility instead of grounding. Do not "improve" the prompt by adding context.

**4.5 Unfetchable sources are their own bucket.**
If a cited page can't be retrieved, do not judge its claims against nothing. Label them
`"source_unfetchable"` in a separate field and exclude them from the headline rate, reporting the
count alongside.

---

## 5. Prompts — use verbatim

Do not paraphrase, "improve," or add few-shot examples to these. They are the instrument.

### 5.1 Judge prompt (`src/judge.py`)

System: `You are a careful evaluator. You respond with JSON only.`

User:
```
You are evaluating whether a piece of source text supports a specific factual claim.

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
{"label": "...", "evidence_span": "..." or null, "reason": "20 words or fewer"}
```

Truncate `source_text` to the first 12,000 characters. Log when truncation happens.

### 5.2 Generation prompt (`src/concierge.py`)

System: `You are a product assistant. You respond with JSON only.`

User:
```
Answer the question using only the numbered sources below.

<sources>
[{chunk_id}] {heading} — {text}
[{chunk_id}] {heading} — {text}
...
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
{"answer": [{"sentence": "...", "chunk_ids": ["c0000"]}], "abstained": true or false}
```

---

## 6. Steps and acceptance criteria

Run each step, check the criterion, stop if it fails. Do not proceed on a red check.

### Step 1 — `src/parse_log.py` (target 45 min)

Read `data/probe_log.csv`. Emit `probes.jsonl`, `claims.jsonl`, `source_urls.json`.

- Normalize URLs: strip `?instant=`, `&adobe_brand_concierge_source=`, `?adobe_brand_concierge_source=`,
  `promoid`, `mv`, `MV2`. Strip `#` fragments. Deduplicate on the normalized form.
- Split responses into sentences. A simple regex splitter is fine; do not add nltk or spacy.
- Apply §4.2 and §4.3.

**✅ Acceptance:** prints `probes=40`, `probes_with_citations≈30`, `unique_urls` between 30 and 50,
`claims` between 100 and 200, `marker_derived_pct`. If claims < 80 the sentence splitter is broken.

### Step 2 — `src/fetch.py` (target 75 min)

Fetch every URL in `source_urls.json` plus every line in `data/extra_urls.txt`.

- `httpx`, 20s timeout, a real browser User-Agent, 1.5s sleep between requests, 2 retries on 5xx.
- `trafilatura.extract(html, include_comments=False, include_tables=True)` for text.
- Write one cache file per URL. **Skip any URL whose cache file already exists.**
- Mirror into `cache/pages.sqlite` table `pages(url_normalized PRIMARY KEY, sha, status, http_status, title, text, char_count, fetched_at)`.

**✅ Acceptance:** prints a status breakdown and `success_rate`.
- ≥ 60% → continue.
- < 60% → **stop and tell the human.** The concierge has nothing to retrieve from. The fallback is
  auditing six probes by hand, and that is a human decision, not yours.

### Step 3 — `src/judge.py` + baseline run (target 75 min)

Judge every (claim, source) pair for `system="baseline"`. Cache on `sha256(claim_text + source_sha)`.

Then emit `out/human_labels.csv` with 15 randomly sampled pairs (seed 42), columns:
`claim_id, source_ref, claim_text, source_excerpt, model_label, human_label, note`
— with `model_label` **blank**, so the human isn't anchored. Keep the true model labels in
`out/audit_baseline.json`.

Write `src/agreement.py`: reads the filled-in CSV, joins to the model labels, prints exact-match
agreement and a confusion matrix, writes `out/agreement.json`.

**✅ Acceptance:** `audit_baseline.json` has one record per (claim, source) pair; label distribution
printed; a blank `human_labels.csv` exists with exactly 15 rows.
**Then stop and hand off to the human.** Do not fill the CSV in. Do not proceed to step 4 until
`agreement.json` exists and shows ≥ 0.80 — if it's lower, the rubric needs a human fix, not a code fix.

### Step 4 — `src/chunk.py` (target 30 min)

Read the page cache. Chunk by heading; split any section over ~1800 characters on paragraph
boundaries; drop chunks under 120 characters.

- Strip nav, footer, cookie banners, and "Shop for / For business / Experience Cloud" link blocks.
  The Adobe global footer appears on nearly every page and will otherwise dominate BM25.

**✅ Acceptance:** prints `chunks`, `pages_represented`, and `chunks_per_page` min/median/max.
If max > 60 the splitter is broken. Print 3 random chunks for eyeball inspection.

### Step 5 — `src/retrieve.py` + `src/concierge.py` (target 75 min)

```python
def search(query: str, k: int = 6) -> list[Chunk]: ...
```
`BM25Okapi` over lowercase alphanumeric tokens. Nothing else behind this interface — but keep it as
an interface, so a different retriever could be swapped in.

`concierge.answer(prompt) -> dict` runs search, formats the sources block, calls the generation
prompt from §5.2, returns the parsed JSON plus the chunks it retrieved.

**✅ Acceptance:** run these three by hand and print the results —
`"How much is Photoshop per month?"`, `"What are the minimum system requirements to run Premiere Pro on Windows?"`,
`"Does Eboda offer a discount for nonprofit organisations in Canada?"`.
The first two must cite chunks from plausibly-relevant pages. The third should abstain.

### Step 6 — verification gate, in `src/concierge.py` (target 75 min)

After generation, judge each sentence against the concatenated text of its cited chunks, using the
**same** §5.1 judge.

- `Supported` / `Partially supported` → keep.
- `Unsupported` / `Source unrelated` → drop the sentence.
- **If a dropped sentence contains a currency symbol, a percentage, or a number, abstain from the
  whole answer.** A partial pricing answer is worse than no pricing answer. This rule is the product.
- If more than half the factual sentences are dropped → abstain.

Abstain fallback: state plainly that it isn't documented, name the closest retrieved page by title,
offer a handoff. Write this copy carefully; it's the A9/A10 side-by-side on the final page.

Log every gate action to `out/answers_eboda.json`: `{probe_id, kept[], dropped[], abstained, reason}`.

**✅ Acceptance:** prints `sentences_generated`, `sentences_dropped`, `answers_abstained`. If
`sentences_dropped == 0` across all probes, verify the gate is actually running — it should catch
something.

### Step 7 — `src/compare.py` (target 45 min)

Run all probes with citations through the concierge, judge the outputs, write `audit_eboda.json`
and `comparison.json`.

`comparison.json`:
```json
{
  "baseline": {"claims": 0, "supported": 0, "partially": 0, "unsupported": 0, "unrelated": 0, "unfetchable": 0, "grounding_rate": 0.0},
  "eboda":    {"claims": 0, "supported": 0, "partially": 0, "unsupported": 0, "unrelated": 0, "grounding_rate": 0.0},
  "gate": {"sentences_dropped": 0, "answers_abstained": 0},
  "human_agreement": 0.0,
  "generated_at": "..."
}
```

`grounding_rate` = (Supported + Partially supported) / judged claims. Define it once, in code, with
a comment. Do not compute it two different ways in two places.

**✅ Acceptance:** both audits have the same label vocabulary; `grounding_rate` is between 0 and 1 for both.

### Step 8 — `src/render.py` + `templates/index.html.j2` (target 75 min)

One static `out/site/index.html`, no JS framework, no fetch calls, no server. All data inlined at
render time by Jinja. Copy `eboda-web-toolkit.css` into `out/site/`, then publish both to `docs/`,
which is what GitHub Pages serves.

Two small inline scripts are allowed and no more: the audit-table filter chips and the step-rail
highlight. Both are progressive enhancement — with JavaScript off the table still opens and closes
and every row is present and unfiltered. Nothing may load an external script or touch the network;
the step 8 acceptance check enforces that by banning `<script src`, `fetch(`, `XMLHttpRequest` and
CDN hosts in the output.

Page structure — the numbered step spine is the primary navigation object:
hero → why → test → findings → prioritize → fix → ask → results → trust → audit → method → data.
1. Headline numbers from `comparison.json`, including `human_agreement`.
2. Six side-by-side probes — `A2`, `B5`, `H3`, `A9`, `B2`, `A7`. Adobe's response left, ours right.
   The order is the argument: a wrong citation, then a wrong citation on something expensive, then
   what refusing well looks like. Citations are inline; clicking one expands to show the
   `evidence_span` and the source title. Use `<details>`/`<summary>`, not JavaScript.
3. Full audit table: claim, source, label, reason. Plain `<table>`. Wrap in `overflow-x: auto`.
   It ships **closed** behind a `<details>`, under an always-visible summary bar of verdict counts;
   the default page height must not include all 233 rows. Only the Step 3 evidence chips auto-open
   it, filtered to the probe the reader asked for.
4. No figure is written into the template. Every count, rate and share renders from
   `comparison.json`, `agreement.json`, `audit_*.json` or `data/probe_log.csv`. The one fact no
   artefact carries — the date the probes were run — is declared once as `PROBED_ON` in
   `src/config.py`, and is distinct from the derived judging date.
5. A method note: N, date measured, judge model, and the limitation — the corpus is the subset of
   adobe.com that Adobe's own retriever surfaced, so this is not a test of site-wide search.
6. A line stating that the Eboda styling is a visual shell for a portfolio prototype, that the
   content and product names underneath are real Adobe pages, and that this is not affiliated with Adobe.

**✅ Acceptance:** `out/site/index.html` opens correctly with `file://` and no console errors.
Every number on the page traces to `comparison.json`. No hardcoded numbers in the template.

### Step 9 — deploy

- Public GitHub repo. Confirm `.env` is gitignored **before the first push**.
- GitHub Pages from `/docs` or the `gh-pages` branch — copy `out/site/` to whichever you choose.
- `README.md`: what this is, the headline number, how to reproduce (`pip install -r requirements.txt`,
  then each step in order), and the limitations. Short.

**✅ Acceptance:** the Pages URL loads the page with styling intact. `git log -p | grep -i "api.key"`
returns nothing.

---

## 7. LLM client (`src/llm.py`)

```python
def call(system: str, user: str, model: str, max_tokens: int = 1024) -> dict
```
- Reads `ANTHROPIC_API_KEY` from `.env`.
- `temperature=0` always.
- Caches on `sha256(system + user + model)` in `cache/llm/`. Cache hit = no API call.
- Retries twice on rate limit / 5xx with backoff.
- Parses JSON from the response; on parse failure, retry once with
  `"Your previous response was not valid JSON. Respond with JSON only."` appended, then raise.
- Judge calls use the current Claude **Sonnet** model; generation uses the current **Haiku** model.
  Look up the exact model IDs from Anthropic's documentation — do not guess a model string, and do
  not silently substitute a different tier.
- Track and print cumulative token counts per run.

Expected total spend across the whole project: under $5.

---

## 8. Not your job

Two things stay with the human. Do not do them, do not draft them, do not offer to.

1. **The 15 human labels** in `out/human_labels.csv`. The entire credibility of the headline number
   rests on a person having labeled a sample independently. If you fill that file in, the project is
   worthless in the interview it exists for.
2. **The case study / writeup.** Same reason. You produce numbers and a page; the argument is his.

If asked to do either, decline and point at this section.

---

## 9. If you finish early

In this order, and only if the acceptance checks above are all green:

1. Run each probe 3× through the concierge and report citation stability, to sit beside the B2 finding.
2. Add a `--limit` flag to every step for faster iteration.
3. A small diagram on the page showing the two pipelines side by side.

Do **not** add: embeddings, a live chat UI, multi-turn conversation, a query router, or a second
judge model. Those are out of scope on purpose.
