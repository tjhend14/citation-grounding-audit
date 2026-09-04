# Citation Grounding Audit

Adobe.com's Brand Concierge beta cites its sources. This measures whether those sources
actually support what it said — claim by claim — and then builds a retrieval pipeline over
the same pages that cites per sentence and verifies each claim before answering.

**[Read the report →](https://tjhend14.github.io/citation-grounding-audit/)**

## The number

40 prompts were run against the Brand Concierge beta. Every factual sentence in its answers
was scored against each page that answer cited, with a claim counting as grounded if *any*
cited page supports it — the charitable reading.

> **61% of Adobe's claims are supported by a page it cited. 39% are not supported by any of them.**
> 93 claims judged, of which 22 fully supported and 35 partially.

The same corpus wired to sentence-level citation and a verification gate grounds 98% of what
it writes — and declines to answer 16 of 31 questions rather than guess. Both halves of that
sentence matter; see Limitations.

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

Run each step in order. Every step prints an acceptance check and stops if it fails. All LLM
calls and fetched pages are cached to disk, so a second run costs nothing.

```bash
python -m src.parse_log    # 1. probe_log.csv -> probes, claims, source URLs
python -m src.fetch        # 2. fetch cited pages into cache/
python -m src.judge        # 3. judge every (claim, source) pair; emit a blank human sample
python -m src.agreement    # 3b. after a human fills out/human_labels.csv
python -m src.chunk        # 4. build the retrieval corpus
python -m src.concierge --acceptance   # 5. three retrieval smoke queries
python -m src.concierge    # 6. generate + verification gate over all cited probes
python -m src.compare      # 7. audit the output, write comparison.json
python -m src.render       # 8. render docs/index.html
```

Judging uses Claude Sonnet 5, generation uses Claude Haiku 4.5. A full run from a cold cache
costs under $1.

`out/human_labels.csv` is filled in by a person, not by the pipeline — the agreement number
means nothing otherwise.

## Limitations

- **This is not a test of site-wide search.** The corpus is the 35 pages Adobe's own retriever
  surfaced. It measures whether the pages a system points at support what it says, not whether
  better pages exist elsewhere on adobe.com.
- **Human agreement is 0.67, below the 0.80 the method targets.** On the binary
  grounded/not-grounded call that the headline rate actually depends on, agreement is 0.87.
  The disagreements sit entirely within the ungrounded labels — whether a wrong-product page is
  "Unsupported" or "Source unrelated" — and do not move the 61%.
- **The 98% is structurally favourable.** The pipeline writes its sentences *from* the passages
  it cites, so a high rate is partly a property of the design rather than a win over Adobe. It
  also answers less than half the time.
- **Pages were captured through a browser, not fetched.** adobe.com accepts the TCP connection
  from an automated HTTP client and then never responds. `src/fetch.py` works as specified; the
  cached text was collected by driving a real browser over the same URLs on 2026-09-04.
- **`temperature=0` is not set.** The `anthropic` 1.x SDK has removed `temperature` from
  `messages.create()` and current models reject sampling parameters. Determinism comes from the
  on-disk response cache instead.
- **Retrieval is BM25 with a small stopword list.** No embeddings, no vector store. Over a
  59-chunk corpus, IDF misreads function words as informative; dropping them was necessary to
  stop a credit-card offer page outranking Adobe's pricing page on "how much is Photoshop".

## Layout

```
data/     probe log, extracted probes/claims/URLs
cache/    fetched pages + every LLM response, keyed by hash (gitignored)
corpus/   chunked passages for retrieval
out/      audit results, agreement, comparison.json
docs/     the rendered page, served by GitHub Pages
src/      one module per step, each runnable on its own
```

## Not affiliated with Adobe

Eboda is a fictional brand; the styling is a visual shell for a portfolio prototype. The content
and product names underneath are real Adobe pages. This project is not affiliated with, endorsed
by, or connected to Adobe Inc.
