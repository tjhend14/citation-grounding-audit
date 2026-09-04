"""Step 5 — BM25 retrieval over the chunk corpus.

    python -m src.retrieve "how much is photoshop"

    search(query, k=6) -> list[Chunk]

BM25Okapi over lowercase alphanumeric tokens. No embeddings, no vector store
(SPEC.md section 1). `search` is deliberately the only thing callers touch, so a
different retriever can be swapped in behind it without changing the concierge.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_JSONL

DEFAULT_K = 6

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words are dropped from both the index and the query.
#
# Why this is needed, given BM25 already downweights common terms: the corpus is
# 59 chunks, and over a corpus that small IDF is a noisy estimate. "per" occurs
# in only 4 chunks, so BM25 scores it as a rare, highly informative term — it
# was the single largest contributor to "How much is Photoshop per month?",
# which pushed an American Express credit-offer page (which repeats "per
# license") above Adobe's actual Photoshop pricing page. IDF cannot tell a
# preposition from a product name; a stopword list can.
#
# Deliberately a short list of English function words, not a query-specific one.
# It has to generalise across all 31 probe questions in step 7, not just the
# three acceptance queries. Words that carry retrieval signal in this domain
# ("much" — Adobe's own FAQs say "How much does Photoshop cost?") are kept.
STOPWORDS = frozenset(
    """a an and are as at be by do does for from in is it of on or that the
    this to with per""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus function words.

    Keeps '22' and '99' from 'US$22.99' so price queries can match.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    chunk_id: str
    url: str
    page_title: str | None
    heading: str | None
    text: str
    char_count: int
    score: float = 0.0

    @classmethod
    def from_record(cls, record: dict) -> "Chunk":
        return cls(**{k: record[k] for k in
                      ("chunk_id", "url", "page_title", "heading", "text", "char_count")})

    def label(self) -> str:
        """Heading if the page had one, else the page title."""
        return self.heading or self.page_title or ""


class BM25Retriever:
    """The retrieval interface. Swap this class out, keep `search` identical."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def from_corpus(cls) -> "BM25Retriever":
        if not CHUNKS_JSONL.exists():
            raise FileNotFoundError(
                f"{CHUNKS_JSONL} not found. Run `python -m src.chunk` first."
            )
        records = [json.loads(l) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls([Chunk.from_record(r) for r in records])

    def search(self, query: str, k: int = DEFAULT_K) -> list[Chunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda p: p[1], reverse=True)[:k]
        out = []
        for chunk, score in ranked:
            if score <= 0:
                continue
            hit = Chunk.from_record(chunk.__dict__)
            hit.score = float(score)
            out.append(hit)
        return out


_retriever: BM25Retriever | None = None


def get_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever.from_corpus()
    return _retriever


def search(query: str, k: int = DEFAULT_K) -> list[Chunk]:
    """SPEC.md section 6 step 5. The whole retrieval surface."""
    return get_retriever().search(query, k)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how much is photoshop per month"
    print(f"query: {query!r}")
    print()
    for i, c in enumerate(search(query), start=1):
        print(f"{i}. [{c.chunk_id}] score={c.score:.2f}  {c.page_title}")
        print(f"   heading: {c.label()}")
        print(f"   {c.text[:160].replace(chr(10), ' ')}...")
        print()
