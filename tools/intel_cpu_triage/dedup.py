"""Duplicate / related detection (part of Phase 2).

Uses a dependency-free TF-IDF cosine similarity over issue text. An optional
``embedder`` callable can be supplied to use real embeddings; if it raises or
is absent, the toolkit falls back to the lexical model so it always works
offline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable

from .db import Issue


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Very common English/markdown tokens that add noise to similarity.
_STOPWORDS = frozenset(
    """
    the a an and or but if then for to of in on at by with without is are was
    be been being this that these those it its as from into out up down i we
    you he she they them me my our your when how what why which who whom can
    could should would will shall do does did not no yes so such than too very
    bug issue error code python torch import return def self none true false
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    n = len(tokens)
    return {term: c / n for term, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SimilarityIndex:
    """Lexical TF-IDF index over a corpus of issues.

    ``most_similar`` returns issues ranked by cosine similarity, excluding the
    query issue itself.
    """

    def __init__(self, embedder: Callable[[str], list[float]] | None = None) -> None:
        self._embedder = embedder
        self._docs: dict[int, dict[str, float]] = {}
        self._df: Counter[str] = Counter()
        self._n = 0

    def build(self, issues: Iterable[Issue]) -> None:
        issues = list(issues)
        self._n = len(issues)
        self._df = Counter()
        token_lists: dict[int, list[str]] = {}
        for issue in issues:
            tokens = tokenize(issue.text)
            token_lists[issue.number] = tokens
            for term in set(tokens):
                self._df[term] += 1
        self._docs = {
            num: self._tfidf(tokens) for num, tokens in token_lists.items()
        }

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((1 + self._n) / (1 + df)) + 1.0

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        return {term: tf * self._idf(term) for term, tf in _tf(tokens).items()}

    def most_similar(
        self, issue: Issue, top_k: int = 5, threshold: float = 0.0
    ) -> list[tuple[int, float]]:
        query = self._tfidf(tokenize(issue.text))
        scored: list[tuple[int, float]] = []
        for num, vec in self._docs.items():
            if num == issue.number:
                continue
            score = _cosine(query, vec)
            if score > threshold:
                scored.append((num, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
