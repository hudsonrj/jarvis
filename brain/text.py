"""Tokenizing, stemming and stopwords — the vocabulary layer.

Kept in its own module because capture, connect, recall, embedding and the
store all need the same notion of a word. When the tokenizer and the stemmer
differ between indexing and querying, recall fails in ways that are very hard
to see, so there is exactly one of each.
"""

from __future__ import annotations

from typing import Iterable

STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "but", "by", "can", "could", "did", "do", "does", "each", "for",
    "from", "get", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "like", "make", "may", "me", "might", "more", "most", "must", "my",
    "no", "not", "of", "on", "one", "only", "or", "other", "our", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "too", "up",
    "use", "used", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "would", "you", "your", "he", "his", "her",
    "had", "been", "being", "am", "were", "shall",
    # portuguese, because the operator of this brain writes in both
    "aos", "com", "como", "da", "das", "de", "do", "dos", "e", "em", "ela",
    "ele", "eles", "essa", "esse", "esta", "este", "eu", "foi", "isso", "já", "mais",
    "mas", "na", "nas", "no", "nos", "não", "o", "os", "ou", "para", "pelo", "por",
    "que", "se", "sem", "ser", "seu", "sua", "são", "também", "tem", "um", "uma",
    "vai", "ver", "você", "é", "ao", "das", "num", "numa", "pela", "pelas", "pelos",
})

# Suffixes stripped so "deploying" matches "deployment" and "notas" matches
# "nota". Ordered longest-first so the most specific one wins. Deliberately
# shallow: an aggressive stemmer collapses distinct words and makes recall worse
# in ways that are hard to debug. It will not unify "running" with "run".
_SUFFIXES: tuple[str, ...] = (
    "ements", "ments", "ement", "ation", "ating", "ition",
    "ment", "ings", "ness", "ing", "ers", "est", "ied",
    "ed", "es", "er", "ly", "s",
    # portuguese
    "acoes", "mente", "ando", "endo", "indo", "coes", "oes", "ais",
)
_MIN_STEM = 4


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric words, keeping internal hyphens and underscores."""
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in "_-":
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def stem(token: str) -> str:
    """Strip one common inflection, never below a legible root."""
    token = token.lower()
    if len(token) <= _MIN_STEM:
        return token
    # "ies" -> "y" catches memories/memory and cities/city, which plain suffix
    # stripping would leave as two different words.
    if token.endswith("ies") and len(token) >= 6:
        return token[:-3] + "y"
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            root = token[: -len(suffix)]
            # "ss" endings ("class", "address") are not plurals.
            if suffix == "s" and root.endswith("s"):
                return token
            return root
    return token


def content_tokens(text: str, min_len: int = 3) -> list[str]:
    """Tokens that carry meaning: no stopwords, no bare numbers, no fragments.

    Stopwords are dropped before any similarity is computed. Left in, they give
    every pair of same-language texts a similarity floor, which is what makes a
    nonsense query look like a weak match for everything.
    """
    return [
        t for t in tokens(text)
        if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()
    ]


def stems(text: str) -> list[str]:
    return [stem(t) for t in tokens(text)]


def fts_query(terms: Iterable[str]) -> str:
    """Prefix-match on stems so inflections of a term still hit the index."""
    parts = []
    for term in terms:
        root = stem(term)
        # Quoting keeps FTS5 from parsing the term as its own syntax.
        parts.append(f'"{root}"*' if len(root) >= 3 else f'"{root}"')
    return " OR ".join(dict.fromkeys(parts))
