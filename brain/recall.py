"""Stage 3 - remember what matters.

Recall blends five signals so that no single failure mode dominates:

==========  ===========================================================
lexical     BM25 over the note index — exact terms, names, identifiers
semantic    cosine over embeddings — paraphrase and synonyms
graph       one-hop expansion from the strongest hits — context by proximity
strength    reinforcement earned from past sessions — what proved useful
recency     a gentle decay — fresher notes win ties
==========  ===========================================================

The first three signals decide *relevance*. The last two are priors: they
multiply relevance rather than adding to it, because a note that does not
answer the query should not surface just for being popular or fresh.

Every result carries its per-signal breakdown, so a bad answer can be traced
to the signal that produced it instead of being a black box.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import Embedder, calibrate, cosine, get_embedder
from .models import Note, Recall, now
from .store import Store
from .text import STOPWORDS, stem, tokens

HALF_LIFE_DAYS = 45.0
DAY = 86400.0


@dataclass(frozen=True)
class Weights:
    """Signal mix. Tune per build; the defaults favour precision over breadth."""

    lexical: float = 0.34
    semantic: float = 0.30
    graph: float = 0.14
    strength: float = 0.14
    recency: float = 0.08

    def total(self) -> float:
        return self.lexical + self.semantic + self.graph + self.strength + self.recency

    def normalized(self) -> "Weights":
        t = self.total()
        if t <= 0:
            return Weights()
        return Weights(
            lexical=self.lexical / t,
            semantic=self.semantic / t,
            graph=self.graph / t,
            strength=self.strength / t,
            recency=self.recency / t,
        )


# Presets referenced by the build profiles in registry.py.
PRECISE = Weights(lexical=0.45, semantic=0.25, graph=0.08, strength=0.14, recency=0.08)
EXPLORATORY = Weights(lexical=0.22, semantic=0.34, graph=0.26, strength=0.10, recency=0.08)
TEAM = Weights(lexical=0.30, semantic=0.26, graph=0.18, strength=0.20, recency=0.06)


class Recaller:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        weights: Weights = Weights(),
        half_life_days: float = HALF_LIFE_DAYS,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.weights = weights.normalized()
        self.half_life_days = half_life_days

    def recall(
        self,
        query: str,
        limit: int = 6,
        expand_graph: bool = True,
        min_score: float = 0.05,
    ) -> list[Recall]:
        """Return the notes most worth putting in front of an agent."""
        if not query.strip():
            return []
        lexical = dict(
            (note.id, score) for note, score in self.store.search_notes(query, limit=limit * 4)
        )
        semantic = self._semantic_scores(query, limit * 6)
        candidate_ids = set(lexical) | set(semantic)
        if not candidate_ids:
            return []

        graph = self._graph_scores(lexical, semantic) if expand_graph else {}
        candidate_ids |= set(graph)

        notes = {n.id: n for n in self.store.get_notes(sorted(candidate_ids))}
        direct = set(lexical) | set(semantic)
        results: list[Recall] = []
        for note_id, note in notes.items():
            signals = {
                "lexical": round(lexical.get(note_id, 0.0), 4),
                "semantic": round(semantic.get(note_id, 0.0), 4),
                "graph": round(graph.get(note_id, 0.0), 4),
                "strength": round(self._strength_score(note), 4),
                "recency": round(self._recency_score(note), 4),
            }
            score = self._blend(signals)
            if score < min_score:
                continue
            results.append(
                Recall(
                    note=note,
                    score=round(score, 4),
                    signals=signals,
                    path="direct" if note_id in direct else "neighbor",
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _blend(self, signals: dict[str, float]) -> float:
        """Relevance, modulated by the priors — never carried by them."""
        w = self.weights
        relevance_weight = w.lexical + w.semantic + w.graph
        if relevance_weight <= 0:
            return 0.0
        relevance = (
            w.lexical * signals["lexical"]
            + w.semantic * signals["semantic"]
            + w.graph * signals["graph"]
        ) / relevance_weight
        prior_weight = w.strength + w.recency
        if prior_weight <= 0:
            return round(relevance, 6)
        prior = (w.strength * signals["strength"] + w.recency * signals["recency"]) / prior_weight
        # prior_weight is the share the priors may swing the result by.
        return round(relevance * (1.0 - prior_weight + prior_weight * prior), 6)

    # --- signals ---------------------------------------------------------

    def _semantic_scores(self, query: str, limit: int) -> dict[str, float]:
        vectors = self.store.all_vectors()
        if not vectors:
            return {}
        query_vec = self.embedder.embed(query)
        scored = []
        for note_id, vec in vectors.items():
            if len(vec) != len(query_vec):
                continue  # vector written by a different embedding model
            sim = calibrate(cosine(query_vec, vec), self.embedder)
            if sim > 0:
                scored.append((note_id, sim))
        scored.sort(key=lambda p: p[1], reverse=True)
        return dict(scored[:limit])

    def _graph_scores(
        self, lexical: dict[str, float], semantic: dict[str, float], seeds: int = 4
    ) -> dict[str, float]:
        """Spread a fraction of each seed's score to its neighbors."""
        combined: dict[str, float] = {}
        for source in (lexical, semantic):
            for note_id, score in source.items():
                combined[note_id] = max(combined.get(note_id, 0.0), score)
        top = sorted(combined.items(), key=lambda p: p[1], reverse=True)[:seeds]
        spread: dict[str, float] = {}
        for note_id, seed_score in top:
            for other, weight, _relation in self.store.neighbors(note_id):
                # A note never boosts itself through its own edges. It may still
                # be boosted by a different seed it is linked to: two relevant,
                # linked notes corroborating each other is the point of the signal.
                if other == note_id:
                    continue
                contribution = seed_score * max(0.0, min(1.0, weight))
                if contribution > spread.get(other, 0.0):
                    spread[other] = contribution
        return spread

    def _strength_score(self, note: Note) -> float:
        """Squash unbounded strength into 0..1; 1.0 strength maps to ~0.5."""
        return note.strength / (1.0 + note.strength)

    def _recency_score(self, note: Note) -> float:
        age_days = max(0.0, (now() - (note.updated_at or now())) / DAY)
        return 0.5 ** (age_days / self.half_life_days)


def explain(recalls: list[Recall], weights: Weights = Weights()) -> str:
    """Human-readable trace of why each note came back."""
    w = weights.normalized()
    lines = [
        f"weights: lexical={w.lexical:.2f} semantic={w.semantic:.2f} graph={w.graph:.2f}"
        f" strength={w.strength:.2f} recency={w.recency:.2f}"
    ]
    for i, r in enumerate(recalls, 1):
        parts = " ".join(f"{k}={v:.2f}" for k, v in r.signals.items())
        lines.append(f"{i}. [{r.score:.3f}] {r.note.title}  ({r.path})\n     {parts}")
    return "\n".join(lines)


def top_gap_terms(query: str, recalls: list[Recall], limit: int = 5) -> list[str]:
    """Query terms that no recalled note actually covers — the brain's blind spots."""
    covered = {stem_ for r in recalls for stem_ in _stems_of(r.note.text)}
    gaps: list[str] = []
    for term in tokens(query):
        if len(term) < 3 or term in STOPWORDS:
            continue
        if stem(term) not in covered and term not in gaps:
            gaps.append(term)
    return gaps[:limit]


def _stems_of(text: str) -> set[str]:
    return {stem(t) for t in tokens(text)}


def coverage(recalls: list[Recall], strong: float = 0.45) -> float:
    """Confidence proxy in 0..1: how close the best hits are to a strong match.

    Below roughly 0.35 the brain is guessing, and a brief should say so rather
    than presenting weak context as an answer.
    """
    if not recalls:
        return 0.0
    top = [r.score for r in recalls[:3]]
    return round(min(1.0, (sum(top) / len(top)) / strong), 4)
