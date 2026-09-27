"""How sharp is this brain?

A claim like "your second brain gets sharper after every session" is worth
nothing unless it is measured. Sharpness is a 0-100 composite of five things
that separate knowledge from a pile of saved text:

============== ====== =====================================================
volume          0.15   how much the brain holds, log-scaled and saturating
linkage         0.25   links per note against a target of 3
connectedness   0.15   share of notes that are not orphans
verification    0.20   share of extracted claims that have been checked
trust           0.25   recalled notes that helped, versus notes that misled
============== ====== =====================================================

Capturing raw material on its own *lowers* the score, because unlinked and
unchecked text dilutes the brain. That is deliberate: the number only goes up
when you finish the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .store import Store

TARGET_LINKS_PER_NOTE = 3.0
SATURATION_NOTES = 300.0

WEIGHTS = {
    "volume": 0.15,
    "linkage": 0.25,
    "connectedness": 0.15,
    "verification": 0.20,
    "trust": 0.25,
}


@dataclass
class Sharpness:
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = " ".join(f"{k}={v:.2f}" for k, v in sorted(self.components.items()))
        counts = " ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        return f"sharpness {self.score:.1f}/100  [{parts}]  ({counts})"

    def weakest(self) -> str:
        """The component with the most headroom, weighted by how much it counts."""
        if not self.components or not self.counts.get("notes"):
            return "volume"
        # Rank by how much score is still on the table for that component.
        return max(
            self.components,
            key=lambda k: (1.0 - self.components[k]) * WEIGHTS.get(k, 0.0),
        )


ADVICE = {
    "volume": "capture more sources — the brain is too thin to be useful yet",
    "linkage": "run connect again or add manual links; ideas are sitting in isolation",
    "connectedness": "orphan notes exist that nothing points to; link or drop them",
    "verification": "check the open claims — unverified knowledge is not knowledge",
    "trust": "give feedback on briefs so the brain learns which notes actually helped",
}


def measure(store: Store) -> Sharpness:
    notes = store.all_notes()
    n_notes = len(notes)
    if n_notes == 0:
        return Sharpness(score=0.0, components={k: 0.0 for k in WEIGHTS}, counts={"notes": 0})

    n_links = store.count("links")
    n_orphans = len(store.orphan_notes())
    # Superseded claims belong to text that no longer exists: they are neither
    # awaiting a verdict nor evidence of diligence, so they leave the ratio.
    n_claims = store.count("claims", "status != 'superseded'")
    n_checked = store.count("claims", "status NOT IN ('unverified', 'superseded')")
    wins = sum(n.wins for n in notes)
    losses = sum(n.losses for n in notes)

    components = {
        "volume": min(1.0, math.log1p(n_notes) / math.log1p(SATURATION_NOTES)),
        "linkage": min(1.0, (n_links / n_notes) / TARGET_LINKS_PER_NOTE),
        "connectedness": 1.0 - (n_orphans / n_notes),
        # No claims at all means nothing has been made checkable yet.
        "verification": (n_checked / n_claims) if n_claims else 0.0,
        # Laplace-smoothed, so one lucky win does not read as perfect trust.
        "trust": (wins + 1) / (wins + losses + 2),
    }
    score = 100.0 * sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return Sharpness(
        score=round(score, 2),
        components={k: round(v, 4) for k, v in components.items()},
        counts={
            "notes": n_notes,
            "links": n_links,
            "orphans": n_orphans,
            "claims": n_claims,
            "checked": n_checked,
            "wins": wins,
            "losses": losses,
            "sources": store.count("sources"),
            "corrections": store.count("corrections"),
        },
    )


def next_move(store: Store) -> str:
    """One concrete suggestion: the cheapest way to raise the score."""
    s = measure(store)
    weak = s.weakest()
    return f"{ADVICE.get(weak, 'keep looping')} (weakest: {weak} at {s.components.get(weak, 0):.2f})"
