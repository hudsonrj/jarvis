"""The loop, assembled.

    capture the source
      -> turn it into linked, checkable knowledge
      -> recall the right context
      -> act with it
      -> write the correction back

``Brain`` is the one object everything else uses: the CLI, the Jarvis tools,
and any script. Open a session around a stretch of work and the brain records
what it was worth:

    with Brain() as brain:
        with brain.session("research") as ses:
            brain.learn_from("https://example.com/post")
            brief = brain.brief("what did the post claim about latency")
            brain.helped(brief)
        print(ses.delta)   # change in sharpness, signed
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import recall as recall_mod
from .capture import capture as capture_any
from .act import Briefer, as_prompt
from .connect import ConnectResult, Connector
from .embedding import Embedder, get_embedder
from .feedback import Learner
from .models import Brief, Claim, Note, Session, Source, now
from .recall import Recaller, Weights, coverage, explain
from .registry import BUILDS
from .sharpness import Sharpness, measure, next_move
from .store import Store

PRESETS: dict[str, Weights] = {
    "default": Weights(),
    "precise": recall_mod.PRECISE,
    "exploratory": recall_mod.EXPLORATORY,
    "team": recall_mod.TEAM,
}


class Brain:
    """A second brain: one sqlite file, five stages, one measurable score."""

    def __init__(
        self,
        path: str | Path | None = None,
        profile: str = "default",
        embedder: Embedder | None = None,
    ) -> None:
        if profile in BUILDS:
            build = BUILDS[profile]
            weights = PRESETS[build.weights_preset]
            budget, limit = build.budget_chars, build.recall_limit
        elif profile in PRESETS:
            weights = PRESETS[profile]
            budget, limit = 3500, 6
        else:
            raise ValueError(
                f"unknown profile {profile!r}; try one of "
                f"{sorted(set(PRESETS) | set(BUILDS))}"
            )
        self.profile = profile
        self.recall_limit = limit
        self.store = Store(path)
        self.embedder = embedder or get_embedder()
        self.connector = Connector(self.store, embedder=self.embedder)
        self.recaller = Recaller(self.store, embedder=self.embedder, weights=weights)
        self.briefer = Briefer(self.store, self.recaller, budget_chars=budget)
        self.current_session: Session | None = None
        self.learner = Learner(self.store, self.connector)

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Brain":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def session_id(self) -> str:
        return self.current_session.id if self.current_session else ""

    @contextmanager
    def session(self, label: str = "") -> Iterator[Session]:
        """Bracket a stretch of work and measure what it did to the brain."""
        ses = Session(label=label, opened_sharpness=measure(self.store).score)
        self.store.save_session(ses)
        self.current_session = ses
        self.learner.session_id = ses.id
        try:
            yield ses
        finally:
            after = measure(self.store)
            ses.ended_at = now()
            ses.closed_sharpness = after.score
            ses.stats = {
                "components": after.components,
                "counts": after.counts,
                "delta": ses.delta,
                "next_move": next_move(self.store),
            }
            self.store.save_session(ses)
            self.current_session = None
            self.learner.session_id = ""

    # --- stage 1 + 2: capture and connect --------------------------------

    def capture(self, target: str, title: str = "") -> list[Source]:
        """Bring something in without interpreting it yet."""
        sources = capture_any(target, title=title)
        stored = [self.store.add_source(s) for s in sources]
        self.store.log(
            "capture",
            {"targets": [s.uri for s in stored], "kind": stored[0].kind if stored else ""},
            session_id=self.session_id,
        )
        return stored

    def connect(self, source: Source, extract_claims: bool = True) -> ConnectResult:
        """Turn one source into linked, checkable notes."""
        result = self.connector.connect(source, extract_claims=extract_claims)
        self.store.log(
            "connect", {"source": source.id, **result.summary()}, session_id=self.session_id
        )
        return result

    def learn_from(self, target: str, title: str = "") -> ConnectResult:
        """Capture and connect in one call — the usual way material comes in."""
        merged = ConnectResult()
        for source in self.capture(target, title=title):
            part = self.connect(source)
            merged.notes += part.notes
            merged.links += part.links
            merged.claims += part.claims
            merged.stubs += part.stubs
        return merged

    def reconnect_all(self) -> ConnectResult:
        """Re-derive notes from every stored source. Use after changing the rules."""
        merged = ConnectResult()
        for source in self.store.list_sources(limit=10_000):
            part = self.connect(source)
            merged.notes += part.notes
            merged.links += part.links
            merged.claims += part.claims
            merged.stubs += part.stubs
        return merged

    # --- stage 3: recall -------------------------------------------------

    def recall(self, query: str, limit: int | None = None) -> list[recall_mod.Recall]:
        return self.recaller.recall(query, limit=limit or self.recall_limit)

    def why(self, query: str, limit: int | None = None) -> str:
        """Show the signal breakdown behind a recall. For debugging bad answers."""
        results = self.recall(query, limit=limit)
        return explain(results, self.recaller.weights)

    # --- stage 4: act ----------------------------------------------------

    def brief(self, query: str, limit: int | None = None) -> Brief:
        return self.briefer.brief(
            query, limit=limit or self.recall_limit, session_id=self.session_id
        )

    def prompt_for(self, query: str) -> str:
        """A ready-to-send context block for an LLM call."""
        return as_prompt(self.brief(query))

    def confidence(self, query: str) -> float:
        return coverage(self.recall(query))

    # --- stage 5: write the correction back ------------------------------

    def helped(self, brief: Brief | str) -> list[Note]:
        """Mark a brief's notes as having earned their place."""
        return self.learner.feedback_on_brief(_brief_id(brief), useful=True)

    def misled(self, brief: Brief | str) -> list[Note]:
        """Mark a brief's notes as having wasted the session's time."""
        return self.learner.feedback_on_brief(_brief_id(brief), useful=False)

    def correct(self, note_id: str, new_body: str, reason: str = "", new_title: str = "") -> object:
        return self.learner.correct(note_id, new_body, reason=reason, new_title=new_title)

    def remember(self, title: str, body: str, tags: list[str] | None = None,
                 link_to: list[str] | None = None) -> Note:
        """Write a new insight — typically something a session figured out."""
        return self.learner.add_note(title, body, tags=tags, link_to=link_to)

    def verify(self, claim_id: str, status: str, evidence: str = "") -> Claim | None:
        return self.learner.verify(claim_id, status, evidence=evidence)

    def open_claims(self, limit: int = 20) -> list[Claim]:
        return self.store.list_claims(status="unverified", limit=limit)

    def decay(self, half_life_days: float = 90.0) -> int:
        return self.learner.decay(half_life_days=half_life_days)

    def prune_orphans(self, keep_stubs: bool = True) -> list[Note]:
        """Report notes nothing links to. Reporting, not deleting — you decide."""
        orphans = self.store.orphan_notes()
        return [n for n in orphans if keep_stubs or n.kind != "stub"]

    # --- measurement -----------------------------------------------------

    def sharpness(self) -> Sharpness:
        return measure(self.store)

    def next_move(self) -> str:
        return next_move(self.store)

    def history(self, limit: int = 10) -> list[Session]:
        return self.store.list_sessions(limit=limit)

    def trend(self, limit: int = 10) -> str:
        """Did the brain actually get sharper, session over session?"""
        sessions = [s for s in self.history(limit) if s.ended_at]
        if not sessions:
            return "no completed sessions yet"
        lines = []
        for s in reversed(sessions):
            arrow = "+" if s.delta > 0 else ("=" if s.delta == 0 else "-")
            label = s.label or s.id
            lines.append(
                f"{arrow} {label:<20} {s.opened_sharpness:>6.1f} -> {s.closed_sharpness:>6.1f}"
                f"  ({s.delta:+.2f})"
            )
        gained = sum(s.delta for s in sessions)
        lines.append(f"  net over {len(sessions)} session(s): {gained:+.2f}")
        return "\n".join(lines)


def _brief_id(brief: Brief | str) -> str:
    return brief if isinstance(brief, str) else brief.id
