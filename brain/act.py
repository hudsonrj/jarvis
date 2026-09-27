"""Stage 4 - put it to work.

Recall returns ranked notes; an agent needs a bounded, cited block of text.
``Briefer`` packs the best notes into a character budget, names the open
claims so the agent knows what is not yet verified, and names the gaps so it
knows what the brain simply does not hold.

Every brief is persisted with the notes it cited. That record is what makes
the write-back in stage 5 possible: feedback arrives about a brief, and the
brain already knows which notes to credit or blame.
"""

from __future__ import annotations

from .models import Brief, Recall
from .recall import Recaller, coverage, top_gap_terms
from .store import Store

DEFAULT_BUDGET = 3500
LOW_CONFIDENCE = 0.35


class Briefer:
    def __init__(self, store: Store, recaller: Recaller, budget_chars: int = DEFAULT_BUDGET) -> None:
        self.store = store
        self.recaller = recaller
        self.budget_chars = budget_chars

    def brief(
        self,
        query: str,
        limit: int = 6,
        session_id: str = "",
        budget_chars: int | None = None,
    ) -> Brief:
        recalls = self.recaller.recall(query, limit=limit)
        budget = budget_chars or self.budget_chars
        kept, context = self._pack(query, recalls, budget)
        open_claims = self.store.claims_for_notes(
            [r.note.id for r in kept], status="unverified"
        )
        gaps = top_gap_terms(query, kept)
        brief = Brief(
            query=query,
            context=context,
            recalls=kept,
            open_claims=open_claims,
            gaps=gaps,
        )
        self.store.save_brief(brief, session_id=session_id)
        self.store.log(
            "act",
            {
                "brief_id": brief.id,
                "query": query,
                "notes": brief.cited_note_ids(),
                "coverage": coverage(kept),
            },
            session_id=session_id,
        )
        # Every citation counts as a use, whether or not it turns out to help.
        for r in kept:
            note = r.note
            note.uses += 1
            self.store.update_note(note)
        return brief

    def _pack(self, query: str, recalls: list[Recall], budget: int) -> tuple[list[Recall], str]:
        conf = coverage(recalls)
        header = [f"# Brief: {query}", ""]
        if not recalls:
            header += [
                "The brain holds nothing relevant to this query.",
                "Capture a source before acting, and do not invent the answer.",
            ]
            return [], "\n".join(header)
        if conf < LOW_CONFIDENCE:
            header += [
                f"_Low confidence ({conf:.2f}). Treat the context below as a lead, not a fact._",
                "",
            ]
        used = len("\n".join(header))
        kept: list[Recall] = []
        blocks: list[str] = []
        for i, r in enumerate(recalls, 1):
            block = self._render(i, r)
            if kept and used + len(block) > budget:
                break
            blocks.append(block)
            used += len(block)
            kept.append(r)
        return kept, "\n".join(header + blocks).strip()

    def _render(self, index: int, r: Recall) -> str:
        source = self.store.get_source(r.note.source_id)
        origin = source.uri if source else "unknown source"
        marks = [f"score {r.score:.2f}", r.path]
        if r.note.kind == "stub":
            marks.append("STUB — not yet written")
        if r.note.losses > r.note.wins:
            marks.append("previously corrected — verify before relying on it")
        lines = [
            f"## [{index}] {r.note.title}",
            f"<!-- note:{r.note.id} | {' | '.join(marks)} -->",
            r.note.body.strip(),
            f"Source: {origin}",
        ]
        claims = self.store.claims_for_notes([r.note.id])
        unverified = [c for c in claims if c.status == "unverified"]
        refuted = [c for c in claims if c.status == "refuted"]
        if refuted:
            lines.append("Refuted here: " + " / ".join(c.text for c in refuted[:2]))
        if unverified:
            lines.append(f"Unverified claims in this note: {len(unverified)}")
        return "\n".join(lines) + "\n"


def as_prompt(brief: Brief) -> str:
    """Render a brief as an instruction block for an LLM call."""
    parts = [
        "You are answering from a curated second brain. Use the context below.",
        "Cite notes by their [n] index. If the context does not answer the question,",
        "say so plainly and name what is missing — never fill the gap with invention.",
        "",
        brief.context,
    ]
    if brief.open_claims:
        parts += [
            "",
            "Unverified claims — flag these rather than asserting them:",
            *[f"- {c.text}" for c in brief.open_claims[:5]],
        ]
    if brief.gaps:
        parts += ["", "The brain holds nothing on: " + ", ".join(brief.gaps)]
    return "\n".join(parts)
