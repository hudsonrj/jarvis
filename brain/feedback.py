"""Stage 5 - write the correction back.

This is the stage that makes the loop a loop. Without it the brain is a
search index over things you once saved; with it, every session leaves the
brain measurably better than it found it.

Three kinds of write-back:

``reinforce``  a recalled note helped, or misled — adjust its strength
``correct``    a note was wrong — store the old text, replace it, relink
``verify``     a claim was checked — record the verdict and the evidence
"""

from __future__ import annotations

from .connect import Connector, _derive_title, extract_claim_texts, keywords_of
from .models import Claim, Correction, Link, Note, now
from .store import Store
from .text import STOPWORDS, tokens

DAY = 86400.0
MAX_STRENGTH = 8.0
MIN_STRENGTH = 0.05


class Learner:
    def __init__(self, store: Store, connector: Connector, session_id: str = "") -> None:
        self.store = store
        self.connector = connector
        self.session_id = session_id

    # --- reinforcement ---------------------------------------------------

    def reinforce(self, note_ids: list[str], useful: bool = True, amount: float = 0.35) -> list[Note]:
        """Reward notes that helped; penalize notes that misled.

        Growth is multiplicative and capped, so a note cannot become
        unfalsifiable by being cited often.
        """
        touched: list[Note] = []
        for note_id in note_ids:
            note = self.store.get_note(note_id)
            if not note:
                continue
            if useful:
                note.wins += 1
                note.strength = min(MAX_STRENGTH, note.strength * (1.0 + amount))
            else:
                note.losses += 1
                note.strength = max(MIN_STRENGTH, note.strength * (1.0 - amount))
            self.store.update_note(note)
            touched.append(note)
        self.store.log(
            "reinforce",
            {"notes": [n.id for n in touched], "useful": useful, "amount": amount},
            session_id=self.session_id,
        )
        return touched

    def feedback_on_brief(self, brief_id: str, useful: bool = True) -> list[Note]:
        """Apply one verdict to every note a brief cited."""
        return self.reinforce(self.store.get_brief_note_ids(brief_id), useful=useful)

    def decay(self, half_life_days: float = 90.0) -> int:
        """Pull every note's strength toward its baseline. Run this periodically.

        Without decay, a note that was useful once outranks a note that is
        useful now, forever.
        """
        count = 0
        for note in self.store.all_notes():
            age_days = max(0.0, (now() - (note.updated_at or now())) / DAY)
            if age_days <= 0:
                continue
            factor = 0.5 ** (age_days / half_life_days)
            target = 1.0 + (note.strength - 1.0) * factor
            if abs(target - note.strength) < 1e-6:
                continue
            note.strength = max(MIN_STRENGTH, min(MAX_STRENGTH, target))
            # Written directly so decay does not count as an update.
            self.store.db.execute(
                "UPDATE notes SET strength = ? WHERE id = ?", (note.strength, note.id)
            )
            count += 1
        self.store.db.commit()
        self.store.log("decay", {"notes": count, "half_life_days": half_life_days}, self.session_id)
        return count

    # --- correction ------------------------------------------------------

    def correct(self, note_id: str, new_body: str, reason: str = "", new_title: str = "") -> Correction | None:
        """Replace a note's content, keeping what it used to say.

        The old text is never thrown away: it goes into ``corrections``, so the
        brain can show how a belief changed and why.
        """
        note = self.store.get_note(note_id)
        if not note:
            return None
        new_body = new_body.strip()
        if not new_body or new_body == note.body:
            return None
        correction = Correction(
            note_id=note.id,
            before=note.body,
            after=new_body,
            reason=reason,
            session_id=self.session_id,
        )
        self.store.add_correction(correction)
        note.body = new_body
        if new_title:
            note.title = new_title
        elif _title_is_stale(note.title, new_body):
            # A title describing the old facts is worse than no title: it is
            # what recall shows and what a brief cites.
            note.title = _derive_title(new_body)
        note.keywords = keywords_of(note.text)
        note.revision += 1
        note.kind = "atomic" if note.kind == "stub" else note.kind
        # A corrected note is more trustworthy than it was, not less.
        note.strength = min(MAX_STRENGTH, note.strength * 1.2)
        self.store.update_note(note)
        self.connector.relink(note)
        self._resettle_claims(note, correction.before)
        self.store.log(
            "correct",
            {"note": note.id, "revision": note.revision, "reason": reason},
            session_id=self.session_id,
        )
        return correction

    def _resettle_claims(self, note: Note, old_body: str) -> None:
        """Retire claims the correction invalidated, then extract fresh ones.

        A claim lifted from text that no longer exists must not keep sitting in
        briefs as though it were still the note's position. It is marked
        ``superseded`` — neither verified nor awaiting a verdict — and the
        corrected body is mined for the claims that replace it.
        """
        body_flat = " ".join(note.body.lower().split())
        for claim in self.store.claims_for_notes([note.id]):
            if claim.status == "superseded":
                continue
            if " ".join(claim.text.lower().split()) in body_flat:
                continue  # survived the edit untouched
            claim.status = "superseded"
            claim.checked_at = now()
            claim.evidence = [*claim.evidence, f"superseded by revision {note.revision}"]
            self.store.add_claim(claim)
        existing = {
            " ".join(c.text.lower().split())
            for c in self.store.claims_for_notes([note.id])
        }
        for text in extract_claim_texts(note.body):
            if " ".join(text.lower().split()) not in existing:
                self.store.add_claim(Claim(note_id=note.id, text=text))

    def add_note(self, title: str, body: str, tags: list[str] | None = None, link_to: list[str] | None = None) -> Note:
        """Write a new insight straight into the brain — a session's own output."""
        note = Note(
            title=title.strip() or "Untitled note",
            body=body.strip(),
            kind="atomic",
            tags=tags or [],
            keywords=keywords_of(f"{title}\n{body}"),
        )
        self.store.add_note(note)
        self.connector.relink(note)
        for other in link_to or []:
            if self.store.get_note(other):
                self.store.add_link(
                    Link(src=note.id, dst=other, relation="derived_from", weight=0.85, origin="manual")
                )
        self.store.log("learn", {"note": note.id, "title": note.title}, session_id=self.session_id)
        return note

    def link(self, src: str, dst: str, relation: str = "relates_to", weight: float = 0.8) -> Link | None:
        """Assert a link a human or agent knows about and the inference missed."""
        if not (self.store.get_note(src) and self.store.get_note(dst)) or src == dst:
            return None
        link = Link(src=src, dst=dst, relation=relation, weight=weight, origin="manual")
        self.store.add_link(link)
        return link

    # --- verification ----------------------------------------------------

    def verify(self, claim_id: str, status: str, evidence: str = "") -> Claim | None:
        """Record a verdict on a claim. This is what makes knowledge checkable."""
        allowed = {"supported", "refuted", "contested", "unverified", "superseded"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        claim = self.store.get_claim(claim_id)
        if not claim:
            return None
        claim.status = status
        claim.checked_at = now()
        if evidence:
            claim.evidence = [*claim.evidence, evidence]
        self.store.add_claim(claim)
        # A refuted claim drags its note's credibility down with it.
        note = self.store.get_note(claim.note_id)
        if note:
            if status == "refuted":
                note.losses += 1
                note.strength = max(MIN_STRENGTH, note.strength * 0.6)
            elif status == "supported":
                note.wins += 1
                note.strength = min(MAX_STRENGTH, note.strength * 1.15)
            self.store.update_note(note)
        self.store.log(
            "verify",
            {"claim": claim.id, "status": status, "note": claim.note_id},
            session_id=self.session_id,
        )
        return claim


def _title_is_stale(title: str, new_body: str) -> bool:
    """True when a title's distinctive words are gone from the corrected text."""
    terms = [t for t in tokens(title) if len(t) > 3 and t not in STOPWORDS]
    if not terms:
        return True
    body_terms = set(tokens(new_body))
    kept = sum(1 for t in terms if t in body_terms)
    return kept / len(terms) < 0.5
