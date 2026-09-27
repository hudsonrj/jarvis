"""Stage 2 - connect the ideas.

A captured source is a blob. This stage breaks it into atomic notes, gives
each one keywords, links them to what the brain already holds, and pulls out
the statements that can later be proved or disproved.

Three link origins, in descending trust:

``wikilink``  an explicit ``[[Target]]`` the author wrote
``overlap``   inferred from shared keywords plus vector similarity
``manual``    asserted by a human or an agent after the fact
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embedding import Embedder, calibrate, cosine, get_embedder
from .models import Claim, Link, Note, Source
from .store import Store
from .text import content_tokens


# A sentence worth checking asserts something, rather than asking or musing.
ASSERTIVE = re.compile(
    r"\b("
    # copulas, auxiliaries and modals
    r"is|are|was|were|will|must|should|does|do|has|have|"
    # verbs that assert a relation or an effect
    r"causes?|means?|requires?|enables?|prevents?|beats?|outperforms?|costs?|takes?|"
    # plain present-tense verbs of change and measurement: a statement like
    # "strength decays with a half life" is as checkable as one built on "is"
    r"decays?|drops?|falls?|grows?|rises?|reduces?|increases?|improves?|degrades?|"
    r"dominates?|exceeds?|stops?|starts?|returns?|produces?|yields?|breaks?|"
    # portuguese
    r"é|são|deve|precisa|significa|permite|impede|custa|cai|cresce|reduz|aumenta"
    r")\b",
    re.I,
)
HEDGED = re.compile(r"\b(maybe|perhaps|might|could be|probably|talvez|possivelmente)\b", re.I)
# Markup and code are not assertions about the world. Without this, a captured
# markdown file donates its fenced blocks, tables and bullet runs to the claim
# list, and the open-questions view fills with things nobody can verify.
NOT_PROSE = re.compile(r"(```|\|\s*-{2,}|\*\*|\]\(|^\s*[$>]|\w+\(\)|::|=>|\{\}|^\s*\d+\.\s+\w+\s*\|)")
CODEY = re.compile(r"[{}<>|=\\/_`~^]")
WIKILINK = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]")
TAG = re.compile(r"(?<!\w)#([A-Za-z][\w/-]{1,40})")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])")


@dataclass
class ConnectResult:
    """What one connect pass produced."""

    notes: list[Note] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    stubs: list[Note] = field(default_factory=list)
    #: Notes a re-derivation deliberately left alone because they were corrected.
    kept: list[Note] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out = {
            "notes": len(self.notes),
            "links": len(self.links),
            "claims": len(self.claims),
            "stubs": len(self.stubs),
        }
        if self.kept:
            out["kept"] = len(self.kept)
        return out


class Connector:
    """Splits sources into linked, checkable notes and writes them to the store."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        min_note_chars: int = 120,
        max_note_chars: int = 1600,
        link_threshold: float = 0.15,
        max_links_per_note: int = 6,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.min_note_chars = min_note_chars
        self.max_note_chars = max_note_chars
        self.link_threshold = link_threshold
        self.max_links_per_note = max_links_per_note

    # --- public ----------------------------------------------------------

    def connect(self, source: Source, extract_claims: bool = True) -> ConnectResult:
        """Derive notes from a source. Safe to run again on the same source.

        Each note carries a stable key naming the slice of the source it came
        from, so a second pass updates the same note rather than adding a copy.
        A note a human has corrected is left alone: re-deriving from the original
        text would silently undo the correction, and the correction is the more
        recent truth.
        """
        result = ConnectResult()
        for chunk in split_into_chunks(source.raw_text, self.min_note_chars, self.max_note_chars):
            note = self._make_note(source, chunk)
            existing = self.store.find_note_by_key(note.key)
            if existing is not None:
                if existing.revision > 1:
                    result.kept.append(existing)
                    result.notes.append(existing)
                    continue
                note = self._merge_onto(existing, note)
            self.store.add_note(note)
            self.store.put_vector(note.id, self.embedder.name, self.embedder.embed(note.text))
            result.notes.append(note)
            if extract_claims:
                for claim in extract_claim_texts(note.body):
                    if any(c.text == claim for c in self.store.claims_for_notes([note.id])):
                        continue
                    result.claims.append(self.store.add_claim(Claim(note_id=note.id, text=claim)))

        for note in result.notes:
            result.links.extend(self._wikilinks(note, result))
            # Siblings are included: two halves of one document are often the
            # most related pair in the whole brain.
            result.links.extend(self._similar_links(note, exclude=set()))
        # Notes split from one source also stay adjacent in reading order.
        result.links.extend(self._sequence_links(result.notes))
        for link in result.links:
            self.store.add_link(link)
        return result

    def relink(self, note: Note) -> list[Link]:
        """Recompute inferred links for one note, e.g. after a correction."""
        self.store.put_vector(note.id, self.embedder.name, self.embedder.embed(note.text))
        links = self._similar_links(note, exclude=set())
        for link in links:
            self.store.add_link(link)
        return links

    # --- internals -------------------------------------------------------

    def _make_note(self, source: Source, chunk: "Chunk") -> Note:
        body = WIKILINK.sub(lambda m: m.group(1), chunk.text).strip()
        return Note(
            title=chunk.title or _derive_title(chunk.text),
            body=body,
            source_id=source.id,
            key=f"{source.id}#{chunk.index}",
            kind="atomic",
            tags=sorted(set(TAG.findall(chunk.text))),
            keywords=keywords_of(chunk.text),
        )

    @staticmethod
    def _merge_onto(existing: Note, fresh: Note) -> Note:
        """Refresh a note's derived content, keeping everything it has earned."""
        fresh.id = existing.id
        fresh.created_at = existing.created_at
        fresh.strength = existing.strength
        fresh.uses = existing.uses
        fresh.wins = existing.wins
        fresh.losses = existing.losses
        fresh.revision = existing.revision
        return fresh

    def _wikilinks(self, note: Note, result: ConnectResult) -> list[Link]:
        raw = self.store.get_source(note.source_id)
        targets = set()
        haystack = f"{note.title}\n{note.body}"
        if raw:
            # The note body has wikilinks flattened, so read them off the source
            # slice that produced this note.
            for target in WIKILINK.findall(raw.raw_text):
                if target.strip().lower() in haystack.lower():
                    targets.add(target.strip())
        links: list[Link] = []
        for target in targets:
            existing = self.store.find_note_by_title(target)
            if existing is None:
                existing = Note(
                    title=target,
                    body=f"Stub created by a [[{target}]] reference in {note.title!r}.",
                    kind="stub",
                    keywords=keywords_of(target),
                )
                self.store.add_note(existing)
                self.store.put_vector(existing.id, self.embedder.name, self.embedder.embed(existing.text))
                result.stubs.append(existing)
            if existing.id != note.id:
                links.append(Link(src=note.id, dst=existing.id, relation="references", weight=0.9, origin="wikilink"))
        return links

    def _similar_links(self, note: Note, exclude: set[str]) -> list[Link]:
        """Infer links from `0.6 x calibrated similarity + 0.4 x keyword overlap`.

        Similarity is calibrated through the embedder's own useful band, the same
        way recall does it. Comparing a raw cosine against a fixed threshold would
        mean something different for every backend, which is the whole reason
        calibration exists.

        Measured on the hashed backend over related and unrelated note pairs, the
        blend scores related pairs at 0.17-0.91 and unrelated pairs at 0.00-0.12
        (the 0.12 being two notes that merely share the word "first"). The default
        threshold of 0.15 sits in that gap. Pairs a human would relate but that
        share no vocabulary still fall through, because the hashed embedder cannot
        see them: see the known limits in docs/SECOND_BRAIN.md.
        """
        vec = self.store.get_vector(note.id)
        note_vec = vec[1] if vec else self.embedder.embed(note.text)
        own_keywords = set(note.keywords)
        scored: list[tuple[str, float]] = []
        for candidate in self.store.all_notes():
            if candidate.id == note.id or candidate.id in exclude:
                continue
            other = self.store.get_vector(candidate.id)
            semantic = calibrate(cosine(note_vec, other[1]), self.embedder) if other else 0.0
            lexical = _overlap(own_keywords, set(candidate.keywords))
            blended = 0.6 * semantic + 0.4 * lexical
            if blended >= self.link_threshold:
                scored.append((candidate.id, blended))
        scored.sort(key=lambda p: p[1], reverse=True)
        return [
            Link(src=note.id, dst=other, relation="relates_to", weight=round(score, 4), origin="overlap")
            for other, score in scored[: self.max_links_per_note]
        ]

    @staticmethod
    def _sequence_links(notes: list[Note]) -> list[Link]:
        return [
            Link(src=a.id, dst=b.id, relation="followed_by", weight=0.5, origin="sequence")
            for a, b in zip(notes, notes[1:])
        ]


# --- chunking --------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    title: str = ""
    #: Position in the source's reading order. Part of a note's stable key.
    index: int = 0


def split_into_chunks(text: str, min_chars: int = 120, max_chars: int = 1600) -> list[Chunk]:
    """Split on markdown headings first, then paragraphs, then hard length.

    Short fragments are merged forward so a note is always big enough to stand
    on its own, and oversized ones are cut so a note never swamps a brief.
    """
    text = text.strip()
    if not text:
        return []
    sections = _split_by_heading(text)
    chunks: list[Chunk] = []
    for title, body in sections:
        pieces = _merge_paragraphs(body, min_chars, max_chars)
        for part, piece in enumerate(pieces, 1):
            # A long section becomes several notes. Giving them all the section
            # heading makes a brief list "[1] Stage 3" and "[2] Stage 3" with no
            # way to tell them apart, so numbered parts carry the distinction.
            label = title if len(pieces) == 1 else f"{title} ({part}/{len(pieces)})"
            chunks.append(Chunk(text=piece, title=label))
    if not chunks:
        chunks = [Chunk(text=text)]
    for i, chunk in enumerate(chunks):
        chunk.index = i
    return chunks


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = HEADING.match(line.strip())
        if match:
            if any(l.strip() for l in buffer):
                sections.append((title, "\n".join(buffer).strip()))
            title = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if any(l.strip() for l in buffer):
        sections.append((title, "\n".join(buffer).strip()))
    return sections or [("", text)]


def _merge_paragraphs(body: str, min_chars: int, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    buffer = ""
    for para in paragraphs:
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(buffer) >= min_chars:
            out.extend(_hard_split(buffer, max_chars))
            buffer = ""
    if buffer:
        if out and len(buffer) < min_chars:
            merged = f"{out[-1]}\n\n{buffer}"
            out[-1:] = _hard_split(merged, max_chars)
        else:
            out.extend(_hard_split(buffer, max_chars))
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = SENTENCE_SPLIT.split(text)
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            out.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        out.append(buffer.strip())
    # A single sentence longer than the cap still has to be cut somewhere.
    final: list[str] = []
    for piece in out:
        while len(piece) > max_chars:
            final.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            final.append(piece)
    return final


# --- feature extraction ----------------------------------------------------


def keywords_of(text: str, top_k: int = 12) -> list[str]:
    """Frequency-ranked content words, then longest-first, then alphabetical.

    The length tie-break matters more than it looks. In a short note almost every
    content word appears exactly once, so a purely alphabetical tie-break makes
    the keyword set "the first twelve content words in alphabetical order" —
    which quietly drops the specific terms late in the alphabet ("reranking",
    "retrieval") in favour of vague early ones ("few", "gives"). Keywords feed
    the inferred-link signal and the search index, so that bias degrades both.
    Length is a crude but honest proxy for specificity.
    """
    counts: dict[str, int] = {}
    for token in content_tokens(text):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda p: (-p[1], -len(p[0]), p[0]))
    return [word for word, _ in ranked[:top_k]]


def extract_claim_texts(body: str, max_claims: int = 5) -> list[str]:
    """Pull out assertive, unhedged sentences — the ones worth verifying."""
    out: list[str] = []
    for raw in SENTENCE_SPLIT.split(body.replace("\n", " ")):
        sentence = " ".join(raw.split()).strip(" -*•")
        if not (40 <= len(sentence) <= 320):
            continue
        if sentence.endswith("?") or not ASSERTIVE.search(sentence):
            continue
        if HEDGED.search(sentence) or not _is_prose(sentence):
            continue
        out.append(sentence)
        if len(out) >= max_claims:
            break
    return out


def _is_prose(sentence: str) -> bool:
    """Reject code, tables and markup masquerading as a verifiable statement."""
    if NOT_PROSE.search(sentence):
        return False
    # A sentence dense in punctuation used by code is a snippet, not a claim.
    symbols = sum(1 for ch in sentence if CODEY.match(ch))
    return symbols / len(sentence) <= 0.04


def _derive_title(text: str, limit: int = 70) -> str:
    """First sentence of the text, trimmed at a word boundary."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    first = " ".join(lines[0].split()).lstrip("#-*•> ").strip() if lines else ""
    if not first:
        first = " ".join(text.split())
    sentence = SENTENCE_SPLIT.split(first)[0]
    title = sentence if len(sentence) >= 12 else first
    return _clip_words(title, limit) or "Untitled note"


def _clip_words(text: str, limit: int) -> str:
    """Cut to at most ``limit`` characters without slicing a word in half."""
    text = text.strip()
    if len(text) <= limit:
        return text.rstrip(" ,;:.")
    cut = text[: limit + 1]
    space = cut.rfind(" ")
    clipped = cut[:space] if space > limit * 0.6 else text[:limit]
    return clipped.rstrip(" ,;:.-")


def _overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: shared keywords over the smaller set.

    Jaccard divides by the union, which punishes two notes for each having a full
    keyword list even when they share the terms that matter. Two notes that agree
    on their key terms should read as related regardless of how much else each
    one covers, and measurement bears that out: on the same note pairs this
    scores genuinely related ones 0.03-0.11 higher than Jaccard while leaving
    unrelated ones at zero, which widens the usable gap.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
