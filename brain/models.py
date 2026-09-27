"""Core records of the second brain.

Every stage of the loop (capture -> connect -> recall -> act -> correct)
reads and writes the dataclasses defined here.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# --- capture ---------------------------------------------------------------


@dataclass
class Source:
    """Something that came in from the outside: a file, a page, a note dump."""

    uri: str
    kind: str  # text | file | url | vault
    title: str = ""
    raw_text: str = ""
    id: str = field(default_factory=lambda: new_id("src"))
    captured_at: float = field(default_factory=now)
    digest: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.digest:
            self.digest = checksum(self.raw_text)
        if not self.title:
            self.title = self.uri


# --- connect ---------------------------------------------------------------


@dataclass
class Note:
    """One atomic idea, small enough to be linked and checked on its own."""

    title: str
    body: str
    source_id: str = ""
    id: str = field(default_factory=lambda: new_id("note"))
    kind: str = "atomic"  # atomic | summary | stub
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    strength: float = 1.0  # reinforcement score, grown by useful recalls
    uses: int = 0
    wins: int = 0
    losses: int = 0
    revision: int = 1

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()


@dataclass
class Link:
    """A typed edge between two notes."""

    src: str
    dst: str
    relation: str = "relates_to"
    weight: float = 0.5
    origin: str = "overlap"  # wikilink | overlap | manual | claim
    created_at: float = field(default_factory=now)


@dataclass
class Claim:
    """A checkable statement pulled out of a note."""

    note_id: str
    text: str
    id: str = field(default_factory=lambda: new_id("claim"))
    status: str = "unverified"  # unverified | supported | refuted | contested
    evidence: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=now)
    checked_at: float | None = None


# --- recall / act / correct -----------------------------------------------


@dataclass
class Recall:
    """A note that came back for a query, with the signals that ranked it."""

    note: Note
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    path: str = "direct"  # direct | neighbor


@dataclass
class Brief:
    """What gets handed to an agent: packed context plus what is missing."""

    query: str
    context: str
    recalls: list[Recall] = field(default_factory=list)
    open_claims: list[Claim] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("brief"))
    created_at: float = field(default_factory=now)

    def cited_note_ids(self) -> list[str]:
        return [r.note.id for r in self.recalls]


@dataclass
class Correction:
    """The write-back: what the brain believed, and what it believes now."""

    note_id: str
    before: str
    after: str
    reason: str = ""
    id: str = field(default_factory=lambda: new_id("corr"))
    session_id: str = ""
    created_at: float = field(default_factory=now)


@dataclass
class Session:
    """One working session, measured at open and at close."""

    id: str = field(default_factory=lambda: new_id("ses"))
    label: str = ""
    started_at: float = field(default_factory=now)
    ended_at: float | None = None
    opened_sharpness: float = 0.0
    closed_sharpness: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return round(self.closed_sharpness - self.opened_sharpness, 4)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
