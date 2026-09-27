"""SQLite storage for the brain: notes, links, claims, vectors, history.

One file on disk holds everything, so a brain is portable and diffable.
Full-text search uses FTS5 when the local sqlite has it, and falls back to
a LIKE scan when it does not.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import (
    Brief,
    Claim,
    Correction,
    Link,
    Note,
    Session,
    Source,
    now,
)
from .text import fts_query as _fts_query
from .text import stem, tokens as _tokens

DEFAULT_PATH = Path.home() / ".jarvis" / "brain.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    raw_text TEXT,
    digest TEXT UNIQUE,
    captured_at REAL,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    key TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    kind TEXT,
    tags TEXT,
    keywords TEXT,
    created_at REAL,
    updated_at REAL,
    strength REAL DEFAULT 1.0,
    uses INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    revision INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_notes_source ON notes(source_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_key ON notes(key) WHERE key != '';

CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relation TEXT,
    weight REAL,
    origin TEXT,
    created_at REAL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT,
    evidence TEXT,
    created_at REAL,
    checked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_claims_note ON claims(note_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);

CREATE TABLE IF NOT EXISTS vectors (
    note_id TEXT PRIMARY KEY,
    model TEXT,
    dim INTEGER,
    data TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    before TEXT,
    after TEXT,
    reason TEXT,
    session_id TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    label TEXT,
    started_at REAL,
    ended_at REAL,
    opened_sharpness REAL,
    closed_sharpness REAL,
    stats TEXT
);

CREATE TABLE IF NOT EXISTS briefs (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    query TEXT,
    note_ids TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    stage TEXT,
    payload TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_events_stage ON events(stage);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, body, keywords, note_id UNINDEXED, tokenize='unicode61'
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _unjson(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class Store:
    """Thin, explicit data layer. No ORM, no magic."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.fts = self._try_fts()
        self.db.commit()

    def _migrate(self) -> None:
        """Bring a brain file written by an older version up to date."""
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(notes)")}
        if "key" not in columns:
            self.db.execute("ALTER TABLE notes ADD COLUMN key TEXT")
            self.db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_key ON notes(key)"
                " WHERE key != ''"
            )

    def _try_fts(self) -> bool:
        try:
            self.db.executescript(FTS_SCHEMA)
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- sources ----------------------------------------------------------

    def find_source_by_digest(self, digest: str) -> Source | None:
        row = self.db.execute(
            "SELECT * FROM sources WHERE digest = ?", (digest,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def add_source(self, source: Source) -> Source:
        existing = self.find_source_by_digest(source.digest)
        if existing:
            return existing
        self.db.execute(
            "INSERT INTO sources (id, uri, kind, title, raw_text, digest, captured_at, meta)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source.id,
                source.uri,
                source.kind,
                source.title,
                source.raw_text,
                source.digest,
                source.captured_at,
                _json(source.meta),
            ),
        )
        self.db.commit()
        return source

    def get_source(self, source_id: str) -> Source | None:
        row = self.db.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def list_sources(self, limit: int = 100) -> list[Source]:
        rows = self.db.execute(
            "SELECT * FROM sources ORDER BY captured_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            uri=row["uri"],
            kind=row["kind"],
            title=row["title"] or "",
            raw_text=row["raw_text"] or "",
            digest=row["digest"] or "",
            captured_at=row["captured_at"] or 0.0,
            meta=_unjson(row["meta"], {}),
        )

    # --- notes ------------------------------------------------------------

    def add_note(self, note: Note) -> Note:
        self.db.execute(
            "INSERT OR REPLACE INTO notes (id, source_id, key, title, body, kind, tags,"
            " keywords, created_at, updated_at, strength, uses, wins, losses, revision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note.id,
                note.source_id,
                note.key,
                note.title,
                note.body,
                note.kind,
                _json(note.tags),
                _json(note.keywords),
                note.created_at,
                note.updated_at,
                note.strength,
                note.uses,
                note.wins,
                note.losses,
                note.revision,
            ),
        )
        self._index_note(note)
        self.db.commit()
        return note

    def update_note(self, note: Note) -> Note:
        note.updated_at = now()
        return self.add_note(note)

    def _index_note(self, note: Note) -> None:
        if not self.fts:
            return
        self.db.execute("DELETE FROM notes_fts WHERE note_id = ?", (note.id,))
        self.db.execute(
            "INSERT INTO notes_fts (title, body, keywords, note_id) VALUES (?, ?, ?, ?)",
            (note.title, note.body, " ".join(note.keywords), note.id),
        )

    def get_note(self, note_id: str) -> Note | None:
        row = self.db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def get_notes(self, note_ids: Sequence[str]) -> list[Note]:
        if not note_ids:
            return []
        marks = ",".join("?" * len(note_ids))
        rows = self.db.execute(
            f"SELECT * FROM notes WHERE id IN ({marks})", tuple(note_ids)
        ).fetchall()
        by_id = {r["id"]: self._row_to_note(r) for r in rows}
        return [by_id[i] for i in note_ids if i in by_id]

    def find_note_by_key(self, key: str) -> Note | None:
        """The note a given slice of a given source produced, if it exists."""
        if not key:
            return None
        row = self.db.execute("SELECT * FROM notes WHERE key = ?", (key,)).fetchone()
        return self._row_to_note(row) if row else None

    def find_note_by_title(self, title: str) -> Note | None:
        row = self.db.execute(
            "SELECT * FROM notes WHERE lower(title) = lower(?) LIMIT 1", (title,)
        ).fetchone()
        return self._row_to_note(row) if row else None

    def all_notes(self, limit: int | None = None) -> list[Note]:
        sql = "SELECT * FROM notes ORDER BY updated_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row_to_note(r) for r in self.db.execute(sql).fetchall()]

    def count(self, table: str, where: str = "", params: Iterable[Any] = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.db.execute(sql, tuple(params)).fetchone()["n"])

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(
            id=row["id"],
            source_id=row["source_id"] or "",
            key=(row["key"] if "key" in row.keys() else "") or "",
            title=row["title"],
            body=row["body"],
            kind=row["kind"] or "atomic",
            tags=_unjson(row["tags"], []),
            keywords=_unjson(row["keywords"], []),
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
            strength=row["strength"] if row["strength"] is not None else 1.0,
            uses=row["uses"] or 0,
            wins=row["wins"] or 0,
            losses=row["losses"] or 0,
            revision=row["revision"] or 1,
        )

    # --- search -----------------------------------------------------------

    def search_notes(self, query: str, limit: int = 20) -> list[tuple[Note, float]]:
        """Lexical search returning (note, relevance in 0..1).

        FTS5 narrows the candidate set; the ranking itself is a BM25 computed
        here, because sqlite's own bm25() output is compressed into a range too
        narrow to blend with the other recall signals.
        """
        terms = sorted({t for t in _tokens(query) if len(t) > 1})
        if not terms:
            return []
        candidates = self._candidates(terms, limit * 5)
        if not candidates:
            return []
        n_docs = max(1, self.count("notes"))
        avg_len = self._avg_note_length()
        df = {stem(t): self.doc_freq(t) for t in terms}
        scored = [
            (note, _bm25(terms, note, df, n_docs, avg_len)) for note in candidates
        ]
        scored = [p for p in scored if p[1] > 0.0]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:limit]

    def _candidates(self, terms: list[str], limit: int) -> list[Note]:
        if self.fts:
            try:
                rows = self.db.execute(
                    "SELECT note_id FROM notes_fts WHERE notes_fts MATCH ?"
                    " ORDER BY rank LIMIT ?",
                    (_fts_query(terms), limit),
                ).fetchall()
                return self.get_notes([r["note_id"] for r in rows])
            except sqlite3.OperationalError:
                pass
        stems = {stem(t) for t in terms}
        hay = [
            n for n in self.all_notes()
            if stems & {stem(t) for t in _tokens(n.text)}
        ]
        return hay[:limit]

    def doc_freq(self, term: str) -> int:
        """How many notes contain the term, counted on stems. Used for idf."""
        root = stem(term)
        if self.fts:
            try:
                row = self.db.execute(
                    "SELECT COUNT(*) AS n FROM notes_fts WHERE notes_fts MATCH ?",
                    (_fts_query([term]),),
                ).fetchone()
                return int(row["n"])
            except sqlite3.OperationalError:
                pass
        pattern = f"%{root}%"
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM notes WHERE lower(title) LIKE ? OR lower(body) LIKE ?",
            (pattern, pattern),
        ).fetchone()
        return int(row["n"])

    def _avg_note_length(self) -> float:
        row = self.db.execute(
            "SELECT AVG(LENGTH(title) + LENGTH(body)) AS a FROM notes"
        ).fetchone()
        chars = row["a"] or 1.0
        return max(1.0, chars / 5.5)  # ~5.5 chars per token

    # --- links ------------------------------------------------------------

    def add_link(self, link: Link) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO links (src, dst, relation, weight, origin, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (link.src, link.dst, link.relation, link.weight, link.origin, link.created_at),
        )
        self.db.commit()

    def neighbors(self, note_id: str) -> list[tuple[str, float, str]]:
        rows = self.db.execute(
            "SELECT dst AS other, weight, relation FROM links WHERE src = ?"
            " UNION SELECT src AS other, weight, relation FROM links WHERE dst = ?",
            (note_id, note_id),
        ).fetchall()
        return [(r["other"], r["weight"] or 0.0, r["relation"] or "") for r in rows]

    def links_of(self, note_id: str) -> list[Link]:
        rows = self.db.execute(
            "SELECT * FROM links WHERE src = ? OR dst = ?", (note_id, note_id)
        ).fetchall()
        return [
            Link(
                src=r["src"],
                dst=r["dst"],
                relation=r["relation"] or "",
                weight=r["weight"] or 0.0,
                origin=r["origin"] or "",
                created_at=r["created_at"] or 0.0,
            )
            for r in rows
        ]

    def orphan_notes(self) -> list[Note]:
        rows = self.db.execute(
            "SELECT * FROM notes WHERE id NOT IN (SELECT src FROM links)"
            " AND id NOT IN (SELECT dst FROM links)"
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

    # --- claims -----------------------------------------------------------

    def add_claim(self, claim: Claim) -> Claim:
        self.db.execute(
            "INSERT OR REPLACE INTO claims (id, note_id, text, status, evidence,"
            " created_at, checked_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                claim.id,
                claim.note_id,
                claim.text,
                claim.status,
                _json(claim.evidence),
                claim.created_at,
                claim.checked_at,
            ),
        )
        self.db.commit()
        return claim

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self.db.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return self._row_to_claim(row) if row else None

    def claims_for_notes(self, note_ids: Sequence[str], status: str | None = None) -> list[Claim]:
        if not note_ids:
            return []
        marks = ",".join("?" * len(note_ids))
        sql = f"SELECT * FROM claims WHERE note_id IN ({marks})"
        params: list[Any] = list(note_ids)
        if status:
            sql += " AND status = ?"
            params.append(status)
        rows = self.db.execute(sql, tuple(params)).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def list_claims(self, status: str | None = None, limit: int = 100) -> list[Claim]:
        sql = "SELECT * FROM claims"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = self.db.execute(sql, params + (limit,)).fetchall()
        return [self._row_to_claim(r) for r in rows]

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        return Claim(
            id=row["id"],
            note_id=row["note_id"],
            text=row["text"],
            status=row["status"] or "unverified",
            evidence=_unjson(row["evidence"], []),
            created_at=row["created_at"] or 0.0,
            checked_at=row["checked_at"],
        )

    # --- vectors ----------------------------------------------------------

    def put_vector(self, note_id: str, model: str, vec: Sequence[float]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO vectors (note_id, model, dim, data) VALUES (?, ?, ?, ?)",
            (note_id, model, len(vec), _json(list(vec))),
        )
        self.db.commit()

    def get_vector(self, note_id: str) -> tuple[str, list[float]] | None:
        row = self.db.execute(
            "SELECT model, data FROM vectors WHERE note_id = ?", (note_id,)
        ).fetchone()
        if not row:
            return None
        return row["model"], _unjson(row["data"], [])

    def all_vectors(self) -> dict[str, list[float]]:
        rows = self.db.execute("SELECT note_id, data FROM vectors").fetchall()
        return {r["note_id"]: _unjson(r["data"], []) for r in rows}

    # --- history ----------------------------------------------------------

    def add_correction(self, correction: Correction) -> Correction:
        self.db.execute(
            "INSERT INTO corrections (id, note_id, before, after, reason, session_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                correction.id,
                correction.note_id,
                correction.before,
                correction.after,
                correction.reason,
                correction.session_id,
                correction.created_at,
            ),
        )
        self.db.commit()
        return correction

    def corrections_for(self, note_id: str) -> list[Correction]:
        rows = self.db.execute(
            "SELECT * FROM corrections WHERE note_id = ? ORDER BY created_at", (note_id,)
        ).fetchall()
        return [
            Correction(
                id=r["id"],
                note_id=r["note_id"],
                before=r["before"] or "",
                after=r["after"] or "",
                reason=r["reason"] or "",
                session_id=r["session_id"] or "",
                created_at=r["created_at"] or 0.0,
            )
            for r in rows
        ]

    def save_session(self, session: Session) -> Session:
        self.db.execute(
            "INSERT OR REPLACE INTO sessions (id, label, started_at, ended_at,"
            " opened_sharpness, closed_sharpness, stats) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.label,
                session.started_at,
                session.ended_at,
                session.opened_sharpness,
                session.closed_sharpness,
                _json(session.stats),
            ),
        )
        self.db.commit()
        return session

    def list_sessions(self, limit: int = 20) -> list[Session]:
        rows = self.db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Session(
                id=r["id"],
                label=r["label"] or "",
                started_at=r["started_at"] or 0.0,
                ended_at=r["ended_at"],
                opened_sharpness=r["opened_sharpness"] or 0.0,
                closed_sharpness=r["closed_sharpness"] or 0.0,
                stats=_unjson(r["stats"], {}),
            )
            for r in rows
        ]

    def save_brief(self, brief: Brief, session_id: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO briefs (id, session_id, query, note_ids, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                brief.id,
                session_id,
                brief.query,
                _json(brief.cited_note_ids()),
                brief.created_at,
            ),
        )
        self.db.commit()

    def get_brief_note_ids(self, brief_id: str) -> list[str]:
        row = self.db.execute(
            "SELECT note_ids FROM briefs WHERE id = ?", (brief_id,)
        ).fetchone()
        return _unjson(row["note_ids"], []) if row else []

    def log(self, stage: str, payload: dict[str, Any], session_id: str = "") -> None:
        self.db.execute(
            "INSERT INTO events (session_id, stage, payload, created_at) VALUES (?, ?, ?, ?)",
            (session_id, stage, _json(payload), now()),
        )
        self.db.commit()

    def events(self, stage: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events"
        params: tuple[Any, ...] = ()
        if stage:
            sql += " WHERE stage = ?"
            params = (stage,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(sql, params + (limit,)).fetchall()
        return [
            {
                "stage": r["stage"],
                "session_id": r["session_id"],
                "payload": _unjson(r["payload"], {}),
                "created_at": r["created_at"],
            }
            for r in rows
        ]




def _bm25(
    terms: list[str],
    note: Note,
    df: dict[str, int],
    n_docs: int,
    avg_len: float,
    k1: float = 1.2,
    b: float = 0.75,
    title_boost: float = 2.0,
) -> float:
    """BM25 over stems, normalized against the best score the query could reach.

    Dividing by the perfect-match ceiling keeps the output in 0..1 so it can be
    mixed with vector similarity and graph signals without rescaling.
    """
    terms = [stem(t) for t in terms]
    body_terms = [stem(t) for t in _tokens(note.body)]
    title_terms = [stem(t) for t in _tokens(note.title)] + [
        stem(t) for t in _tokens(" ".join(note.keywords))
    ]
    doc_len = len(body_terms) + len(title_terms)
    if doc_len == 0:
        return 0.0
    norm = k1 * (1 - b + b * doc_len / avg_len)
    score = 0.0
    ceiling = 0.0
    for term in terms:
        idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
        ceiling += idf * (k1 + 1)
        tf = body_terms.count(term) + title_boost * title_terms.count(term)
        if tf:
            score += idf * (tf * (k1 + 1)) / (tf + norm)
    if ceiling <= 0:
        return 0.0
    return min(1.0, score / ceiling)

