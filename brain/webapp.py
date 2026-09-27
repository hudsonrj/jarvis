"""A local web app for the second brain.

    python -m brain serve

Serves a single page on 127.0.0.1 that drives the whole loop: capture a source,
ask a question and read the cited brief, say whether it helped, verify the open
claims, correct a note that has gone stale, and watch sharpness move.

Built on ``http.server`` so the package keeps its no-dependency promise. It binds
to localhost by default.

A token is optional and off by default, because a brain on localhost does not
need one. Set ``BRAIN_TOKEN`` (or pass ``--token``) and every request must carry
it as HTTP Basic auth, with the token as the password and any username. Browsers
prompt for it and remember it; ``curl -u :$BRAIN_TOKEN`` works too. The token is
compared in constant time.

Basic auth sends the token on every request, so plain HTTP over an untrusted
network would leak it. Over a tailnet that traffic is already encrypted between
devices, which is the case this is meant for. On anything less private, put it
behind a reverse proxy with TLS.
"""

from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .act import as_prompt
from .capture import CaptureError
from .models import Brief
from .pipeline import Brain
from .recall import coverage
from .sharpness import measure

STATIC_DIR = Path(__file__).parent / "web"
MAX_BODY = 4 * 1024 * 1024


class BrainService:
    """Everything the HTTP layer is allowed to do, with writes serialized.

    The handler runs in a thread per request, and one sqlite connection cannot
    take two concurrent writes, so every call through here holds one lock. The
    work is milliseconds on a personal brain; correctness is worth more than the
    parallelism given up.
    """

    def __init__(self, brain: Brain) -> None:
        if getattr(brain.store, "same_thread", True):
            raise ValueError(
                "the web app answers requests on worker threads, so its brain must be "
                "opened with same_thread=False — otherwise sqlite refuses the first "
                "concurrent request, which is a confusing way to find out under load"
            )
        self.brain = brain
        self.lock = threading.Lock()
        self.last_brief: Brief | None = None

    # --- reads ------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        with self.lock:
            s = measure(self.brain.store)
            sessions = [
                {
                    "id": x.id,
                    "label": x.label or x.id,
                    "opened": round(x.opened_sharpness, 2),
                    "closed": round(x.closed_sharpness, 2),
                    "delta": x.delta,
                }
                for x in self.brain.history(limit=8)
                if x.ended_at
            ]
            return {
                "path": str(self.brain.store.path),
                "profile": self.brain.profile,
                "embedder": self.brain.embedder.name,
                "sharpness": s.score,
                "components": s.components,
                "counts": s.counts,
                "weakest": s.weakest(),
                "next_move": self.brain.next_move(),
                "sessions": sessions,
            }

    def notes(self, query: str = "", limit: int = 50) -> dict[str, Any]:
        with self.lock:
            if query.strip():
                found = [n for n, _ in self.brain.store.search_notes(query, limit=limit)]
            else:
                found = self.brain.store.all_notes(limit=limit)
            return {"notes": [self._note_row(n) for n in found]}

    def note(self, note_id: str) -> dict[str, Any]:
        if not note_id:
            raise BadRequest("pass ?id=<note id>")
        with self.lock:
            note = self.brain.store.get_note(note_id)
            if not note:
                raise NotFound(f"no note {note_id}")
            source = self.brain.store.get_source(note.source_id)
            links = []
            for link in self.brain.store.links_of(note.id):
                other_id = link.dst if link.src == note.id else link.src
                other = self.brain.store.get_note(other_id)
                if other:
                    links.append({
                        "id": other.id,
                        "title": other.title,
                        "relation": link.relation,
                        "origin": link.origin,
                        "weight": round(link.weight, 3),
                        "direction": "out" if link.src == note.id else "in",
                    })
            return {
                "note": self._note_row(note) | {"body": note.body, "tags": note.tags},
                "source": {"uri": source.uri, "title": source.title, "kind": source.kind}
                if source else None,
                "links": links,
                "claims": [self._claim_row(c) for c in self.brain.store.claims_for_notes([note.id])],
                "history": [
                    {"before": c.before, "after": c.after, "reason": c.reason, "at": c.created_at}
                    for c in self.brain.store.corrections_for(note.id)
                ],
            }

    def claims(self, status: str = "unverified", limit: int = 50) -> dict[str, Any]:
        with self.lock:
            rows = self.brain.store.list_claims(status=status or None, limit=limit)
            out = []
            for claim in rows:
                note = self.brain.store.get_note(claim.note_id)
                out.append(self._claim_row(claim) | {"note_title": note.title if note else ""})
            return {"claims": out}

    def graph(self, limit: int = 120) -> dict[str, Any]:
        """Nodes and edges for the map view, biggest notes first."""
        with self.lock:
            notes = self.brain.store.all_notes(limit=limit)
            keep = {n.id for n in notes}
            edges = []
            seen: set[tuple[str, str]] = set()
            for note in notes:
                for link in self.brain.store.links_of(note.id):
                    if link.src not in keep or link.dst not in keep:
                        continue
                    pair = tuple(sorted((link.src, link.dst)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    edges.append({
                        "source": link.src,
                        "target": link.dst,
                        "origin": link.origin,
                        "weight": round(link.weight, 3),
                    })
            degree: dict[str, int] = {}
            for e in edges:
                degree[e["source"]] = degree.get(e["source"], 0) + 1
                degree[e["target"]] = degree.get(e["target"], 0) + 1
            return {
                "nodes": [
                    {
                        "id": n.id,
                        "title": n.title,
                        "kind": n.kind,
                        "strength": round(n.strength, 3),
                        "degree": degree.get(n.id, 0),
                    }
                    for n in notes
                ],
                "edges": edges,
            }

    # --- writes -----------------------------------------------------------

    def capture(self, target: str) -> dict[str, Any]:
        if not target.strip():
            raise BadRequest("nothing to capture")
        with self.lock:
            before = measure(self.brain.store).score
            try:
                result = self.brain.learn_from(target)
            except CaptureError as exc:
                raise BadRequest(str(exc)) from exc
            after = measure(self.brain.store)
            return {
                "summary": result.summary(),
                "notes": [self._note_row(n) for n in result.notes],
                "claims": [self._claim_row(c) for c in result.claims],
                "stubs": [self._note_row(n) for n in result.stubs],
                "kept": [self._note_row(n) for n in result.kept],
                "sharpness": after.score,
                "delta": round(after.score - before, 2),
            }

    def ask(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise BadRequest("empty question")
        with self.lock:
            brief = self.brain.brief(query)
            self.last_brief = brief
            return {
                "brief_id": brief.id,
                "query": brief.query,
                "context": brief.context,
                "prompt": as_prompt(brief),
                "coverage": coverage(brief.recalls),
                "gaps": brief.gaps,
                "recalls": [
                    {
                        "id": r.note.id,
                        "title": r.note.title,
                        "body": r.note.body,
                        "score": r.score,
                        "path": r.path,
                        "signals": r.signals,
                        "revision": r.note.revision,
                        "source": self._source_label(r.note.source_id),
                    }
                    for r in brief.recalls
                ],
                "open_claims": [self._claim_row(c) for c in brief.open_claims],
            }

    def feedback(self, brief_id: str, verdict: str) -> dict[str, Any]:
        with self.lock:
            useful = verdict == "helped"
            notes = (
                self.brain.helped(brief_id) if useful else self.brain.misled(brief_id)
            )
            if not notes:
                raise BadRequest("that brief cited nothing to adjust")
            return {
                "adjusted": [self._note_row(n) for n in notes],
                "sharpness": measure(self.brain.store).score,
            }

    def verify(self, claim_id: str, status: str, evidence: str = "") -> dict[str, Any]:
        with self.lock:
            try:
                claim = self.brain.verify(claim_id, status, evidence=evidence)
            except ValueError as exc:
                raise BadRequest(str(exc)) from exc
            if not claim:
                raise NotFound(f"no claim {claim_id}")
            return {
                "claim": self._claim_row(claim),
                "sharpness": measure(self.brain.store).score,
            }

    def correct(self, note_id: str, body: str, reason: str = "", title: str = "") -> dict[str, Any]:
        if not body.strip():
            raise BadRequest("a correction needs text")
        with self.lock:
            correction = self.brain.correct(note_id, body, reason=reason, new_title=title)
            if correction is None:
                raise BadRequest("no such note, or the text is unchanged")
            note = self.brain.store.get_note(note_id)
            return {
                "note": self._note_row(note) | {"body": note.body} if note else None,
                "sharpness": measure(self.brain.store).score,
            }

    def remember(self, title: str, body: str) -> dict[str, Any]:
        if not body.strip():
            raise BadRequest("a note needs a body")
        with self.lock:
            note = self.brain.remember(title or body[:60], body)
            return {"note": self._note_row(note), "sharpness": measure(self.brain.store).score}

    def link(self, src: str, dst: str, relation: str = "relates_to") -> dict[str, Any]:
        with self.lock:
            link = self.brain.learner.link(src, dst, relation=relation or "relates_to")
            if link is None:
                raise BadRequest("both notes must exist and differ")
            return {"ok": True, "sharpness": measure(self.brain.store).score}

    # --- helpers ----------------------------------------------------------

    def _note_row(self, note: Any) -> dict[str, Any]:
        return {
            "id": note.id,
            "title": note.title,
            "kind": note.kind,
            "strength": round(note.strength, 3),
            "uses": note.uses,
            "wins": note.wins,
            "losses": note.losses,
            "revision": note.revision,
            "updated_at": note.updated_at,
            "excerpt": note.body[:220],
        }

    @staticmethod
    def _claim_row(claim: Any) -> dict[str, Any]:
        return {
            "id": claim.id,
            "note_id": claim.note_id,
            "text": claim.text,
            "status": claim.status,
            "evidence": claim.evidence,
        }

    def _source_label(self, source_id: str) -> str:
        source = self.brain.store.get_source(source_id)
        if not source:
            return ""
        return source.uri if source.kind in {"file", "url"} else (source.title or source.kind)


class BadRequest(Exception):
    status = 400


class NotFound(Exception):
    status = 404


def make_token(explicit: str | None = None) -> str:
    """The configured token, or empty when the app should stay open."""
    return (explicit if explicit is not None else os.environ.get("BRAIN_TOKEN", "")).strip()


def new_token(nbytes: int = 24) -> str:
    return secrets.token_urlsafe(nbytes)


def make_handler(service: BrainService, token: str = "") -> type[BaseHTTPRequestHandler]:
    routes_get: dict[str, Callable[[dict[str, list[str]]], dict[str, Any]]] = {
        "/api/state": lambda q: service.state(),
        "/api/notes": lambda q: service.notes(_one(q, "q"), _int(q, "limit", 50)),
        "/api/note": lambda q: service.note(_one(q, "id")),
        "/api/claims": lambda q: service.claims(_one(q, "status", "unverified"), _int(q, "limit", 50)),
        "/api/graph": lambda q: service.graph(_int(q, "limit", 120)),
    }
    routes_post: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "/api/capture": lambda b: service.capture(str(b.get("target", ""))),
        "/api/ask": lambda b: service.ask(str(b.get("query", ""))),
        "/api/feedback": lambda b: service.feedback(str(b.get("brief_id", "")), str(b.get("verdict", "helped"))),
        "/api/verify": lambda b: service.verify(
            str(b.get("claim_id", "")), str(b.get("status", "")), str(b.get("evidence", ""))
        ),
        "/api/correct": lambda b: service.correct(
            str(b.get("note_id", "")), str(b.get("body", "")),
            str(b.get("reason", "")), str(b.get("title", "")),
        ),
        "/api/remember": lambda b: service.remember(str(b.get("title", "")), str(b.get("body", ""))),
        "/api/link": lambda b: service.link(
            str(b.get("src", "")), str(b.get("dst", "")), str(b.get("relation", ""))
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "jarvis-brain"
        protocol_version = "HTTP/1.1"

        def _authorized(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            if not header.startswith("Basic "):
                return False
            try:
                decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return False
            _user, _, supplied = decoded.partition(":")
            # Constant time: a timing difference here leaks the token one byte
            # at a time to anyone who can reach the port.
            return hmac.compare_digest(supplied, token)

        def _challenge(self) -> None:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Second Brain", charset="UTF-8"')
            body = b'{"error": "this brain needs a token"}'
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._challenge()
                return
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                handler = routes_get.get(parsed.path)
                if handler is None:
                    self._error(404, f"no route {parsed.path}")
                    return
                self._run(lambda: handler(parse_qs(parsed.query)))
                return
            self._static(parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                # Drain the body first, or the connection desyncs on keep-alive.
                pending = int(self.headers.get("Content-Length") or 0)
                if 0 < pending <= MAX_BODY:
                    self.rfile.read(pending)
                self._challenge()
                return
            parsed = urlparse(self.path)
            handler = routes_post.get(parsed.path)
            if handler is None:
                self._error(404, f"no route {parsed.path}")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._error(413, "body too large")
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._error(400, "body must be JSON")
                return
            if not isinstance(body, dict):
                self._error(400, "body must be a JSON object")
                return
            self._run(lambda: handler(body))

        # --- plumbing ----------------------------------------------------

        def _run(self, work: Callable[[], dict[str, Any]]) -> None:
            try:
                self._json(200, work())
            except (BadRequest, NotFound) as exc:
                self._error(exc.status, str(exc))
            except Exception as exc:  # a local tool should say what broke
                self._error(500, f"{type(exc).__name__}: {exc}")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _static(self, path: str) -> None:
            if path == "/favicon.ico":
                # Answer it rather than leaving a 404 in the user's console.
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            name = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (STATIC_DIR / name).resolve()
            # Serve only from the bundled directory, whatever the request says.
            if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
                self._error(404, f"not found: {path}")
                return
            raw = target.read_bytes()
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: Any) -> None:
            return  # the CLI prints what matters; access logs are noise here

    return Handler


def serve(
    brain: Brain,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """Build the server. The caller decides whether to block on it."""
    handler = make_handler(BrainService(brain), token=make_token(token))
    return ThreadingHTTPServer((host, port), handler)


def run(
    brain: Brain,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str | None = None,
) -> None:
    resolved = make_token(token)
    httpd = serve(brain, host=host, port=port, token=resolved)
    shown = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print(f"second brain at http://{shown}:{port}")
    print(f"  file:     {brain.store.path}")
    print(f"  profile:  {brain.profile}   embedder: {brain.embedder.name}")
    print(f"  auth:     {'token required' if resolved else 'open — anyone who can reach the port'}")
    if host not in {"127.0.0.1", "localhost"} and not resolved:
        print("  WARNING: reachable beyond this machine with no token. Set BRAIN_TOKEN.")
    print("  ctrl-c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def _one(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key) or []
    return values[0] if values else default


def _int(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return max(1, min(1000, int(_one(query, key, str(default)))))
    except ValueError:
        return default
