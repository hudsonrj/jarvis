"""Command line for the second brain.

    python -m brain capture notes/meeting.md
    python -m brain learn https://example.com/post
    python -m brain ask "what did we decide about the index"
    python -m brain why "index decision"
    python -m brain claims --open
    python -m brain verify claim_abc supported --evidence "dashboard link"
    python -m brain correct note_abc "the corrected text" --reason "was out of date"
    python -m brain status
    python -m brain serve          # the web app, at http://localhost:8787
    python -m brain loop
"""

from __future__ import annotations

import argparse
import sys

from .act import as_prompt
from .capture import CaptureError
from .pipeline import PRESETS, Brain
from .recall import explain
from .registry import BUILDS, describe_loop
from .sharpness import measure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain",
        description="A second brain that gets sharper after every session.",
    )
    parser.add_argument("--db", default=None, help="path to the brain file (default ~/.jarvis/brain.db)")
    parser.add_argument(
        "--profile",
        default="default",
        choices=sorted(set(PRESETS) | set(BUILDS)),
        help="recall tuning: a preset, or one of the build profiles",
    )
    parser.add_argument("--session", default="", help="label this run as a session")
    parser.add_argument("--hash-embed", action="store_true", help="skip Ollama, use the offline embedder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="bring a source in without interpreting it")
    p.add_argument("target", help="file, directory, url, or literal text")
    p.add_argument("--title", default="")

    p = sub.add_parser("learn", help="capture and connect in one step")
    p.add_argument("target")
    p.add_argument("--title", default="")

    p = sub.add_parser("ask", help="recall context and print a cited brief")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prompt", action="store_true", help="print as an LLM prompt block")

    p = sub.add_parser("why", help="show the signals behind a recall")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=6)

    p = sub.add_parser("remember", help="write a new note straight into the brain")
    p.add_argument("title")
    p.add_argument("body")
    p.add_argument("--tag", action="append", default=[])

    p = sub.add_parser("correct", help="replace a note's body, keeping its history")
    p.add_argument("note_id")
    p.add_argument("body")
    p.add_argument("--reason", default="")
    p.add_argument("--title", default="")

    p = sub.add_parser("claims", help="list extracted claims")
    p.add_argument("--open", action="store_true", help="only unverified ones")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("verify", help="record a verdict on a claim")
    p.add_argument("claim_id")
    p.add_argument("status", choices=["supported", "refuted", "contested", "unverified"])
    p.add_argument("--evidence", default="")

    p = sub.add_parser("feedback", help="tell the brain whether a brief helped")
    p.add_argument("brief_id")
    p.add_argument("verdict", choices=["helped", "misled"])

    p = sub.add_parser("notes", help="list notes")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("orphans", help="notes nothing links to")

    p = sub.add_parser("decay", help="pull strengths back toward baseline")
    p.add_argument("--half-life", type=float, default=90.0)

    p = sub.add_parser("serve", help="open the web app for this brain")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; anything but localhost exposes the brain with no auth")

    sub.add_parser("status", help="sharpness, counts, and the cheapest next move")
    sub.add_parser("trend", help="sharpness session over session")
    sub.add_parser("reconnect", help="re-derive notes from every stored source")
    sub.add_parser("loop", help="print the loop and the 20 projects behind it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "loop":
        print(describe_loop())
        return 0

    embedder = None
    if args.hash_embed:
        from .embedding import HashEmbedder

        embedder = HashEmbedder()

    try:
        brain = Brain(
            args.db,
            profile=args.profile,
            embedder=embedder,
            # The web server answers requests on worker threads.
            same_thread=args.cmd != "serve",
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.session:
            with brain.session(args.session) as ses:
                code = _dispatch(brain, args)
            print(f"\nsession {ses.label}: sharpness {ses.opened_sharpness:.1f} -> "
                  f"{ses.closed_sharpness:.1f} ({ses.delta:+.2f})")
            return code
        return _dispatch(brain, args)
    finally:
        brain.close()


def _dispatch(brain: Brain, args: argparse.Namespace) -> int:
    if args.cmd == "capture":
        try:
            sources = brain.capture(args.target, title=args.title)
        except CaptureError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for s in sources:
            print(f"{s.id}  {s.kind:<5} {s.title}")
        print(f"\n{len(sources)} source(s) captured. Run 'learn' or 'reconnect' to turn them into notes.")
        return 0

    if args.cmd == "learn":
        try:
            result = brain.learn_from(args.target, title=args.title)
        except CaptureError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"notes={len(result.notes)} links={len(result.links)} "
              f"claims={len(result.claims)} stubs={len(result.stubs)}")
        for note in result.notes:
            print(f"  {note.id}  {note.title}")
        if result.claims:
            print("\nclaims to check:")
            for c in result.claims:
                print(f"  {c.id}  {c.text[:100]}")
        return 0

    if args.cmd == "ask":
        query = " ".join(args.query)
        brief = brain.brief(query, limit=args.limit)
        print(as_prompt(brief) if args.prompt else brief.context)
        if brief.recalls:
            print(f"\nbrief {brief.id} — say what it was worth:"
                  f"\n  python -m brain feedback {brief.id} helped")
        return 0

    if args.cmd == "why":
        print(explain(brain.recall(" ".join(args.query), limit=args.limit), brain.recaller.weights))
        return 0

    if args.cmd == "remember":
        note = brain.remember(args.title, args.body, tags=args.tag)
        print(f"{note.id}  {note.title}")
        return 0

    if args.cmd == "correct":
        correction = brain.correct(args.note_id, args.body, reason=args.reason, new_title=args.title)
        if not correction:
            print("error: no such note, or the text is unchanged", file=sys.stderr)
            return 1
        print(f"corrected {correction.note_id} ({correction.id})")
        return 0

    if args.cmd == "claims":
        claims = brain.open_claims(limit=args.limit) if args.open else brain.store.list_claims(limit=args.limit)
        if not claims:
            print("no claims" + (" awaiting a verdict" if args.open else ""))
            return 0
        for c in claims:
            print(f"{c.id}  [{c.status:<10}] {c.text[:110]}")
        return 0

    if args.cmd == "verify":
        claim = brain.verify(args.claim_id, args.status, evidence=args.evidence)
        if not claim:
            print(f"error: no claim {args.claim_id}", file=sys.stderr)
            return 1
        print(f"{claim.id} -> {claim.status}")
        return 0

    if args.cmd == "feedback":
        notes = brain.helped(args.brief_id) if args.verdict == "helped" else brain.misled(args.brief_id)
        if not notes:
            print(f"error: no brief {args.brief_id}, or it cited nothing", file=sys.stderr)
            return 1
        print(f"{args.verdict}: adjusted {len(notes)} note(s)")
        for n in notes:
            print(f"  {n.id}  strength={n.strength:.2f} wins={n.wins} losses={n.losses}  {n.title}")
        return 0

    if args.cmd == "notes":
        for n in brain.store.all_notes(limit=args.limit):
            links = len(brain.store.links_of(n.id))
            print(f"{n.id}  s={n.strength:4.2f} links={links:<3} r{n.revision}  {n.title}")
        return 0

    if args.cmd == "orphans":
        orphans = brain.prune_orphans()
        if not orphans:
            print("no orphans — everything is connected to something")
            return 0
        for n in orphans:
            print(f"{n.id}  [{n.kind}]  {n.title}")
        return 0

    if args.cmd == "decay":
        print(f"decayed {brain.decay(half_life_days=args.half_life)} note(s)")
        return 0

    if args.cmd == "status":
        s = measure(brain.store)
        print(f"brain: {brain.store.path}")
        print(f"profile: {brain.profile}   embedder: {brain.embedder.name}")
        print(s)
        print(f"next move: {brain.next_move()}")
        return 0

    if args.cmd == "trend":
        print(brain.trend())
        return 0

    if args.cmd == "serve":
        from .webapp import run as run_web

        run_web(brain, host=args.host, port=args.port)
        return 0

    if args.cmd == "reconnect":
        result = brain.reconnect_all()
        print(f"re-derived notes={len(result.notes)} links={len(result.links)} claims={len(result.claims)}")
        return 0

    print(f"error: unhandled command {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
