#!/usr/bin/env python3
"""A full turn of the loop, printed step by step.

    python examples/second_brain_demo.py

Runs against a temporary brain, so it never touches ~/.jarvis/brain.db, and
forces the offline embedder so it works with nothing else installed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("BRAIN_USE_OLLAMA_EMBED", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import Brain  # noqa: E402
from brain.embedding import HashEmbedder  # noqa: E402

MONDAY = """# Search latency budget
Our p99 latency budget for the search endpoint is 180 milliseconds. The retrieval step
takes most of it, because the vector scan is linear over every stored note.

## The fix we rejected
We considered caching the whole index in memory. That was rejected because the index is
40 GB and the box has only 16 GB of RAM, so it does not fit.

## What we shipped
We shipped an approximate index with a candidate cutoff. The p99 dropped to 95 ms and
recall fell by about two percent, which the product team accepted.
"""

TUESDAY = """# Reranking after the first pass
Reranking the top candidates after the first retrieval pass recovers most of the recall
that an approximate index gives up, and it costs only a few milliseconds because it runs
over a short list rather than the whole corpus.
"""


def rule(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        brain = Brain(Path(tmp) / "demo.db", profile="researcher", embedder=HashEmbedder())
        try:
            run(brain)
        finally:
            brain.close()
    return 0


def run(brain: Brain) -> None:
    rule("SESSION 1 — capture and connect")
    with brain.session("monday: latency review") as monday:
        result = brain.learn_from(MONDAY, title="latency review")
        print(f"notes={len(result.notes)} links={len(result.links)} claims={len(result.claims)}")
        for note in result.notes:
            print(f"  · {note.title}")

        rule("SESSION 1 — recall and act")
        brief = brain.brief("what is our p99 latency budget and what did we ship")
        print(brief.context)

        rule("SESSION 1 — write the correction back")
        brain.helped(brief)
        print("marked the cited notes as useful")
        claim = brain.open_claims(limit=1)[0]
        print(f"verifying: {claim.text[:80]}...")
        brain.verify(claim.id, "supported", evidence="grafana dashboard")
    print(f"\nsharpness after session 1: {monday.closed_sharpness:.1f} ({monday.delta:+.2f})")
    print(f"next move: {brain.next_move()}")

    rule("SESSION 2 — new material meets what is already there")
    with brain.session("tuesday: reranking") as tuesday:
        result = brain.learn_from(TUESDAY, title="reranking")
        new_note = result.notes[0]
        inferred = [l for l in result.links if l.origin == "overlap"]
        print(f"inferred link(s): {len(inferred)}")
        for link in inferred:
            other = brain.store.get_note(link.dst)
            if other:
                print(f"  · {new_note.title!r} → {other.title!r} ({link.weight:.2f})")

        if not inferred:
            # Worth showing rather than hiding: the offline fallback embedder
            # cannot see this relation, and lowering the threshold far enough to
            # catch it would also start linking notes that merely share a word.
            # The brain knows it is under-linked, and says so.
            print("  none — the offline embedder cannot see this relation.")
            print(f"  the brain's own diagnosis: {brain.next_move()}")
            related = brain.recall("approximate index and recall tradeoff", limit=3)
            target = next((r.note for r in related if r.note.id != new_note.id), None)
            if target:
                brain.learner.link(new_note.id, target.id, relation="extends")
                print(f"  linked by hand: {new_note.title!r} --extends--> {target.title!r}")
    print(f"\nsharpness after session 2: {tuesday.closed_sharpness:.1f} ({tuesday.delta:+.2f})")
    print("(raw capture dilutes the brain; it is the linking and checking that pay)")

    rule("SESSION 3 — the facts changed; correct, do not append")
    with brain.session("wednesday: new SLO") as wednesday:
        hit = brain.recall("p99 latency budget for search", limit=1)[0]
        print(f"the brain currently says: {hit.note.body.splitlines()[0]}")
        brain.correct(
            hit.note.id,
            "The p99 latency budget for the search endpoint was raised to 250 milliseconds "
            "after the new SLO was agreed, which gives the retrieval step more headroom.",
            reason="new SLO agreed with the product team",
        )
        after = brain.recall("p99 latency budget for search", limit=1)[0]
        print(f"the brain now says:       {after.note.body.splitlines()[0]}")
        history = brain.store.corrections_for(after.note.id)
        print(f"previous version kept in history: {len(history)} correction(s)")
        stale = [c for c in brain.store.claims_for_notes([after.note.id]) if c.status == "superseded"]
        print(f"claims retired as superseded: {len(stale)}")
    print(f"\nsharpness after session 3: {wednesday.closed_sharpness:.1f} ({wednesday.delta:+.2f})")

    rule("WHY a note came back — the signal breakdown")
    print(brain.why("approximate index and reranking", limit=3))

    rule("DID IT GET SHARPER?")
    print(brain.trend())
    print(f"\n{brain.sharpness()}")
    print(f"next move: {brain.next_move()}")


if __name__ == "__main__":
    raise SystemExit(main())
