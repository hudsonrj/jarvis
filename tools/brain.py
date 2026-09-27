"""Second-brain tools for the Jarvis agent.

These wrap the ``brain`` package so the assistant can save what it is told,
recall it later with citations, correct itself when it was wrong, and report
how sharp its own memory is.

Everything Jarvis does here goes through one shared brain file (default
``~/.jarvis/brain.db``), so what is learned by voice is the same knowledge the
``python -m brain`` command line sees.

Replies are written to be spoken: short, no markdown, no note ids read aloud.
"""

from __future__ import annotations

import logging
import os

from langchain.tools import tool

from brain import Brain
from brain.capture import CaptureError
from brain.recall import coverage

logger = logging.getLogger(__name__)

_brain: Brain | None = None
_profile = os.getenv("BRAIN_PROFILE", "default")
_last_brief_id: str | None = None


def get_brain() -> Brain:
    """One long-lived brain per process, opened on first use."""
    global _brain
    if _brain is None:
        # The path comes from BRAIN_DB via brain.store.default_path().
        _brain = Brain(profile=_profile)
        logger.info(
            "🧠 brain open at %s (profile=%s, embedder=%s)",
            _brain.store.path,
            _brain.profile,
            _brain.embedder.name,
        )
    return _brain


def close_brain() -> None:
    global _brain
    if _brain is not None:
        _brain.close()
        _brain = None


@tool("brain_remember")
def brain_remember(content: str) -> str:
    """Save something into the second brain so it can be recalled in later sessions.

    Use this whenever the user states a fact, a decision, a preference, or a
    lesson worth keeping, or says "remember this", "save that", "anota isso".
    It also accepts a file path, a directory of notes, or a URL to read.

    Input:
    - The text to remember, or a path or URL to capture.
    """
    brain = get_brain()
    try:
        result = brain.learn_from(content)
    except CaptureError as exc:
        return f"I could not save that: {exc}"
    if not result.notes:
        return "There was nothing in that worth saving."
    titles = ", ".join(n.title for n in result.notes[:3])
    parts = [f"Saved {len(result.notes)} note{'s' if len(result.notes) != 1 else ''}: {titles}."]
    if result.links:
        parts.append(f"I linked it to {len(result.links)} related idea"
                     f"{'s' if len(result.links) != 1 else ''}.")
    if result.claims:
        parts.append(f"I flagged {len(result.claims)} statement"
                     f"{'s' if len(result.claims) != 1 else ''} to verify later.")
    return " ".join(parts)


@tool("brain_recall")
def brain_recall(query: str) -> str:
    """Recall what the second brain knows about a topic, with its sources.

    Use this before answering anything that depends on what the user told you
    earlier, on their projects, decisions, or preferences — anything from a past
    session. If it returns nothing, say you do not know rather than guessing.

    Input:
    - A natural language question or topic.
    """
    global _last_brief_id
    brain = get_brain()
    brief = brain.brief(query)
    _last_brief_id = brief.id
    if not brief.recalls:
        return (
            f"I have nothing stored about {query!r}. "
            "Tell me and I will remember it for next time."
        )
    conf = coverage(brief.recalls)
    lines = []
    for i, r in enumerate(brief.recalls, 1):
        source = brain.store.get_source(r.note.source_id)
        # A spoken source is only worth naming when it points somewhere real; a
        # pasted-text source's title is derived from the note itself.
        origin = ""
        if source and source.kind in {"file", "url"} and source.title:
            origin = f" (from {source.title})"
        lines.append(f"{i}. {r.note.title}{origin}: {_condense(r.note.body)}")
    header = "Here is what I have" if conf >= 0.35 else "I only have weak matches, so treat this as a lead"
    tail = ""
    if brief.open_claims:
        n = len(brief.open_claims)
        tail = f"\n{n} statement{'s' if n != 1 else ''} here {'are' if n != 1 else 'is'} still unverified."
    return f"{header}:\n" + "\n".join(lines) + tail


@tool("brain_correct")
def brain_correct(topic: str, correction: str) -> str:
    """Fix something the second brain has wrong. This is how it gets sharper.

    Use this when the user says you were wrong, that something is out of date,
    or corrects a fact you recalled — "actually it is", "no, that changed",
    "that is wrong". The old version is kept in history, never lost.

    Input:
    - topic: what the wrong note is about, so it can be found.
    - correction: what is actually true.
    """
    brain = get_brain()
    hits = brain.recall(topic, limit=1)
    if not hits:
        note = brain.remember(topic[:70], correction)
        return f"I had nothing on that, so I saved it as a new note: {note.title}."
    target = hits[0].note
    old_title = target.title
    result = brain.correct(target.id, correction, reason=f"corrected by user re: {topic}")
    if result is None:
        return f"My note on {old_title!r} already says that."
    # Re-read: a correction can rename the note when the old title went stale.
    updated = brain.store.get_note(target.id) or target
    renamed = "" if updated.title == old_title else f" It is now filed as {updated.title!r}."
    return (
        f"Corrected my note on {old_title!r}, now at revision {updated.revision}. "
        f"I kept what it used to say.{renamed}"
    )


@tool("brain_feedback")
def brain_feedback(verdict: str) -> str:
    """Record whether the last recall was actually useful.

    Use this when the user says the recalled context helped, or that it was
    wrong or irrelevant — "that was it", "that helped", "not what I meant".
    Useful notes gain strength and surface earlier next time; misleading ones
    lose it.

    Input:
    - verdict: "helped" or "misled".
    """
    brain = get_brain()
    if _last_brief_id is None:
        return "I have not recalled anything yet in this session."
    useful = verdict.strip().lower() not in {"misled", "wrong", "bad", "no", "irrelevant"}
    notes = brain.helped(_last_brief_id) if useful else brain.misled(_last_brief_id)
    if not notes:
        return "That recall cited nothing, so there is nothing to adjust."
    direction = "strengthened" if useful else "weakened"
    return f"Noted. I {direction} {len(notes)} note{'s' if len(notes) != 1 else ''}."


@tool("brain_status")
def brain_status(detail: str = "") -> str:
    """Report how sharp the second brain currently is, and what would improve it.

    Use this when the user asks what you know, how much you remember, how big
    or how good your memory is, or what you should work on.

    Input:
    - detail: pass "trend" to hear how it changed over recent sessions.
    """
    brain = get_brain()
    if detail.strip().lower().startswith("trend"):
        return brain.trend()
    s = brain.sharpness()
    c = s.counts
    unchecked = c.get("claims", 0) - c.get("checked", 0)
    return (
        f"Sharpness {s.score:.0f} out of 100, across {_plural(c.get('notes', 0), 'note')} "
        f"and {_plural(c.get('links', 0), 'link')} from {_plural(c.get('sources', 0), 'source')}. "
        f"{_plural(unchecked, 'statement')} still need checking. "
        f"To improve: {brain.next_move()}"
    )


@tool("brain_open_questions")
def brain_open_questions(limit: str = "5") -> str:
    """List the statements the second brain has not verified yet.

    Use this when the user asks what is uncertain, what needs checking, or what
    the open questions are.

    Input:
    - limit: how many to list, as a number. Defaults to 5.
    """
    brain = get_brain()
    try:
        n = max(1, min(20, int(str(limit).strip() or 5)))
    except ValueError:
        n = 5
    claims = brain.open_claims(limit=n)
    if not claims:
        return "Nothing is waiting to be verified."
    lines = [f"{i}. {c.text}" for i, c in enumerate(claims, 1)]
    return f"{len(claims)} statement(s) still unverified:\n" + "\n".join(lines)


BRAIN_TOOLS = [
    brain_remember,
    brain_recall,
    brain_correct,
    brain_feedback,
    brain_status,
    brain_open_questions,
]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _condense(text: str, limit: int = 240) -> str:
    """Trim a note body to something speakable, cutting at a sentence if possible."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: stop + 1] if stop > limit * 0.5 else cut.rstrip() + "...")
