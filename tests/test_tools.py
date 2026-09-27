"""The Jarvis-facing tool wrappers."""

import pytest

tools_brain = pytest.importorskip(
    "tools.brain", reason="langchain is not installed in this environment"
)


@pytest.fixture
def jarvis(tmp_path, monkeypatch):
    """A fresh brain behind the tool wrappers, isolated per test."""
    monkeypatch.setenv("BRAIN_DB", str(tmp_path / "brain.db"))
    monkeypatch.setattr(tools_brain, "_brain", None)
    monkeypatch.setattr(tools_brain, "_last_brief_id", None)
    yield tools_brain
    tools_brain.close_brain()


def call(tool, **kwargs):
    return tool.invoke(kwargs)


def test_all_six_tools_are_exported(jarvis):
    names = [t.name for t in jarvis.BRAIN_TOOLS]
    assert names == [
        "brain_remember",
        "brain_recall",
        "brain_correct",
        "brain_feedback",
        "brain_status",
        "brain_open_questions",
    ]


def test_every_tool_documents_itself_for_the_agent(jarvis):
    for tool in jarvis.BRAIN_TOOLS:
        assert tool.description and len(tool.description) > 80


def test_remember_then_recall(jarvis):
    saved = call(jarvis.brain_remember, content=(
        "We decided to deploy Jarvis on the office mini PC instead of the cloud, because "
        "the microphone has to be local and latency matters."
    ))
    assert "Saved" in saved
    out = call(jarvis.brain_recall, query="where are we deploying jarvis")
    assert "mini PC" in out


def test_recall_admits_when_it_knows_nothing(jarvis):
    out = call(jarvis.brain_recall, query="what is my bank password")
    assert "nothing stored" in out
    # It echoes the question and offers to learn, and invents no answer.
    assert "Here is what I have" not in out
    assert "weak matches" not in out


def test_remember_rejects_an_empty_input(jarvis):
    assert "could not save" in call(jarvis.brain_remember, content="   ")


def test_remember_reports_a_missing_file(jarvis):
    assert "could not save" in call(jarvis.brain_remember, content="notes/does-not-exist.md")


def test_correct_updates_what_recall_returns(jarvis):
    call(jarvis.brain_remember, content=(
        "We decided to deploy Jarvis on the office mini PC because the microphone has to be local."
    ))
    reply = call(jarvis.brain_correct,
                 topic="deploying jarvis",
                 correction="The Jarvis deployment moved to a Raspberry Pi 5 with a USB microphone.")
    assert "revision 2" in reply
    out = call(jarvis.brain_recall, query="where is jarvis deployed")
    assert "Raspberry Pi 5" in out
    assert "mini PC" not in out


def test_correct_saves_a_new_note_when_nothing_matches(jarvis):
    reply = call(jarvis.brain_correct,
                 topic="quarterly travel policy",
                 correction="Travel needs written approval from a director from now on.")
    assert "new note" in reply
    assert "approval" in call(jarvis.brain_recall, query="travel policy approval")


def test_feedback_needs_a_prior_recall(jarvis):
    assert "not recalled anything" in call(jarvis.brain_feedback, verdict="helped")


def test_feedback_strengthens_after_a_recall(jarvis):
    call(jarvis.brain_remember, content=(
        "The retrieval step dominates the latency budget of the search endpoint in production."
    ))
    call(jarvis.brain_recall, query="what dominates the latency budget")
    assert "strengthened" in call(jarvis.brain_feedback, verdict="helped")
    call(jarvis.brain_recall, query="what dominates the latency budget")
    assert "weakened" in call(jarvis.brain_feedback, verdict="misled")


def test_status_is_speakable_and_pluralized(jarvis):
    call(jarvis.brain_remember, content=(
        "The retrieval step dominates the latency budget of the search endpoint in production."
    ))
    out = call(jarvis.brain_status, detail="")
    assert "Sharpness" in out
    assert "1 note " in out and "1 notes" not in out
    assert "#" not in out and "**" not in out


def test_status_trend_reads_back_sessions(jarvis):
    call(jarvis.brain_remember, content="Something worth keeping about the retrieval budget here.")
    assert isinstance(call(jarvis.brain_status, detail="trend"), str)


def test_open_questions_lists_unverified_claims(jarvis):
    call(jarvis.brain_remember, content=(
        "The p99 latency budget for the search endpoint is 180 milliseconds in production."
    ))
    out = call(jarvis.brain_open_questions, limit="5")
    assert "unverified" in out
    assert "180 milliseconds" in out


def test_open_questions_handles_a_bad_limit(jarvis):
    call(jarvis.brain_remember, content=(
        "The p99 latency budget for the search endpoint is 180 milliseconds in production."
    ))
    assert "unverified" in call(jarvis.brain_open_questions, limit="not a number")


def test_open_questions_when_there_are_none(jarvis):
    assert "Nothing" in call(jarvis.brain_open_questions, limit="5")


def test_tools_share_one_brain_file(jarvis):
    call(jarvis.brain_remember, content="A fact about the deployment target worth remembering.")
    assert jarvis.get_brain() is jarvis.get_brain()
    assert "deployment" in call(jarvis.brain_recall, query="deployment target").lower()
