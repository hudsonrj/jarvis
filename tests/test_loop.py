"""The loop as a whole: does a session actually leave the brain sharper?"""

import pytest

from brain import Brain
from brain.registry import BUILDS, PROJECTS, STAGES, by_stage, describe_loop, find


def test_a_full_loop_raises_sharpness(brain, latency_doc):
    with brain.session("first") as ses:
        brain.learn_from(latency_doc)
        b = brain.brief("what is our p99 latency budget")
        brain.helped(b)
        for claim in brain.open_claims(limit=2):
            brain.verify(claim.id, "supported", evidence="dashboard")
    assert ses.delta > 0
    assert ses.opened_sharpness == 0.0
    assert ses.stats["next_move"]


def test_each_session_is_recorded_with_its_delta(brain, latency_doc):
    with brain.session("one"):
        brain.learn_from(latency_doc)
    with brain.session("two"):
        for claim in brain.open_claims(limit=3):
            brain.verify(claim.id, "supported")
    history = brain.history()
    assert [s.label for s in history] == ["two", "one"]
    assert all(s.ended_at for s in history)
    assert "net over 2 session(s)" in brain.trend()


def test_capture_alone_does_not_inflate_the_score(brain, latency_doc):
    """Raw material is not knowledge: unlinked, unchecked text must not flatter."""
    with brain.session("verified") as good:
        brain.learn_from(latency_doc)
        for claim in brain.open_claims(limit=5):
            brain.verify(claim.id, "supported")
    with brain.session("dumped") as dump:
        brain.learn_from(
            "# Unrelated dump\nA loose paragraph about office plants that connects to nothing "
            "and asserts nothing anyone could ever check or disprove."
        )
    assert good.delta > 0
    assert dump.delta < good.delta


def test_correction_is_recalled_instead_of_the_original(brain):
    with brain.session("learn"):
        result = brain.learn_from(
            "# Deployment target\nJarvis runs on the office mini PC because the microphone "
            "has to be local and the latency of a round trip matters."
        )
        note = result.notes[0]
        brain.correct(
            note.id,
            "Jarvis now runs on a Raspberry Pi 5 with a USB microphone, because the mini PC "
            "was needed elsewhere and the Pi draws far less power.",
            reason="hardware changed",
        )
    hits = brain.recall("where does jarvis run")
    assert hits
    assert "Raspberry Pi 5" in hits[0].note.body
    assert "mini PC" not in hits[0].note.title


def test_feedback_reorders_later_recalls(brain):
    with brain.session("seed"):
        brain.learn_from(
            "# Retrieval notes\nThe retrieval step dominates the latency budget of the "
            "search endpoint in production today."
        )
        brain.learn_from(
            "# Retrieval draft\nThe retrieval step dominates the latency budget of the "
            "search endpoint in staging as well."
        )
    first = brain.recall("retrieval latency budget")
    assert len(first) >= 2
    underdog = first[-1].note.id
    for _ in range(8):
        brain.learner.reinforce([underdog], useful=True)
    brain.learner.reinforce([first[0].note.id], useful=False)
    assert brain.recall("retrieval latency budget")[0].note.id == underdog


def test_recall_refuses_to_answer_what_it_does_not_hold(brain, latency_doc):
    with brain.session("s"):
        brain.learn_from(latency_doc)
    brief = brain.brief("what is the vendor's payment schedule")
    assert brief.recalls == [] or "Low confidence" in brief.context


def test_why_explains_the_signals(brain, latency_doc):
    with brain.session("s"):
        brain.learn_from(latency_doc)
    out = brain.why("p99 latency budget")
    assert "weights:" in out
    assert "lexical=" in out


def test_profiles_tune_recall_differently(embedder, latency_doc):
    scores = {}
    for profile in ("researcher", "creator", "team"):
        with Brain(":memory:", profile=profile, embedder=embedder) as b:
            b.learn_from(latency_doc)
            hits = b.recall("latency budget")
            scores[profile] = hits[0].score if hits else 0.0
    assert len(set(round(v, 4) for v in scores.values())) > 1


def test_an_unknown_profile_is_rejected(embedder):
    with pytest.raises(ValueError, match="unknown profile"):
        Brain(":memory:", profile="nonsense", embedder=embedder)


def test_reconnect_is_idempotent(brain, latency_doc):
    """Counting only sources here once hid a bug that doubled every note."""
    with brain.session("s"):
        brain.learn_from(latency_doc)
    before = (
        brain.store.count("sources"),
        brain.store.count("notes"),
        brain.store.count("claims"),
    )
    brain.reconnect_all()
    brain.reconnect_all()
    assert (
        brain.store.count("sources"),
        brain.store.count("notes"),
        brain.store.count("claims"),
    ) == before


def test_learning_the_same_file_twice_does_not_double_the_brain(brain, tmp_path, latency_doc):
    doc = tmp_path / "notes.md"
    doc.write_text(latency_doc)
    with brain.session("s"):
        brain.learn_from(str(doc))
        notes = brain.store.count("notes")
        brain.learn_from(str(doc))
    assert brain.store.count("notes") == notes


def test_the_brain_survives_a_reopen(tmp_path, embedder, latency_doc):
    path = tmp_path / "brain.db"
    with Brain(path, embedder=embedder) as b:
        with b.session("write"):
            b.learn_from(latency_doc)
        expected = b.sharpness().score
    with Brain(path, embedder=embedder) as b:
        assert b.sharpness().score == expected
        assert b.recall("p99 latency budget")
        assert len(b.history()) == 1


def test_a_session_closes_even_when_the_work_raises(brain):
    with pytest.raises(RuntimeError):
        with brain.session("boom"):
            brain.remember("Partial", "Something worth keeping was saved before the failure.")
            raise RuntimeError("boom")
    assert brain.current_session is None
    assert brain.history()[0].ended_at is not None


# --- the registry of 20 projects ---------------------------------------

def test_twenty_projects_are_registered():
    assert len(PROJECTS) == 20
    assert len({p.n for p in PROJECTS}) == 20
    assert {p.n for p in PROJECTS} == set(range(1, 21))


def test_every_stage_has_five_projects():
    for stage in STAGES:
        assert len(by_stage(stage)) == 5


def test_every_project_names_a_local_adapter():
    for p in PROJECTS:
        assert p.local_adapter and p.slot


def test_truncated_urls_are_labelled_not_guessed():
    """The source list cut most paths off; none may be invented here."""
    for p in PROJECTS:
        if not p.url_complete:
            assert "truncated" in p.url
        else:
            assert p.url.startswith("https://github.com/")


def test_the_three_builds_reference_real_projects():
    assert set(BUILDS) == {"creator", "researcher", "team"}
    for build in BUILDS.values():
        assert len(build.stack) == 4
        for name in build.stack:
            assert find(name) is not None, name


def test_each_build_covers_all_four_stages():
    for build in BUILDS.values():
        stages = [find(name).stage for name in build.stack]
        assert stages == list(STAGES)


def test_describe_loop_mentions_every_project():
    text = describe_loop()
    for p in PROJECTS:
        assert p.name in text


def test_reconnect_reports_the_notes_it_preserved(brain, latency_doc):
    with brain.session("s"):
        note = brain.learn_from(latency_doc).notes[0]
        brain.correct(note.id, "A corrected body that supersedes what the source said.")
    assert note.id in {n.id for n in brain.reconnect_all().kept}
