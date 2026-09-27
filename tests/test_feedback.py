import pytest

from brain.capture import capture_text
from brain.feedback import MAX_STRENGTH, MIN_STRENGTH, Learner, _title_is_stale
from brain.models import Note, now


@pytest.fixture
def learner(store, connector, latency_doc):
    connector.connect(store.add_source(capture_text(latency_doc)))
    return Learner(store, connector, session_id="ses_test")


def test_reinforce_raises_strength_and_records_a_win(store, learner):
    note = store.all_notes()[0]
    before = note.strength
    learner.reinforce([note.id], useful=True)
    after = store.get_note(note.id)
    assert after.strength > before
    assert after.wins == 1


def test_penalize_lowers_strength_and_records_a_loss(store, learner):
    note = store.all_notes()[0]
    learner.reinforce([note.id], useful=False)
    after = store.get_note(note.id)
    assert after.strength < 1.0
    assert after.losses == 1


def test_strength_is_capped_in_both_directions(store, learner):
    note = store.all_notes()[0]
    for _ in range(60):
        learner.reinforce([note.id], useful=True)
    assert store.get_note(note.id).strength <= MAX_STRENGTH
    for _ in range(200):
        learner.reinforce([note.id], useful=False)
    assert store.get_note(note.id).strength >= MIN_STRENGTH


def test_reinforcing_an_unknown_note_is_a_no_op(learner):
    assert learner.reinforce(["note_does_not_exist"]) == []


def test_correction_keeps_the_old_text(store, learner):
    note = store.all_notes()[0]
    original = note.body
    correction = learner.correct(note.id, "The budget was raised to 250 milliseconds.", reason="new SLO")
    assert correction is not None
    assert correction.before == original
    assert store.get_note(note.id).body == "The budget was raised to 250 milliseconds."
    assert store.corrections_for(note.id)[0].reason == "new SLO"


def test_correction_bumps_the_revision(store, learner):
    note = store.all_notes()[0]
    learner.correct(note.id, "Completely different content about the deployment target.")
    assert store.get_note(note.id).revision == note.revision + 1


def test_correcting_to_the_same_text_is_a_no_op(store, learner):
    note = store.all_notes()[0]
    assert learner.correct(note.id, note.body) is None


def test_correcting_an_unknown_note_returns_none(learner):
    assert learner.correct("note_nope", "anything") is None


def test_a_stale_title_is_replaced(store, learner):
    note = store.add_note(Note(title="Deployed on the office mini PC", body="We run it on the mini PC."))
    learner.correct(note.id, "We moved everything to a Raspberry Pi 5 with a USB microphone.")
    assert "mini" not in store.get_note(note.id).title.lower()


def test_an_explicit_title_always_wins(store, learner):
    note = store.all_notes()[0]
    learner.correct(note.id, "Some entirely new content here.", new_title="Chosen title")
    assert store.get_note(note.id).title == "Chosen title"


def test_a_still_accurate_title_is_kept(store, learner):
    note = store.add_note(Note(title="Latency budget", body="The latency budget is 180 ms."))
    learner.correct(note.id, "The latency budget is now 250 ms after the new SLO was agreed.")
    assert store.get_note(note.id).title == "Latency budget"


def test_title_staleness_heuristic():
    assert _title_is_stale("Deployed on the office mini PC", "We moved to a Raspberry Pi 5.")
    assert not _title_is_stale("Latency budget", "The latency budget is now 250 ms.")


def test_correction_supersedes_claims_that_no_longer_apply(store, learner):
    seed = store.list_claims()[0]
    note = store.get_note(seed.note_id)
    claims_before = store.claims_for_notes([note.id])
    assert claims_before, "fixture should produce at least one claim"
    learner.correct(
        note.id,
        "The deployment target is a Raspberry Pi 5 and the microphone must stay local.",
    )
    after = {c.id: c for c in store.claims_for_notes([note.id])}
    for old in claims_before:
        assert after[old.id].status == "superseded"
        assert any("superseded" in e for e in after[old.id].evidence)


def test_correction_extracts_fresh_claims(store, learner):
    note = store.get_note(store.list_claims()[0].note_id)
    learner.correct(
        note.id,
        "The new deployment target is a Raspberry Pi 5, and the microphone must stay local "
        "because network round trips are what break the latency budget.",
    )
    fresh = [c for c in store.claims_for_notes([note.id]) if c.status == "unverified"]
    assert fresh
    assert "Raspberry Pi 5" in " ".join(c.text for c in fresh)


def test_a_surviving_claim_is_not_superseded(store, connector):
    text = (
        "# Budget\nThe p99 latency budget for the search endpoint is 180 milliseconds. "
        "Nothing else in this note matters very much at all right now."
    )
    result = connector.connect(store.add_source(capture_text(text)))
    note, claim = result.notes[0], result.claims[0]
    learner = Learner(store, connector)
    learner.correct(note.id, claim.text + " We also added a timeout of two seconds on the client.")
    assert store.get_claim(claim.id).status == "unverified"


def test_verify_supported_strengthens_the_note(store, learner):
    claim = store.list_claims()[0]
    before = store.get_note(claim.note_id).strength
    learner.verify(claim.id, "supported", evidence="dashboard link")
    note = store.get_note(claim.note_id)
    assert store.get_claim(claim.id).status == "supported"
    assert note.strength > before
    assert "dashboard link" in store.get_claim(claim.id).evidence


def test_verify_refuted_weakens_the_note(store, learner):
    claim = store.list_claims()[0]
    before = store.get_note(claim.note_id).strength
    learner.verify(claim.id, "refuted", evidence="measured 52 GB")
    assert store.get_note(claim.note_id).strength < before
    assert store.get_note(claim.note_id).losses == 1


def test_verify_rejects_an_invalid_status(learner, store):
    with pytest.raises(ValueError):
        learner.verify(store.list_claims()[0].id, "probably-true")


def test_verify_an_unknown_claim_returns_none(learner):
    assert learner.verify("claim_nope", "supported") is None


def test_decay_pulls_strength_toward_baseline(store, learner):
    note = store.all_notes()[0]
    note.strength = 6.0
    note.updated_at = now() - (90 * 86400)
    store.add_note(note)
    learner.decay(half_life_days=90.0)
    decayed = store.get_note(note.id).strength
    assert 2.0 < decayed < 6.0  # halfway back toward 1.0


def test_decay_leaves_a_fresh_note_alone(store, learner):
    note = store.all_notes()[0]
    note.strength = 4.0
    store.update_note(note)
    learner.decay(half_life_days=90.0)
    assert store.get_note(note.id).strength == pytest.approx(4.0, abs=0.01)


def test_add_note_links_it_into_the_graph(store, learner):
    note = learner.add_note(
        "Index tradeoff",
        "Cutting the p99 from 180 ms to 95 ms cost about two percent of recall on search.",
    )
    assert store.links_of(note.id)


def test_manual_link_is_recorded(store, learner):
    a, b = store.all_notes()[:2]
    link = learner.link(a.id, b.id, relation="contradicts")
    assert link is not None
    assert any(l.relation == "contradicts" and l.origin == "manual" for l in store.links_of(a.id))


def test_a_note_cannot_link_to_itself(store, learner):
    a = store.all_notes()[0]
    assert learner.link(a.id, a.id) is None
