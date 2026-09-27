from brain.capture import capture_text
from brain.models import Link, Note
from brain.sharpness import WEIGHTS, measure, next_move


def test_an_empty_brain_scores_zero(store):
    s = measure(store)
    assert s.score == 0.0
    assert s.weakest() == "volume"
    assert "capture" in next_move(store)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_score_stays_in_range(store, connector, latency_doc):
    connector.connect(store.add_source(capture_text(latency_doc)))
    assert 0.0 <= measure(store).score <= 100.0


def test_orphans_lower_connectedness(store):
    store.add_note(Note(title="Alone", body="Nothing points at this note at all."))
    assert measure(store).components["connectedness"] == 0.0


def test_linking_raises_the_score(store):
    a = store.add_note(Note(title="A", body="First note about retrieval quality."))
    b = store.add_note(Note(title="B", body="Second note about retrieval quality."))
    before = measure(store).score
    store.add_link(Link(src=a.id, dst=b.id))
    assert measure(store).score > before


def test_verifying_a_claim_raises_the_score(store, connector, latency_doc):
    from brain.feedback import Learner

    connector.connect(store.add_source(capture_text(latency_doc)))
    before = measure(store)
    assert before.components["verification"] == 0.0
    Learner(store, connector).verify(store.list_claims()[0].id, "supported")
    after = measure(store)
    assert after.components["verification"] > 0.0
    assert after.score > before.score


def test_superseded_claims_leave_the_verification_ratio(store, connector):
    from brain.feedback import Learner

    text = (
        "# Budget\nThe p99 latency budget for the search endpoint is 180 milliseconds, and "
        "the retrieval step is what consumes nearly all of that budget today."
    )
    result = connector.connect(store.add_source(capture_text(text)))
    assert result.claims
    Learner(store, connector).correct(
        result.notes[0].id,
        "The deployment target changed to a Raspberry Pi 5 and the microphone stays local.",
    )
    counts = measure(store).counts
    superseded = store.count("claims", "status = 'superseded'")
    assert superseded > 0
    # Superseded claims count neither as pending nor as checked.
    assert counts["claims"] == store.count("claims") - superseded


def test_misleading_notes_lower_trust(store, connector, latency_doc):
    from brain.feedback import Learner

    connector.connect(store.add_source(capture_text(latency_doc)))
    learner = Learner(store, connector)
    before = measure(store).components["trust"]
    learner.reinforce([n.id for n in store.all_notes()], useful=False)
    assert measure(store).components["trust"] < before


def test_next_move_names_the_weakest_component(store, connector, latency_doc):
    connector.connect(store.add_source(capture_text(latency_doc)))
    s = measure(store)
    assert s.weakest() in s.components
    assert s.weakest() in next_move(store)
