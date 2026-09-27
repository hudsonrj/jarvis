import pytest

from brain.act import Briefer, as_prompt
from brain.capture import capture_text
from brain.models import Note


@pytest.fixture
def briefer(store, connector, recaller, latency_doc):
    connector.connect(store.add_source(capture_text(latency_doc)))
    return Briefer(store, recaller)


def test_brief_cites_notes_with_ids_and_sources(briefer):
    brief = briefer.brief("what is our p99 latency budget")
    assert brief.recalls
    assert "note:" in brief.context
    assert "Source:" in brief.context
    assert brief.cited_note_ids()


def test_brief_indexes_are_sequential(briefer):
    brief = briefer.brief("what did we ship for the index")
    for i in range(1, len(brief.recalls) + 1):
        assert f"## [{i}] " in brief.context


def test_brief_refuses_to_pad_when_nothing_is_known(briefer):
    brief = briefer.brief("what is the airspeed velocity of an unladen swallow")
    assert brief.recalls == []
    assert "holds nothing relevant" in brief.context
    assert "do not invent" in brief.context


def test_brief_respects_the_character_budget(briefer):
    brief = briefer.brief("latency budget index retrieval", budget_chars=400)
    assert len(brief.recalls) == 1  # one note always goes in, even if oversized


def test_brief_flags_open_claims(briefer):
    brief = briefer.brief("what is our p99 latency budget")
    assert brief.open_claims
    assert "Unverified claims" in brief.context


def test_brief_marks_low_confidence(store, recaller, embedder):
    note = store.add_note(Note(
        title="Tangential note",
        body="The office fern needs watering twice a week during the summer months.",
        keywords=["fern", "watering", "office"],
    ))
    store.put_vector(note.id, embedder.name, embedder.embed(note.text))
    brief = Briefer(store, recaller).brief("watering schedule for plants")
    if brief.recalls:
        assert "Low confidence" in brief.context


def test_brief_counts_a_use_for_every_cited_note(store, briefer):
    brief = briefer.brief("what is our p99 latency budget")
    for note_id in brief.cited_note_ids():
        assert store.get_note(note_id).uses == 1


def test_brief_is_persisted_with_its_citations(store, briefer):
    brief = briefer.brief("what did we ship")
    assert store.get_brief_note_ids(brief.id) == brief.cited_note_ids()


def test_brief_warns_about_a_previously_corrected_note(store, briefer):
    note = store.all_notes()[0]
    note.losses = 3
    note.wins = 0
    store.update_note(note)
    brief = briefer.brief(note.title)
    assert "previously corrected" in brief.context


def test_as_prompt_forbids_invention_and_names_gaps(briefer):
    prompt = as_prompt(briefer.brief("p99 latency budget for kubernetes autoscaling"))
    assert "never fill the gap with invention" in prompt
    assert "holds nothing on" in prompt


def test_as_prompt_lists_unverified_claims(briefer):
    prompt = as_prompt(briefer.brief("what is our p99 latency budget"))
    assert "Unverified claims" in prompt
