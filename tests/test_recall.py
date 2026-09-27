import pytest

from brain.capture import capture_text
from brain.models import Link, Note
from brain.recall import Recaller, Weights, coverage, top_gap_terms
from brain.store import stem


@pytest.fixture
def populated(store, connector):
    docs = [
        "# Hybrid recall\nRecall blends lexical search with vector similarity, because lexical "
        "search alone misses paraphrase and vectors alone miss exact identifiers.",
        "# Memory reinforcement\nNotes that are used and marked useful gain strength. Strength "
        "decays with a half life so stale context stops dominating the ranking.",
        "# Cloud invoice\nThe monthly invoice for the storage bucket was 41 dollars, billed to "
        "the business account on the first of the month.",
    ]
    for d in docs:
        connector.connect(store.add_source(capture_text(d)))
    return store


def test_the_relevant_note_wins(populated, recaller):
    results = recaller.recall("how does recall blend lexical and vector search")
    assert results[0].note.title == "Hybrid recall"


def test_an_unrelated_note_does_not_win_on_priors(populated, recaller):
    """Strength and recency are equal for every note here, so they must not rank."""
    results = recaller.recall("how does recall blend lexical and vector search")
    titles = [r.note.title for r in results]
    assert titles.index("Hybrid recall") < titles.index("Cloud invoice") if "Cloud invoice" in titles else True


def test_out_of_domain_query_returns_nothing(populated, recaller):
    assert recaller.recall("kubernetes pod autoscaling thresholds") == []


def test_coverage_reflects_confidence(populated, recaller):
    strong = coverage(recaller.recall("lexical search and vector similarity"))
    weak = coverage(recaller.recall("kubernetes pod autoscaling thresholds"))
    assert strong > weak
    assert weak == 0.0


def test_empty_query_returns_nothing(populated, recaller):
    assert recaller.recall("   ") == []


def test_signals_are_reported_for_every_hit(populated, recaller):
    for r in recaller.recall("lexical search"):
        assert set(r.signals) == {"lexical", "semantic", "graph", "strength", "recency"}


def test_inflected_query_still_matches(store, connector, recaller):
    connector.connect(store.add_source(capture_text(
        "# Deployment\nThe Jarvis deployment moved to a Raspberry Pi because the microphone "
        "has to stay local and the power draw is much lower there."
    )))
    assert recaller.recall("where are we deploying jarvis")


def test_stem_matches_inflections():
    assert stem("deploying") == stem("deployment") == stem("deploys")
    assert stem("memories") == stem("memory")
    assert stem("class") == stem("classes")


def test_strength_breaks_ties_between_equal_matches(store, embedder):
    body = "The retrieval step dominates the latency budget for the search endpoint."
    weak = store.add_note(Note(title="Weak copy", body=body, keywords=["retrieval", "latency"]))
    strong = store.add_note(Note(title="Strong copy", body=body, keywords=["retrieval", "latency"]))
    for note in (weak, strong):
        store.put_vector(note.id, embedder.name, embedder.embed(note.text))
    strong.strength = 6.0
    store.update_note(strong)
    results = Recaller(store, embedder=embedder).recall("retrieval latency budget")
    assert results[0].note.id == strong.id


def _linked_pair(store, connector, embedder, give_vector=True):
    connector.connect(store.add_source(capture_text(
        "# Index choice\nWe picked an approximate vector index over an exact scan for the "
        "search endpoint because the exact scan blew the latency budget."
    )))
    hidden = store.add_note(Note(
        title="Vendor contract terms",
        body="The contract with the vendor renews in March and is billed annually in advance.",
        keywords=["contract", "vendor", "renews"],
    ))
    if give_vector:
        store.put_vector(hidden.id, embedder.name, embedder.embed(hidden.text))
    seed = [n for n in store.all_notes() if n.title == "Index choice"][0]
    store.add_link(Link(src=seed.id, dst=hidden.id, relation="relates_to", weight=1.0, origin="manual"))
    return seed, hidden


def test_graph_expansion_lifts_a_linked_note(store, connector, embedder):
    """A note linked to a strong hit scores higher than it would on its own."""
    _seed, hidden = _linked_pair(store, connector, embedder)
    query = "approximate vector index choice"
    recaller = Recaller(store, embedder=embedder)
    expanded = {r.note.id: r for r in recaller.recall(query, min_score=0.0)}
    flat = {r.note.id: r for r in recaller.recall(query, expand_graph=False, min_score=0.0)}

    assert expanded[hidden.id].signals["graph"] > 0
    assert flat[hidden.id].signals["graph"] == 0
    assert expanded[hidden.id].score > flat[hidden.id].score


def test_two_relevant_linked_notes_corroborate_each_other(store, connector, embedder):
    """Both ends of a link may be seeds; neither may boost itself."""
    seed, hidden = _linked_pair(store, connector, embedder)
    results = {
        r.note.id: r
        for r in Recaller(store, embedder=embedder).recall(
            "approximate vector index choice", min_score=0.0
        )
    }
    assert results[seed.id].signals["graph"] > 0
    assert results[hidden.id].signals["graph"] > 0


def test_a_note_reachable_only_by_link_is_marked_neighbor(store, connector, embedder):
    """With no vector and no shared terms, the graph is the only way in."""
    _seed, hidden = _linked_pair(store, connector, embedder, give_vector=False)
    results = Recaller(store, embedder=embedder).recall(
        "approximate vector index choice", min_score=0.0
    )
    hit = next(r for r in results if r.note.id == hidden.id)
    assert hit.path == "neighbor"
    assert hit.signals["lexical"] == 0 and hit.signals["semantic"] == 0

    flat = Recaller(store, embedder=embedder).recall(
        "approximate vector index choice", expand_graph=False, min_score=0.0
    )
    assert hidden.id not in {r.note.id for r in flat}


def test_weights_are_normalized():
    w = Weights(lexical=2, semantic=2, graph=2, strength=2, recency=2).normalized()
    assert abs(w.total() - 1.0) < 1e-9


def test_zero_weights_fall_back_to_defaults():
    w = Weights(0, 0, 0, 0, 0).normalized()
    assert abs(w.total() - 1.0) < 1e-9


def test_gap_terms_name_what_is_missing(populated, recaller):
    results = recaller.recall("lexical search")
    gaps = top_gap_terms("lexical search with kubernetes autoscaling", results)
    assert "kubernetes" in gaps
    assert "lexical" not in gaps
