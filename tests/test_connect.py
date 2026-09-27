import pytest

from brain.capture import capture_text
from brain.connect import (
    extract_claim_texts,
    keywords_of,
    split_into_chunks,
)


def test_split_by_heading(latency_doc):
    chunks = split_into_chunks(latency_doc)
    titles = [c.title for c in chunks]
    assert "p99 latency budget" in titles
    assert "What we shipped" in titles


def test_short_fragments_are_merged_forward():
    text = "One.\n\nTwo.\n\nThree.\n\nFour."
    chunks = split_into_chunks(text, min_chars=200, max_chars=1000)
    assert len(chunks) == 1


def test_oversized_text_is_cut():
    text = "This sentence is exactly the kind of filler used to pad a chunk. " * 60
    chunks = split_into_chunks(text, min_chars=100, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_single_unbreakable_sentence_is_still_cut():
    chunks = split_into_chunks("x" * 3000, min_chars=100, max_chars=400)
    assert all(len(c.text) <= 400 for c in chunks)


def test_empty_text_yields_no_chunks():
    assert split_into_chunks("   \n\n  ") == []


def test_keywords_drop_stopwords_and_short_tokens():
    kw = keywords_of("The retrieval step is the one that takes most of the latency budget")
    assert "the" not in kw and "is" not in kw
    assert "retrieval" in kw and "latency" in kw


def test_keywords_are_deterministic():
    text = "recall and retrieval and recall again with vectors"
    assert keywords_of(text) == keywords_of(text)


def test_claims_are_assertive_and_unhedged():
    body = (
        "The p99 latency budget for the search endpoint is 180 milliseconds. "
        "Maybe we should consider caching the whole index in memory somewhere. "
        "Should we rewrite the retrieval layer from scratch this quarter? "
        "The approximate index reduced the p99 to 95 milliseconds in production."
    )
    claims = extract_claim_texts(body)
    joined = " ".join(claims)
    assert "180 milliseconds" in joined
    assert "Maybe" not in joined     # hedged
    assert "Should we" not in joined  # a question


def test_connect_creates_notes_claims_and_vectors(store, connector, latency_doc):
    source = store.add_source(capture_text(latency_doc))
    result = connector.connect(source)
    assert len(result.notes) == 3
    assert result.claims
    for note in result.notes:
        assert store.get_vector(note.id) is not None


def test_wikilink_creates_a_stub_and_links_to_it(store, connector):
    text = (
        "# Retrieval design\n"
        "Our reranking approach is borrowed from [[Hindsight]], which reorders candidates "
        "after the first pass instead of trusting the initial vector ordering at all.\n"
    )
    result = connector.connect(store.add_source(capture_text(text)))
    assert [n.title for n in result.stubs] == ["Hindsight"]
    assert any(l.origin == "wikilink" for l in result.links)
    # The body reads as prose; the brackets are not left in it.
    assert "[[" not in result.notes[0].body


def test_wikilink_reuses_an_existing_note(store, connector):
    first = connector.connect(
        store.add_source(capture_text("# Hindsight\nReranking after the first retrieval pass."))
    )
    second = connector.connect(
        store.add_source(
            capture_text("# Design\nWe adopted the [[Hindsight]] method for the second pass here.")
        )
    )
    assert second.stubs == []
    targets = {l.dst for l in second.links if l.origin == "wikilink"}
    assert first.notes[0].id in targets


def test_related_documents_link_across_sources(store, connector):
    connector.connect(store.add_source(capture_text(
        "# Hybrid recall\nRecall blends lexical search with vector similarity, because "
        "lexical alone misses paraphrase and vectors alone miss exact identifiers."
    )))
    second = connector.connect(store.add_source(capture_text(
        "# Why vectors are not enough\nVector similarity handles paraphrase but misses exact "
        "identifiers, so recall has to blend lexical search with vectors."
    )))
    assert any(l.origin == "overlap" for l in second.links)


def test_unrelated_documents_do_not_link(store, connector):
    connector.connect(store.add_source(capture_text(
        "# Hybrid recall\nRecall blends lexical search with vector similarity for paraphrase."
    )))
    second = connector.connect(store.add_source(capture_text(
        "# Office plants\nThe fern by the window needs watering twice a week in summer."
    )))
    assert not [l for l in second.links if l.origin == "overlap"]


def test_tags_are_captured(store, connector):
    result = connector.connect(store.add_source(capture_text(
        "# Tagged\nThis note is about retrieval quality and ranking. #retrieval #memory"
    )))
    assert set(result.notes[0].tags) == {"retrieval", "memory"}


def test_reconnecting_the_same_source_is_idempotent_on_sources(store, connector, latency_doc):
    source = capture_text(latency_doc)
    a = store.add_source(source)
    b = store.add_source(capture_text(latency_doc))
    assert a.id == b.id
    assert store.count("sources") == 1


def test_link_scoring_separates_related_from_unrelated(store, connector, embedder):
    """The inferred-link threshold must sit inside a real gap, not be a guess."""
    from brain.connect import _overlap, keywords_of
    from brain.embedding import calibrate, cosine

    def blended(a: str, b: str) -> float:
        semantic = calibrate(cosine(embedder.embed(a), embedder.embed(b)), embedder)
        lexical = _overlap(set(keywords_of(a)), set(keywords_of(b)))
        return 0.6 * semantic + 0.4 * lexical

    related = blended(
        "We shipped an approximate index with a candidate cutoff and the p99 dropped to 95 ms.",
        "Reranking the top candidates after the first retrieval pass recovers the recall an "
        "approximate index gives up.",
    )
    unrelated = blended(
        "We shipped an approximate index with a candidate cutoff for the search endpoint.",
        "The fern by the office window needs watering twice a week during the summer.",
    )
    # The hard case: two unrelated notes that happen to share the word "first".
    incidental = blended(
        "Reranking the top candidates after the first retrieval pass recovers recall.",
        "The office coffee machine is descaled on the first Monday of each month.",
    )
    assert unrelated < connector.link_threshold <= related
    assert incidental < connector.link_threshold


def test_a_related_second_document_links_to_the_first(store, connector):
    connector.connect(store.add_source(capture_text(
        "# What we shipped\nWe shipped an approximate index with a candidate cutoff. The p99 "
        "dropped to 95 ms and recall fell by about two percent, which the team accepted."
    )))
    second = connector.connect(store.add_source(capture_text(
        "# Reranking after the first pass\nReranking the top candidates after the first "
        "retrieval pass recovers most of the recall that an approximate index gives up."
    )))
    assert any(l.origin == "overlap" for l in second.links)


@pytest.mark.parametrize("sentence", [
    "The p99 latency budget for the search endpoint is 180 milliseconds in production.",
    "Strength decays with a half life so that stale context stops dominating recall.",
    "The approximate index reduces the p99 from 180 milliseconds down to 95 milliseconds.",
    "A second brain must get sharper after every session, otherwise it is just a folder.",
])
def test_these_are_claims(sentence):
    assert extract_claim_texts(sentence) == [sentence]


@pytest.mark.parametrize("sentence", [
    "```bash pip install pytest && python -m pytest tests/ -q ``` 142 tests, no network",
    "| test_a_full_loop | The loop improves the brain, measurably | and more text here |",
    "Run Ollama for real semantic recall. - **Under the fallback, links are not inferable**",
    "The function get_brain() returns a Brain and is memoised across the whole process.",
    "Maybe we should consider caching the whole index in memory somewhere next quarter.",
    "Should we rewrite the whole retrieval layer from scratch during this quarter?",
])
def test_these_are_not_claims(sentence):
    assert extract_claim_texts(sentence) == []


def test_a_captured_markdown_file_does_not_donate_its_code_blocks(store, connector, tmp_path):
    doc = tmp_path / "guide.md"
    doc.write_text(
        "# Setup guide\n"
        "The default database path is one file on disk, and it is portable between machines.\n"
        "\n"
        "```bash\n"
        "pip install pytest && python -m pytest tests/ -q\n"
        "python -m brain status --db ~/.jarvis/brain.db\n"
        "```\n"
        "\n"
        "| Component | Weight |\n"
        "| --- | --- |\n"
        "| volume | 0.15 |\n"
    )
    from brain.capture import capture_file

    result = connector.connect(store.add_source(capture_file(doc)))
    for claim in result.claims:
        assert "```" not in claim.text
        assert "pip install" not in claim.text
        assert "|" not in claim.text
