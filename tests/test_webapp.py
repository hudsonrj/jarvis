"""The web app: the service layer, and a real HTTP round trip."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from brain import Brain
from brain.webapp import BadRequest, BrainService, NotFound, serve


@pytest.fixture
def service(embedder, latency_doc):
    # same_thread=False matches how the CLI opens it for `brain serve`.
    brain = Brain(":memory:", embedder=embedder, same_thread=False)
    brain.learn_from(latency_doc)
    yield BrainService(brain)
    brain.close()


def test_serving_a_single_thread_brain_fails_immediately(embedder):
    """Better a clear error at startup than a sqlite crash under load."""
    brain = Brain(":memory:", embedder=embedder)
    try:
        with pytest.raises(ValueError, match="same_thread=False"):
            BrainService(brain)
    finally:
        brain.close()


# --- service ------------------------------------------------------------


def test_state_reports_sharpness_and_the_next_move(service):
    s = service.state()
    assert 0 <= s["sharpness"] <= 100
    assert set(s["components"]) == {
        "volume", "linkage", "connectedness", "verification", "trust"
    }
    assert s["weakest"] in s["components"]
    assert s["next_move"]
    assert s["embedder"] == "hash-trigram-256"


def test_ask_returns_cited_hits_with_their_signals(service):
    r = service.ask("what is our p99 latency budget")
    assert r["recalls"]
    hit = r["recalls"][0]
    assert set(hit["signals"]) == {"lexical", "semantic", "graph", "strength", "recency"}
    assert hit["body"] and hit["title"]
    assert "never fill the gap with invention" in r["prompt"]


def test_ask_on_an_unknown_topic_returns_no_hits(service):
    assert service.ask("kubernetes pod autoscaling thresholds")["recalls"] == []


def test_ask_rejects_an_empty_question(service):
    with pytest.raises(BadRequest):
        service.ask("   ")


def test_capture_reports_the_change_in_sharpness(service):
    r = service.capture(
        "# Reranking\nReranking the top candidates after the first retrieval pass "
        "recovers most of the recall an approximate index gives up."
    )
    assert r["summary"]["notes"] >= 1
    assert "delta" in r and "sharpness" in r


def test_capture_rejects_a_missing_file(service):
    with pytest.raises(BadRequest, match="no such file"):
        service.capture("notes/definitely-missing.md")


def test_capture_rejects_an_empty_target(service):
    with pytest.raises(BadRequest):
        service.capture("  ")


def test_feedback_adjusts_the_cited_notes(service):
    brief = service.ask("what is our p99 latency budget")
    r = service.feedback(brief["brief_id"], "helped")
    assert r["adjusted"]
    assert all(n["wins"] >= 1 for n in r["adjusted"])


def test_feedback_on_an_unknown_brief_is_rejected(service):
    with pytest.raises(BadRequest):
        service.feedback("brief_nope", "helped")


def test_verify_records_a_verdict_and_its_evidence(service):
    claim = service.claims()["claims"][0]
    r = service.verify(claim["id"], "supported", "grafana dashboard")
    assert r["claim"]["status"] == "supported"
    assert "grafana dashboard" in r["claim"]["evidence"]


def test_verify_rejects_an_invalid_status(service):
    claim = service.claims()["claims"][0]
    with pytest.raises(BadRequest):
        service.verify(claim["id"], "probably")


def test_verify_an_unknown_claim_is_not_found(service):
    with pytest.raises(NotFound):
        service.verify("claim_nope", "supported")


def test_note_detail_carries_links_claims_and_history(service):
    note_id = service.notes()["notes"][0]["id"]
    service.correct(note_id, "A corrected body that supersedes what the source said.", "changed")
    detail = service.note(note_id)
    assert detail["note"]["revision"] == 2
    assert detail["history"][0]["reason"] == "changed"
    assert "before" in detail["history"][0] and "after" in detail["history"][0]
    assert isinstance(detail["links"], list)


def test_note_without_an_id_is_rejected(service):
    with pytest.raises(BadRequest):
        service.note("")


def test_note_that_does_not_exist_is_not_found(service):
    with pytest.raises(NotFound):
        service.note("note_nope")


def test_correct_rejects_empty_text(service):
    note_id = service.notes()["notes"][0]["id"]
    with pytest.raises(BadRequest):
        service.correct(note_id, "   ")


def test_notes_can_be_searched(service):
    hits = service.notes(query="latency budget")["notes"]
    assert hits
    assert any("latency" in n["title"].lower() or "latency" in n["excerpt"].lower() for n in hits)


def test_graph_edges_only_reference_returned_nodes(service):
    g = service.graph()
    ids = {n["id"] for n in g["nodes"]}
    assert g["nodes"]
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_graph_reports_each_pair_once(service):
    g = service.graph()
    pairs = [tuple(sorted((e["source"], e["target"]))) for e in g["edges"]]
    assert len(pairs) == len(set(pairs))


def test_remember_and_link(service):
    a = service.remember("Manual note", "Written straight into the brain to test linking.")["note"]
    b = service.notes()["notes"][1]
    assert service.link(a["id"], b["id"], "extends")["ok"]
    with pytest.raises(BadRequest):
        service.link(a["id"], a["id"])


def test_service_survives_concurrent_calls(service):
    """The handler runs one thread per request; the lock has to hold."""
    errors: list[Exception] = []

    def work(i: int) -> None:
        try:
            service.ask("latency budget")
            service.state()
            service.capture(f"Concurrency probe number {i} about the retrieval budget and its cost.")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --- HTTP ---------------------------------------------------------------


@pytest.fixture
def server(embedder, latency_doc):
    brain = Brain(":memory:", embedder=embedder, same_thread=False)
    brain.learn_from(latency_doc)
    httpd = serve(brain, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    brain.close()


def get(base: str, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(base: str, path: str, payload: dict | str) -> tuple[int, dict]:
    raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=raw, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_page_is_served(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        body = r.read().decode()
    assert r.status == 200
    assert "<title>Second Brain</title>" in body
    assert "/api/state" in body


def test_api_state_over_http(server):
    status, body = get(server, "/api/state")
    assert status == 200
    assert "sharpness" in body


def test_ask_and_feedback_over_http(server):
    status, brief = post(server, "/api/ask", {"query": "what is our p99 latency budget"})
    assert status == 200 and brief["recalls"]
    status, fb = post(server, "/api/feedback", {"brief_id": brief["brief_id"], "verdict": "helped"})
    assert status == 200 and fb["adjusted"]


def test_bad_request_is_reported_as_400(server):
    status, body = post(server, "/api/ask", {"query": ""})
    assert status == 400
    assert "error" in body


def test_malformed_json_is_reported(server):
    status, body = post(server, "/api/ask", "this is not json")
    assert status == 400
    assert body["error"] == "body must be JSON"


def test_a_json_array_body_is_rejected(server):
    status, body = post(server, "/api/ask", "[1, 2, 3]")
    assert status == 400


def test_unknown_routes_are_404(server):
    assert get(server, "/api/nope")[0] == 404
    assert post(server, "/api/nope", {})[0] == 404


def test_path_traversal_is_refused(server):
    for attempt in ["/../brain/store.py", "/..%2f..%2fetc%2fpasswd", "/web/../../main.py"]:
        try:
            with urllib.request.urlopen(server + attempt, timeout=10) as r:
                assert r.status == 404, attempt
        except urllib.error.HTTPError as e:
            assert e.code == 404, attempt


def test_favicon_is_answered_not_missing(server):
    with urllib.request.urlopen(server + "/favicon.ico", timeout=10) as r:
        assert r.status == 204
