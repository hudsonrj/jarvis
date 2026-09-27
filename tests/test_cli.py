"""The command line is the other half of the interface; it must not rot."""

import pytest

from brain.cli import main


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "brain.db")


def run(db, *args):
    return main(["--db", db, "--hash-embed", *args])


def test_loop_lists_the_projects(capsys):
    assert main(["loop"]) == 0
    out = capsys.readouterr().out
    assert "sage-wiki" in out and "MateClaw" in out


def test_status_on_an_empty_brain(db, capsys):
    assert run(db, "status") == 0
    out = capsys.readouterr().out
    assert "sharpness 0.0/100" in out
    assert "capture more sources" in out


def test_learn_then_ask(db, capsys):
    assert run(db, "learn", "# Index choice\nWe picked an approximate vector index over an "
                            "exact scan because the exact scan blew the latency budget.") == 0
    assert "notes=" in capsys.readouterr().out
    assert run(db, "ask", "why the approximate index") == 0
    out = capsys.readouterr().out
    assert "approximate" in out
    assert "brief" in out


def test_ask_prints_a_prompt_block(db, capsys):
    run(db, "learn", "# Budget\nThe p99 latency budget for the search endpoint is 180 ms.")
    capsys.readouterr()
    run(db, "ask", "--prompt", "latency budget")
    assert "never fill the gap with invention" in capsys.readouterr().out


def test_ask_with_an_empty_brain_says_so(db, capsys):
    assert run(db, "ask", "anything at all") == 0
    assert "holds nothing relevant" in capsys.readouterr().out


def test_learn_reports_a_capture_error(db, capsys):
    assert run(db, "learn", "notes/definitely-missing.md") == 1
    assert "no such file" in capsys.readouterr().err


def test_why_shows_the_weights(db, capsys):
    run(db, "learn", "# Budget\nThe p99 latency budget for the search endpoint is 180 ms.")
    capsys.readouterr()
    run(db, "why", "latency budget")
    assert "weights:" in capsys.readouterr().out


def test_claims_and_verify_round_trip(db, capsys):
    run(db, "learn", "# Budget\nThe p99 latency budget for the search endpoint is 180 "
                     "milliseconds, and the retrieval step consumes nearly all of it.")
    capsys.readouterr()
    run(db, "claims", "--open")
    claim_id = capsys.readouterr().out.split()[0]
    assert run(db, "verify", claim_id, "supported", "--evidence", "dashboard") == 0
    assert "supported" in capsys.readouterr().out


def test_verify_an_unknown_claim_fails(db, capsys):
    assert run(db, "verify", "claim_nope", "supported") == 1
    assert "no claim" in capsys.readouterr().err


def test_feedback_round_trip(db, capsys):
    run(db, "learn", "# Budget\nThe p99 latency budget for the search endpoint is 180 ms.")
    capsys.readouterr()
    run(db, "ask", "latency budget")
    brief_id = [w for w in capsys.readouterr().out.split() if w.startswith("brief_")][0]
    assert run(db, "feedback", brief_id, "helped") == 0
    assert "adjusted" in capsys.readouterr().out


def test_feedback_on_an_unknown_brief_fails(db, capsys):
    assert run(db, "feedback", "brief_nope", "helped") == 1
    assert "no brief" in capsys.readouterr().err


def test_remember_and_correct(db, capsys):
    run(db, "remember", "Deployment", "Jarvis runs on the office mini PC for now.")
    note_id = capsys.readouterr().out.split()[0]
    assert run(db, "correct", note_id, "Jarvis runs on a Raspberry Pi 5.", "--reason", "moved") == 0
    assert "corrected" in capsys.readouterr().out


def test_correct_an_unknown_note_fails(db, capsys):
    assert run(db, "correct", "note_nope", "text") == 1
    assert "no such note" in capsys.readouterr().err


def test_notes_and_orphans(db, capsys):
    run(db, "remember", "Alone", "A note nothing else points at yet.")
    capsys.readouterr()
    run(db, "notes")
    assert "Alone" in capsys.readouterr().out
    run(db, "orphans")
    assert "Alone" in capsys.readouterr().out


def test_session_flag_reports_the_delta(db, capsys):
    assert run(db, "--session", "cli run", "learn",
               "# Budget\nThe p99 latency budget for the search endpoint is 180 ms.") == 0
    out = capsys.readouterr().out
    assert "session cli run: sharpness" in out
    assert "->" in out


def test_trend_after_a_session(db, capsys):
    run(db, "--session", "one", "learn", "# A\nThe latency budget is 180 milliseconds today.")
    capsys.readouterr()
    run(db, "trend")
    assert "net over" in capsys.readouterr().out


def test_decay_and_reconnect_run(db, capsys):
    run(db, "learn", "# A\nThe latency budget for the search endpoint is 180 milliseconds.")
    capsys.readouterr()
    assert run(db, "decay", "--half-life", "30") == 0
    assert "decayed" in capsys.readouterr().out
    assert run(db, "reconnect") == 0
    assert "re-derived" in capsys.readouterr().out


def test_an_unknown_profile_is_rejected(db, capsys):
    with pytest.raises(SystemExit):
        main(["--db", db, "--profile", "nonsense", "status"])


def test_build_profiles_are_accepted(db):
    for profile in ("creator", "researcher", "team"):
        assert main(["--db", db, "--hash-embed", "--profile", profile, "status"]) == 0
