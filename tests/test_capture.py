import pytest

from brain.capture import (
    CaptureError,
    capture,
    capture_file,
    capture_text,
    capture_vault,
    strip_html,
)


def test_capture_text_sets_digest_and_title():
    s = capture_text("# Title here\nSome body text about the loop.")
    assert s.kind == "text"
    assert s.title == "Title here"
    assert len(s.digest) == 64


def test_capture_text_rejects_empty():
    with pytest.raises(CaptureError):
        capture_text("   \n  ")


def test_identical_text_yields_identical_digest():
    assert capture_text("same bytes").digest == capture_text("same bytes").digest


def test_capture_file_reads_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Heading\nBody of the note.")
    s = capture_file(f)
    assert s.kind == "file"
    assert s.title == "Heading"
    assert "Body of the note." in s.raw_text


def test_capture_file_rejects_unsupported_suffix(tmp_path):
    f = tmp_path / "photo.jpeg"
    f.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(CaptureError, match="unsupported"):
        capture_file(f)


def test_capture_file_rejects_directory(tmp_path):
    with pytest.raises(CaptureError, match="directory"):
        capture_file(tmp_path)


def test_capture_vault_walks_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# A\n" + "content a " * 20)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\n" + "content b " * 20)
    (tmp_path / "skip.jpeg").write_bytes(b"\xff")
    titles = {s.title for s in capture_vault(tmp_path)}
    assert titles == {"A", "B"}


def test_capture_vault_rejects_empty_dir(tmp_path):
    with pytest.raises(CaptureError):
        capture_vault(tmp_path)


def test_strip_html_keeps_block_structure():
    out = strip_html(
        "<html><head><title>T</title><style>b{color:red}</style></head>"
        "<body><p>First &amp; best</p><li>One</li><li>Two</li>"
        "<script>evil()</script></body></html>"
    )
    assert "color:red" not in out
    assert "evil()" not in out
    assert out.splitlines()[0] == "First & best"
    assert "One" in out and "Two" in out


def test_dispatch_does_not_stat_long_prose():
    """A long string is prose; os.stat() on it would raise, not return False."""
    prose = "This is a sentence about the loop. " * 40
    assert capture(prose)[0].kind == "text"


def test_dispatch_handles_multiline_text():
    assert capture("line one\nline two that is long enough to matter")[0].kind == "text"


def test_dispatch_finds_a_real_file(tmp_path, monkeypatch):
    f = tmp_path / "x.md"
    f.write_text("# X\nsome content here")
    monkeypatch.chdir(tmp_path)
    assert capture("x.md")[0].kind == "file"
