"""Stage 1 - capture the input.

Turns anything you point at into a ``Source`` record: pasted text, a local
file, a web page, or a whole folder of markdown. Capture never interprets;
it only normalizes and de-duplicates, so re-capturing the same bytes is a
no-op and the connect stage can be re-run at will.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from pathlib import Path

from .models import Source, checksum

TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".rst", ".org",
    ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".html", ".csv", ".sql", ".sh",
}
MAX_BYTES = 2_000_000
USER_AGENT = "jarvis-second-brain/1.0"


class CaptureError(RuntimeError):
    pass


def capture_text(text: str, title: str = "", uri: str = "") -> Source:
    """Capture a literal string — a voice note, a paste, an agent's own output."""
    text = text.strip()
    if not text:
        raise CaptureError("nothing to capture: empty text")
    return Source(
        uri=uri or f"text:{checksum(text)[:12]}",
        kind="text",
        title=title or _first_line(text),
        raw_text=text,
    )


def capture_file(path: str | Path) -> Source:
    """Capture one local file."""
    p = Path(path).expanduser()
    if not p.exists():
        raise CaptureError(f"no such file: {p}")
    if p.is_dir():
        raise CaptureError(f"{p} is a directory — use capture_vault()")
    if p.suffix.lower() not in TEXT_SUFFIXES:
        raise CaptureError(f"unsupported file type {p.suffix or '(none)'} for {p.name}")
    if p.stat().st_size > MAX_BYTES:
        raise CaptureError(f"{p.name} is larger than {MAX_BYTES} bytes")
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise CaptureError(f"{p.name} is empty")
    return Source(
        uri=f"file://{p.resolve()}",
        kind="file",
        title=_title_from_markdown(text) or p.stem,
        raw_text=text,
        meta={"suffix": p.suffix.lower(), "bytes": p.stat().st_size},
    )


def capture_url(url: str, timeout: float = 20.0) -> Source:
    """Fetch a page and reduce it to readable text."""
    if not url.startswith(("http://", "https://")):
        raise CaptureError(f"not an http(s) url: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(MAX_BYTES)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CaptureError(f"could not fetch {url}: {exc}") from exc
    encoding = "utf-8"
    if "charset=" in ctype:
        encoding = ctype.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    raw = body.decode(encoding, errors="replace")
    title = _html_title(raw)
    text = strip_html(raw) if "html" in ctype or raw.lstrip().startswith("<") else raw
    text = text.strip()
    if not text:
        raise CaptureError(f"{url} yielded no readable text")
    return Source(
        uri=url,
        kind="url",
        title=title or url,
        raw_text=text,
        meta={"content_type": ctype},
    )


def capture_vault(directory: str | Path, pattern: str = "**/*.md", limit: int = 500) -> list[Source]:
    """Capture a folder of notes — an Obsidian vault, a docs/ tree, a wiki export."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise CaptureError(f"not a directory: {root}")
    sources: list[Source] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or len(sources) >= limit:
            continue
        try:
            sources.append(capture_file(path))
        except CaptureError:
            continue
    if not sources:
        raise CaptureError(f"no capturable files in {root} matching {pattern}")
    return sources


def capture(target: str, title: str = "") -> list[Source]:
    """Dispatch on what the target looks like. One entry point for the CLI."""
    if target.startswith(("http://", "https://")):
        return [capture_url(target)]
    path = _as_path(target)
    if path is not None:
        if path.is_dir():
            return capture_vault(path)
        if path.is_file():
            return [capture_file(path)]
        if _looks_like_path(target.strip()):
            # Storing the literal string "notes/meeting.md" as knowledge would
            # quietly fill the brain with typos instead of reporting them.
            raise CaptureError(f"no such file or directory: {target.strip()}")
    return [capture_text(target, title=title)]


def _looks_like_path(candidate: str) -> bool:
    """True when the input was clearly meant as a location, not as prose."""
    if " " in candidate or "\t" in candidate:
        return False
    return "/" in candidate or "\\" in candidate or Path(candidate).suffix.lower() in TEXT_SUFFIXES


def _as_path(target: str) -> Path | None:
    """Interpret the target as a filesystem path, or decide it is prose.

    Multi-line or very long strings are prose by definition, and the check has
    to come before touching the filesystem: os.stat() on an over-long name
    raises rather than reporting "not found".
    """
    candidate = target.strip()
    if not candidate or "\n" in candidate or "\x00" in candidate:
        return None
    if len(candidate) > 255 or len(candidate.encode("utf-8")) > 255:
        return None
    try:
        path = Path(candidate).expanduser()
        path.exists()
    except (OSError, ValueError, RuntimeError):
        return None
    return path


# --- text reduction --------------------------------------------------------

_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg|head)\b.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANKS_RE = re.compile(r"\n{3,}")
_BLOCK_END_RE = re.compile(r"</(p|div|li|tr|h[1-6]|section|article|blockquote)>", re.I)
_BR_RE = re.compile(r"<(br|hr)\s*/?>", re.I)


def strip_html(raw: str) -> str:
    """Drop markup but keep block structure, so paragraphs survive."""
    text = _SCRIPT_RE.sub(" ", raw)
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_END_RE.sub("\n\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return _BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()


def _html_title(raw: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
    return html.unescape(_TAG_RE.sub("", match.group(1))).strip() if match else ""


def _title_from_markdown(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            break
    return ""


def _first_line(text: str, limit: int = 80) -> str:
    line = text.strip().splitlines()[0].strip().lstrip("#").strip()
    return line[:limit] + ("..." if len(line) > limit else "")
