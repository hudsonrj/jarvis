"""Vectors for semantic recall.

Two backends, same interface:

* ``OllamaEmbedder`` calls the local Ollama server, so it uses the same
  runtime Jarvis already needs for its LLM.
* ``HashEmbedder`` needs nothing at all. It projects a bag of character
  n-grams into a fixed space, which is weaker than a real model but is
  deterministic, offline, and good enough to catch paraphrases that the
  lexical index misses.

``get_embedder()`` picks Ollama when it answers, and falls back silently so
the brain never hard-depends on a running server.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .text import content_tokens, stem

DIM = 256
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("BRAIN_EMBED_MODEL", "nomic-embed-text")


class Embedder(Protocol):
    name: str
    #: Cosine value below which a match is noise for this backend.
    sim_floor: float
    #: Cosine value at or above which a match is as good as this backend gets.
    sim_ceiling: float

    def embed(self, text: str) -> list[float]: ...


def calibrate(sim: float, embedder: "Embedder") -> float:
    """Map a raw cosine into 0..1 using the backend's own useful range.

    Backends differ widely in scale: a 0.45 cosine is a strong hit for the
    hashed fallback and mediocre for a trained model. Calibrating here keeps
    the recall weights meaningful whichever backend is in use.
    """
    floor = getattr(embedder, "sim_floor", 0.15)
    ceiling = getattr(embedder, "sim_ceiling", 0.90)
    if ceiling <= floor:
        return max(0.0, min(1.0, sim))
    return max(0.0, min(1.0, (sim - floor) / (ceiling - floor)))


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return list(vec)
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class HashEmbedder:
    """Hashed word, stem and character-trigram features. No network, no models.

    Stopwords are excluded. Left in, they give any two texts in the same
    language a similarity floor, and a nonsense query then looks like a weak
    match for everything in the brain.
    """

    name = "hash-trigram-256"
    # Hashed features saturate early: ~0.65 cosine is about as close as two
    # genuinely different wordings get.
    sim_floor = 0.12
    sim_ceiling = 0.65

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = content_tokens(text)
        if not words:
            return vec
        for word in words:
            self._add(vec, f"w:{word}", 1.0)
            # The stem carries the match across inflections.
            self._add(vec, f"s:{stem(word)}", 0.7)
            # Trigrams only inside content words, and weighted low: they are a
            # fallback for typos and compounds, not the main signal.
            padded = f" {word} "
            for i in range(len(padded) - 2):
                self._add(vec, f"t:{padded[i:i + 3]}", 0.25)
        # Sublinear damping keeps long notes from dominating cosine scores.
        vec = [math.copysign(math.log1p(abs(v)), v) for v in vec]
        return l2_normalize(vec)

    def _add(self, vec: list[float], token: str, weight: float) -> None:
        h = _fnv1a(token)
        idx = h % self.dim
        sign = 1.0 if (h >> 32) & 1 else -1.0
        vec[idx] += sign * weight


class OllamaEmbedder:
    """Embeddings from a local Ollama server."""

    # Trained embedding models put unrelated text around 0.3-0.5, so the useful
    # band starts much higher than for the hashed fallback. These two numbers are
    # an estimate for a general-purpose model, not a measurement: if recall links
    # too much or too little with your model, measure a handful of known-related
    # and known-unrelated pairs and move the floor to sit between them.
    sim_floor = 0.30
    sim_ceiling = 0.92

    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST, timeout: float = 20.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # Localhost must not go through the outbound proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        vec = body.get("embedding") or []
        if not vec:
            raise ValueError(f"empty embedding from {self.name}")
        return l2_normalize(vec)

    def available(self) -> bool:
        try:
            self.embed("ping")
            return True
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError):
            return False


_cached: Embedder | None = None


def get_embedder(prefer_ollama: bool | None = None, force_hash: bool = False) -> Embedder:
    """Return a usable embedder, preferring Ollama when it responds."""
    global _cached
    if force_hash:
        return HashEmbedder()
    if _cached is not None:
        return _cached
    if prefer_ollama is None:
        prefer_ollama = os.getenv("BRAIN_USE_OLLAMA_EMBED", "1") != "0"
    if prefer_ollama:
        candidate = OllamaEmbedder()
        if candidate.available():
            _cached = candidate
            return _cached
    _cached = HashEmbedder()
    return _cached


def reset_embedder() -> None:
    global _cached
    _cached = None


def _fnv1a(text: str) -> int:
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8", "replace"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
