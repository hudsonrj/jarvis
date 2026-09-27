import os
import sys
from pathlib import Path

# Tests must never reach for a model server: the hashed embedder is deterministic.
os.environ["BRAIN_USE_OLLAMA_EMBED"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from brain import Brain
from brain.connect import Connector
from brain.embedding import HashEmbedder
from brain.recall import Recaller
from brain.store import Store


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def connector(store, embedder):
    return Connector(store, embedder=embedder)


@pytest.fixture
def recaller(store, embedder):
    return Recaller(store, embedder=embedder)


@pytest.fixture
def brain(embedder):
    b = Brain(":memory:", embedder=embedder)
    yield b
    b.close()


@pytest.fixture
def latency_doc():
    return (
        "# p99 latency budget\n"
        "Our p99 latency budget for the search endpoint is 180 milliseconds. The retrieval\n"
        "step takes most of it because the vector scan is linear over every stored note.\n"
        "\n"
        "## The fix we rejected\n"
        "We considered caching the whole index in memory. That was rejected because the\n"
        "index is 40 GB and the box has only 16 GB of RAM, so it does not fit.\n"
        "\n"
        "## What we shipped\n"
        "We shipped an approximate index with a candidate cutoff. The p99 dropped to 95 ms\n"
        "and recall fell by about two percent, which the product team accepted.\n"
    )
