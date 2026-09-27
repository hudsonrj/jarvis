"""Jarvis second brain — capture, connect, remember, act, correct.

    from brain import Brain

    with Brain(profile="researcher") as b:
        with b.session("monday") as ses:
            b.learn_from("notes/latency.md")
            brief = b.brief("what do we know about p99 latency")
            print(brief.context)
            b.helped(brief)
        print(ses.delta)
"""

from .act import Briefer, as_prompt
# ``capture`` itself is not re-exported: that name must keep pointing at the
# brain.capture module, not at the dispatcher inside it. Use ``capture_any``.
from .capture import (
    CaptureError,
    capture_file,
    capture_text,
    capture_url,
    capture_vault,
)
from .capture import capture as capture_any
from .connect import Connector, keywords_of
from .embedding import HashEmbedder, OllamaEmbedder, cosine, get_embedder
from .feedback import Learner
from .models import Brief, Claim, Correction, Link, Note, Recall, Session, Source
from .pipeline import PRESETS, Brain
from .recall import Recaller, Weights, coverage, explain
from .registry import BUILDS, PROJECTS, STAGES, describe_loop
from .sharpness import Sharpness, measure, next_move
from .store import Store

__version__ = "1.0.0"

__all__ = [
    "Brain", "Store", "Briefer", "Connector", "Learner", "Recaller", "Weights",
    "Brief", "Claim", "Correction", "Link", "Note", "Recall", "Session", "Source",
    "Sharpness", "measure", "next_move", "coverage", "explain", "as_prompt",
    "capture_any", "capture_file", "capture_text", "capture_url", "capture_vault",
    "CaptureError", "keywords_of", "cosine", "get_embedder", "HashEmbedder",
    "OllamaEmbedder", "PRESETS", "BUILDS", "PROJECTS", "STAGES", "describe_loop",
    "__version__",
]
