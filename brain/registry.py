"""The 20 projects, mapped onto the four stages of the loop.

Why a registry instead of 20 integrations? Because the projects in the source
list are interchangeable within a stage. Each one fills a *slot*: something
that captures, something that links, something that remembers, something that
acts. This module records which slot each project fills, and ``brain/`` ships a
working reference adapter for every slot, so the loop runs today and any entry
here can replace a stage without touching the others.

Honesty note on the URLs: the list this was built from had most paths cut off
mid-slug (``github.com/chubbyguan/chu…``). Each entry records the owner and the
visible fragment, and ``url_complete`` says whether the path is the full one.
Nothing here is a guessed repository name.
"""

from __future__ import annotations

from dataclasses import dataclass

CAPTURE = "capture"
CONNECT = "connect"
REMEMBER = "remember"
ACT = "act"

STAGES = {
    CAPTURE: "Capture the input — get the source in, normalized and de-duplicated.",
    CONNECT: "Connect the ideas — split into atomic notes, link them, make claims checkable.",
    REMEMBER: "Remember what matters — rank by relevance and by what proved useful.",
    ACT: "Put it to work — pack cited context for an agent, then take the correction back.",
}


@dataclass(frozen=True)
class Project:
    n: int
    name: str
    stage: str
    owner: str
    path_fragment: str
    url_complete: bool
    slot: str
    #: Which module in this package covers the same slot.
    local_adapter: str

    @property
    def url(self) -> str:
        if self.url_complete:
            return f"https://github.com/{self.owner}/{self.path_fragment}"
        return f"https://github.com/{self.owner}  (path in source list was truncated: {self.path_fragment}…)"


PROJECTS: tuple[Project, ...] = (
    # --- capture the input ---
    Project(1, "Chubby Skills", CAPTURE, "chubbyguan", "chu", False,
            "skill packs as capturable sources", "brain.capture.capture_vault"),
    Project(2, "OpenWiki", CAPTURE, "kdsz001", "OpenWi", False,
            "wiki export ingestion", "brain.capture.capture_vault"),
    Project(3, "Agent Second Brain", CAPTURE, "smixs", "agent-se", False,
            "agent-side capture of its own output", "brain.capture.capture_text"),
    Project(4, "DocMason", CAPTURE, "JetXu-LLM", "DocM", False,
            "document set construction", "brain.capture.capture_file"),
    Project(5, "DocsAgent", CAPTURE, "docsagent", "docs", False,
            "docs crawl and normalize", "brain.capture.capture_url"),
    # --- connect the ideas ---
    Project(6, "claude-obsidian", CONNECT, "AgriciDaniel", "c", False,
            "markdown vault as the graph substrate", "brain.connect.Connector"),
    Project(7, "SwarmVault", CONNECT, "swarmclawai", "sw", False,
            "multi-agent shared vault", "brain.connect.Connector"),
    Project(8, "sage-wiki", CONNECT, "xoai", "sage-wiki", True,
            "wiki-style bidirectional linking", "brain.connect.Connector._wikilinks"),
    Project(9, "Vault Curate", CONNECT, "notoriouslab", "v", False,
            "pruning and dedupe of the graph", "brain.feedback.Learner.correct"),
    Project(10, "COG Second Brain", CONNECT, "huytieu", "COG-se", False,
            "concept graph over notes", "brain.connect.Connector._similar_links"),
    # --- remember what matters ---
    Project(11, "Hindsight", REMEMBER, "vectorize-io", "h", False,
            "retrieval with hindsight reranking", "brain.recall.Recaller"),
    Project(12, "memU", REMEMBER, "NevaMind-AI", "me", False,
            "agent memory store", "brain.store.Store"),
    Project(13, "TencentDB Agent Memory", REMEMBER, "TencentCloud", "T", False,
            "durable managed memory backend", "brain.store.Store"),
    Project(14, "agentmemory", REMEMBER, "rohitg00", "agent", False,
            "simple agent memory API", "brain.recall.Recaller.recall"),
    Project(15, "OpenViking", REMEMBER, "volcengine", "Ope", False,
            "vector search infrastructure", "brain.embedding"),
    # --- put it to work ---
    Project(16, "Open Second Brain", ACT, "itechmeat", "open", False,
            "end-to-end reference implementation", "brain.pipeline.Brain"),
    Project(17, "Second Brain Cloudflare", ACT, "rahilp", "second-", False,
            "edge-hosted brain", "brain.store.Store"),
    Project(18, "makerskills", ACT, "coreyhaines31", "", False,
            "skills that consume the brain", "tools.brain"),
    Project(19, "Row-Bot", ACT, "siddsachar", "row", False,
            "spreadsheet-driven action loop", "brain.act.Briefer"),
    Project(20, "MateClaw", ACT, "mateaix", "matecl", False,
            "team-facing assistant surface", "brain.pipeline.Brain.session"),
)


@dataclass(frozen=True)
class Build:
    """A recommended stack, and the recall tuning that fits how it is used."""

    name: str
    stack: tuple[str, ...]
    weights_preset: str
    rationale: str
    budget_chars: int = 3500
    recall_limit: int = 6


BUILDS: dict[str, Build] = {
    "creator": Build(
        name="creator",
        stack=("Chubby Skills", "SwarmVault", "Hindsight", "makerskills"),
        weights_preset="exploratory",
        rationale=(
            "A creator needs collision between distant ideas, so recall leans on "
            "semantic similarity and graph expansion rather than exact terms."
        ),
        budget_chars=4200,
        recall_limit=8,
    ),
    "researcher": Build(
        name="researcher",
        stack=("DocsAgent", "sage-wiki", "memU", "Row-Bot"),
        weights_preset="precise",
        rationale=(
            "A researcher is wrong in expensive ways, so recall favours exact "
            "terms and citations, and every claim is meant to be verified."
        ),
        budget_chars=3500,
        recall_limit=6,
    ),
    "team": Build(
        name="team",
        stack=("DocMason", "COG Second Brain", "TencentDB Agent Memory", "MateClaw"),
        weights_preset="team",
        rationale=(
            "A team's brain is judged by what held up in practice, so reinforcement "
            "earned from real sessions counts for more than freshness."
        ),
        budget_chars=3000,
        recall_limit=6,
    ),
}


def by_stage(stage: str) -> tuple[Project, ...]:
    return tuple(p for p in PROJECTS if p.stage == stage)


def find(name: str) -> Project | None:
    lowered = name.lower()
    for p in PROJECTS:
        if p.name.lower() == lowered:
            return p
    return None


def describe_loop() -> str:
    lines = ["The loop:", ""]
    for i, (stage, text) in enumerate(STAGES.items(), 1):
        lines.append(f"{i}. {stage} — {text}")
        for p in by_stage(stage):
            lines.append(f"     {p.n:>2}. {p.name:<26} {p.slot}")
    lines += ["", "Builds:"]
    for build in BUILDS.values():
        lines.append(f"  {build.name:<11} {' -> '.join(build.stack)}")
        lines.append(f"  {'':<11} recall={build.weights_preset}; {build.rationale}")
    return "\n".join(lines)
