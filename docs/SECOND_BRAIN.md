# The second brain

A second brain that gets sharper after every session — and can prove it.

```
capture the source
  → turn it into linked, checkable knowledge
  → recall the right context
  → act with it
  → write the correction back
```

The loop is the whole design. Each stage is a module, each module is replaceable,
and the last stage is the one most systems skip — which is why most systems are
search indexes over things you once saved, rather than memory.

Everything lives in one SQLite file (`~/.jarvis/brain.db` by default), so a brain
is portable, inspectable with `sqlite3`, and trivially backed up.

**No third-party dependencies.** The `brain/` package is pure standard library.
LangChain is needed only by `tools/brain.py`, which exposes the brain to the
Jarvis agent, and Ollama is optional (see *Embeddings* below).

---

## The five stages

| Stage | Module | What it does |
|---|---|---|
| 1. Capture | `brain/capture.py` | Normalizes text, files, URLs and folders into `Source` records, de-duplicated by SHA-256. Never interprets, so connect can be re-run at any time. |
| 2. Connect | `brain/connect.py` | Splits sources into atomic notes, extracts keywords and tags, resolves `[[wikilinks]]`, infers links from similarity, and pulls out checkable claims. |
| 3. Remember | `brain/recall.py` | Ranks notes with five blended signals and reports the breakdown for each hit. |
| 4. Act | `brain/act.py` | Packs the best notes into a cited, budgeted brief that names its own gaps, and records which notes it cited. |
| 5. Correct | `brain/feedback.py` | Reinforces what helped, weakens what misled, replaces what was wrong while keeping the old version, and records verdicts on claims. |

`brain/pipeline.py` assembles them into the `Brain` facade; `brain/sharpness.py`
measures the result; `brain/registry.py` records which of the 20 projects fills
which slot. Three surfaces sit on top: `tools/brain.py` (the Jarvis agent),
`brain/cli.py` (the command line) and `brain/webapp.py` with `brain/web/`
(the web app). All three drive the same brain file.

---

## Stage 2: what makes knowledge "checkable"

Chunking splits on markdown headings first, then paragraphs, then hard length.
Fragments below `min_note_chars` merge forward so no note is too small to stand
alone; anything over `max_note_chars` is cut at a sentence boundary. When one
section yields several notes they are titled `Heading (2/5)`, because a brief
listing `[1] Stage 3` and `[2] Stage 3` gives the reader no way to tell them apart.

Each note carries a **stable key** — which slice of which source produced it —
that is independent of its body. Connecting the same source again therefore
updates the same notes instead of adding copies, which is what makes `learn` and
`reconnect` safe to re-run. A note whose revision is above 1 has been corrected
by a human, and re-derivation leaves it alone rather than reinstating what the
source originally said.

Links carry an origin, in descending trust:

| Origin | Meaning |
|---|---|
| `wikilink` | An explicit `[[Target]]` someone wrote. A missing target becomes a **stub** note, so the gap is visible instead of silent. |
| `manual` | Asserted after the fact by a human or an agent. |
| `overlap` | Inferred: `0.6 × calibrated similarity + 0.4 × keyword overlap`, above a threshold. |
| `sequence` | Adjacent pieces of one document, in reading order. |

The inferred-link score was tuned against measurement, not taste. Over related
and unrelated note pairs on the hashed backend, the blend scores related pairs at
**0.17–0.91** and unrelated pairs at **0.00–0.12** — the 0.12 being two notes that
merely share the word "first". The threshold sits at **0.15**, inside that gap.

Two details in that formula are load-bearing:

- **Similarity is calibrated**, not raw. A fixed threshold against a raw cosine
  means something different for every backend — the same problem calibration
  exists to solve for recall.
- **Overlap coefficient, not Jaccard.** Jaccard divides by the union, which
  punishes two notes for each having a full keyword list even when they agree on
  the terms that matter. On the same pairs, overlap scores genuinely related notes
  0.03–0.11 higher while leaving unrelated ones at zero, widening the usable gap.

**Keywords** rank by frequency, then by length, then alphabetically. That middle
tie-break matters more than it looks: in a short note almost every content word
appears exactly once, so a purely alphabetical tie-break makes the keyword set
"the first twelve content words in alphabetical order" — quietly dropping the
specific terms late in the alphabet (*reranking*, *retrieval*) in favour of vague
early ones (*few*, *gives*). Keywords feed both the link signal and the search
index, so that bias degraded both.

**Claims** are the assertive, unhedged sentences in a note: they contain a
copula or a modal, they are not questions, and they carry no hedge word
("maybe", "probably", "talvez"). Each starts `unverified` and can be moved to
`supported`, `refuted` or `contested` with evidence. A claim's verdict moves its
note's credibility with it.

---

## Stage 3: how recall ranks

| Signal | What it catches |
|---|---|
| `lexical` | BM25 over stems — exact terms, names, identifiers |
| `semantic` | Cosine over embeddings — paraphrase and synonyms |
| `graph` | One-hop spread from the strongest hits — context by proximity |
| `strength` | Reinforcement earned in past sessions — what proved useful |
| `recency` | A 45-day half-life — fresher notes win ties |

The first three decide **relevance**. The last two are **priors**, and they
*multiply* relevance instead of adding to it:

```
relevance = (w_lex·lexical + w_sem·semantic + w_graph·graph) / (w_lex + w_sem + w_graph)
prior     = (w_str·strength + w_rec·recency) / (w_str + w_rec)
score     = relevance × (1 − k + k·prior)        where k = w_str + w_rec
```

This matters. With an additive prior, `strength` and `recency` are near-constant
across notes and become a floor that swamps a weak relevance signal — an
unrelated invoice outranks the note that actually answers the question. Under
multiplication, a note that does not answer the query scores near zero no matter
how popular or fresh it is, and the priors only break ties.

`Brain.why(query)` prints the full breakdown, so a bad answer is traceable to the
signal that caused it.

**Stemming** is shared by the index and the query (`brain/text.py`), because a
tokenizer that differs between the two fails invisibly. It is deliberately
shallow — `deploying`/`deployment`/`deploys` unify, `running`/`run` do not.
Over-stemming collapses distinct words and is far harder to debug than a miss.

---

## Stage 4: a brief that admits what it lacks

A brief is bounded by a character budget, cites every note with its id and
source, and carries three kinds of honesty:

- when nothing relevant exists, it says so and instructs the caller not to invent;
- below ~0.35 coverage it is stamped **low confidence**;
- it names the query terms no recalled note covers, and the claims still unverified;
- a note with more losses than wins is flagged *previously corrected — verify*.

---

## Stage 5: the write-back

```python
brain.helped(brief)          # every cited note gains strength
brain.misled(brief)          # every cited note loses strength
brain.correct(note_id, text) # replace content, keep the old version
brain.verify(claim_id, "refuted", evidence="measured 52 GB")
brain.decay()                # pull strengths back toward baseline
```

Strength changes multiplicatively and is capped in both directions, so no note
becomes unfalsifiable by being cited often.

A correction does more than swap text. It also:

- keeps the previous body in `corrections`, so how a belief changed is recoverable;
- refreshes the title when the old one described facts that are gone — a stale
  title is what recall shows and what a brief cites, so it is worse than none;
- marks claims whose text no longer exists as `superseded`, and mines the new body
  for the claims that replace them;
- re-embeds and re-links the note.

---

## Measuring sharpness

`brain/sharpness.py` scores the brain 0–100:

| Component | Weight | Measures |
|---|---|---|
| volume | 0.15 | Note count, log-scaled and saturating |
| linkage | 0.25 | Links per note against a target of 3 |
| connectedness | 0.15 | Share of notes that are not orphans |
| verification | 0.20 | Share of live claims with a verdict |
| trust | 0.25 | Laplace-smoothed wins ÷ (wins + losses) |

**Capturing raw material on its own lowers the score.** Unlinked, unchecked text
dilutes the brain, and a metric that rewarded hoarding would be worthless. The
number rises when you finish the loop — and `test_capture_alone_does_not_inflate_the_score`
holds that line.

Sessions record the score at open and close, so the claim is auditable:

```
$ python -m brain trend
+ monday research       0.0 ->   56.7  (+56.67)
+ tuesday triage       56.7 ->   61.2  (+4.53)
  net over 2 session(s): +61.20
```

`brain.next_move()` names the component with the most score left on the table
and what to do about it.

---

## Embeddings

Two interchangeable backends:

- **`OllamaEmbedder`** — uses the Ollama server Jarvis already runs. Set
  `BRAIN_EMBED_MODEL` (default `nomic-embed-text`).
- **`HashEmbedder`** — hashed word, stem and character-trigram features. No
  network, no model files, fully deterministic. Weaker, but it means the brain
  never hard-depends on a running server, and tests are reproducible.

`get_embedder()` tries Ollama and falls back silently. Set
`BRAIN_USE_OLLAMA_EMBED=0` to force the offline path.

Backends disagree on scale — 0.45 cosine is strong for the hashed fallback and
mediocre for a trained model — so each declares its useful band (`sim_floor`,
`sim_ceiling`) and `calibrate()` maps raw cosine into 0..1. Without this the
recall weights would mean something different per backend.

Stopwords are excluded from the hashed features. Left in, they give any two texts
in the same language a similarity floor, and a nonsense query then looks like a
weak match for everything.

---

## The 20 projects, and why this is a registry

The projects in the source list are **interchangeable within a stage**. Each one
fills a slot: something that captures, something that links, something that
remembers, something that acts. `brain/registry.py` records the mapping, and
`brain/` ships a working reference adapter for every slot — so the loop runs
today, and any entry can replace a stage without touching the others.

```
$ python -m brain loop
```

| # | Project | Stage | Slot | Local adapter |
|---|---|---|---|---|
| 1 | Chubby Skills | capture | skill packs as capturable sources | `capture_vault` |
| 2 | OpenWiki | capture | wiki export ingestion | `capture_vault` |
| 3 | Agent Second Brain | capture | agent-side capture of its own output | `capture_text` |
| 4 | DocMason | capture | document set construction | `capture_file` |
| 5 | DocsAgent | capture | docs crawl and normalize | `capture_url` |
| 6 | claude-obsidian | connect | markdown vault as graph substrate | `Connector` |
| 7 | SwarmVault | connect | multi-agent shared vault | `Connector` |
| 8 | sage-wiki | connect | bidirectional linking | `Connector._wikilinks` |
| 9 | Vault Curate | connect | pruning and dedupe | `Learner.correct` |
| 10 | COG Second Brain | connect | concept graph over notes | `Connector._similar_links` |
| 11 | Hindsight | remember | retrieval with reranking | `Recaller` |
| 12 | memU | remember | agent memory store | `Store` |
| 13 | TencentDB Agent Memory | remember | durable managed backend | `Store` |
| 14 | agentmemory | remember | simple agent memory API | `Recaller.recall` |
| 15 | OpenViking | remember | vector search infrastructure | `brain.embedding` |
| 16 | Open Second Brain | act | end-to-end reference | `Brain` |
| 17 | Second Brain Cloudflare | act | edge-hosted brain | `Store` |
| 18 | makerskills | act | skills that consume the brain | `tools.brain` |
| 19 | Row-Bot | act | spreadsheet-driven action loop | `Briefer` |
| 20 | MateClaw | act | team-facing assistant surface | `Brain.session` |

> **On the URLs.** The source list had most repository paths cut off mid-slug
> (`github.com/chubbyguan/chu…`). Each registry entry stores the owner and the
> visible fragment, with `url_complete=False` where the path is incomplete. No
> repository name is guessed, and `test_truncated_urls_are_labelled_not_guessed`
> enforces that. Only `xoai/sage-wiki` arrived complete. The slot mapping — the
> part that actually shapes the architecture — does not depend on the URLs.

### The three builds

Each build covers all four stages and tunes recall to how it is used:

| Build | Stack | Recall | Why |
|---|---|---|---|
| **creator** | Chubby Skills → SwarmVault → Hindsight → makerskills | `exploratory` | Needs collision between distant ideas, so semantic and graph signals lead. |
| **researcher** | DocsAgent → sage-wiki → memU → Row-Bot | `precise` | Wrong in expensive ways, so exact terms and citations lead, and claims get verified. |
| **team** | DocMason → COG Second Brain → TencentDB Agent Memory → MateClaw | `team` | Judged by what held up in practice, so earned reinforcement outweighs freshness. |

```bash
python -m brain --profile researcher ask "what did we decide about the index"
BRAIN_PROFILE=creator python main.py
```

---

## Using it

### From Jarvis, by voice

Six tools are registered on the agent in `main.py`:

| Tool | When the agent reaches for it |
|---|---|
| `brain_remember` | The user states a fact, decision or preference; or says "remember this" |
| `brain_recall` | Before answering anything from an earlier session |
| `brain_correct` | The user says you were wrong, or something changed |
| `brain_feedback` | The user signals the recall was or was not what they meant |
| `brain_status` | "What do you know?", "how good is your memory?" |
| `brain_open_questions` | "What is still uncertain?" |

Replies are written to be spoken: no markdown, no ids read aloud, and an
explicit *I don't know* when nothing is stored.

### Deployed on a machine

`deploy/install.sh` installs it as a systemd service under `/root/apps`. Because
the package is pure standard library, the deploy is a directory copy and a unit
file — no virtualenv, no build. See [`deploy/README.md`](../deploy/README.md).

### From the web app

```bash
python -m brain serve            # http://localhost:8787
```

One page, five views, driving the whole loop:

| View | What it does |
|---|---|
| **Ask** | Asks the brain. Each hit shows its five signal bars, its score, its source, and whether it arrived directly or through a link. Coverage is drawn, and stamped *low confidence* below 0.35. A **Copy LLM prompt** button hands the cited block to any model. |
| **Capture** | Paste, a file path, a folder of markdown, or a URL. Reports the notes, links and claims derived, and the change in sharpness — including when it goes **down**. |
| **Claims** | Every extracted statement by status. Record *supported*, *refuted* or *contested* with evidence, in one click. |
| **Notes** | Search, open a note, read its links and claims, see **how it changed** as a before/after diff, and correct it. |
| **Map** | The graph. Node size follows how many links a note has, brightness follows earned strength, and edge colour follows link origin — an explicit `wikilink` reads through a mesh of inferred ones. Click a node to open the note. |

The sharpness strip sits above every view, so the five components and the
cheapest next move are always in sight.

It is built on `http.server`, so the app adds no dependency, and it works with
no network at all: no CDN, no web fonts, no external anything.

**It binds to 127.0.0.1 and has no authentication.** Anyone who can reach the
port has full read and write access to the brain. `--host` exists for putting it
on a machine you reach over a tailnet or an SSH tunnel; it is not a reason to put
it on the open internet.

### From the command line

```bash
python -m brain learn ~/notes/vault          # capture + connect a folder
python -m brain learn https://example.com/post
python -m brain ask "what did we decide about the index"
python -m brain why "index decision"         # the signals behind the ranking
python -m brain claims --open
python -m brain verify claim_abc supported --evidence "dashboard link"
python -m brain feedback brief_abc helped
python -m brain correct note_abc "the corrected text" --reason "was out of date"
python -m brain --session "monday" learn ~/notes/inbox
python -m brain status
python -m brain trend
python -m brain serve --port 8787
```

### From Python

```python
from brain import Brain

with Brain(profile="researcher") as b:
    with b.session("monday") as ses:
        b.learn_from("~/notes/latency.md")
        brief = b.brief("what do we know about p99 latency")
        print(brief.context)
        b.helped(brief)
        for claim in b.open_claims():
            b.verify(claim.id, "supported", evidence="dashboard")
    print(ses.delta)      # signed change in sharpness
```

### Environment

| Variable | Default | Effect |
|---|---|---|
| `BRAIN_DB` | `~/.jarvis/brain.db` | Where the brain lives. Read by the web app, the command line and the voice tools alike, so all three open the same file. |
| `BRAIN_PROFILE` | `default` | Recall tuning for the Jarvis tools |
| `BRAIN_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `BRAIN_USE_OLLAMA_EMBED` | `1` | `0` forces the offline embedder |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

---

## Known limits

Stated plainly, because a memory system that oversells itself is worse than none.

- **Claim extraction is heuristic.** A regex for copulas, modals and common
  verbs of change finds assertions; it does not understand them. Code blocks,
  tables and markup are filtered out (`_is_prose`), but expect false positives on
  descriptive prose and misses on claims phrased without a recognised verb.
- **Verification is manual.** The brain tracks verdicts and evidence; it does not
  go and check. Automating that is the natural next step, and the honest
  position today is that `supported` means *a person or an agent said so*.
- **The hashed embedder is a fallback, not a model.** It catches shared
  vocabulary and stems; it will not connect "latency" to "slowness" the way a
  trained model does. Run Ollama for real semantic recall.
- **Under the fallback, some obvious links are simply not inferable.** Measured:
  a note on *reranking recovers the recall an approximate index gives up* scores
  0.122 against the note it extends, while two unrelated notes sharing the word
  "first" score 0.118. No threshold separates those. The threshold stays where
  measurement supports it and the brain reports its own under-linking instead —
  `next_move()` says so, and `Learner.link()` is the remedy.
  `examples/second_brain_demo.py` shows exactly this happening.
- **Stemming is shallow and English/Portuguese only.** `running` does not unify
  with `run`, and other languages get tokenization but no stemming.
- **Recall does not cross languages.** Portuguese notes answer Portuguese
  questions and English notes answer English ones, but a Portuguese question
  against an English note scores ~0.15 and comes back stamped *low confidence*
  rather than wrong-but-confident. A multilingual embedding model fixes this;
  the hashed fallback cannot.
- **Similarity linking is quadratic** in note count per connect pass. Fine for
  thousands of notes on one machine; a real vector index is needed beyond that.
- **Re-deriving will not revisit a corrected note.** `connect` skips any note
  whose revision is above 1, because re-reading the original source would undo
  the correction. If you *want* the source's version back, delete the note and
  re-connect.
- **One writer at a time.** SQLite with WAL handles concurrent readers; it is not
  a multi-agent write bus. That is the slot SwarmVault and a managed backend fill.

---

## Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

193 tests, no network, no model server — `BRAIN_USE_OLLAMA_EMBED=0` is set in
`tests/conftest.py`, and the hashed embedder is deterministic.

The ones worth reading first, because they pin the behaviour the design is
actually claiming:

| Test | Holds the line on |
|---|---|
| `test_a_full_loop_raises_sharpness` | The loop improves the brain, measurably |
| `test_capture_alone_does_not_inflate_the_score` | Hoarding is not learning |
| `test_correction_is_recalled_instead_of_the_original` | The write-back actually takes effect |
| `test_feedback_reorders_later_recalls` | Reinforcement changes what surfaces |
| `test_an_unrelated_note_does_not_win_on_priors` | Priors modulate, never carry |
| `test_out_of_domain_query_returns_nothing` | It declines instead of guessing |
| `test_brief_refuses_to_pad_when_nothing_is_known` | Briefs do not invent |
| `test_correction_supersedes_claims_that_no_longer_apply` | Stale claims stop circulating |
| `test_the_brain_survives_a_reopen` | It is durable, not in-memory theatre |
| `test_truncated_urls_are_labelled_not_guessed` | No fabricated repository names |
| `test_service_survives_concurrent_calls` | The web app's writes stay serialized |
| `test_path_traversal_is_refused` | The server serves only its own files |
| `test_link_scoring_separates_related_from_unrelated` | The link threshold sits in a real, measured gap |
| `test_a_captured_markdown_file_does_not_donate_its_code_blocks` | Code and tables never become "claims" |
| `test_connecting_the_same_source_twice_does_not_duplicate_notes` | Re-ingesting does not double the brain |
| `test_re_derivation_does_not_undo_a_correction` | Re-deriving never resurrects superseded facts |
