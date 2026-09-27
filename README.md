# 🧠 Jarvis – Local Voice-Controlled AI Assistant

**Jarvis** is a voice-activated, conversational AI assistant powered by a local LLM (Qwen via Ollama). It listens for a wake word, processes spoken commands using a local language model with LangChain, and responds out loud via TTS. It supports tool-calling for dynamic functions like checking the current time.

---

## 🚀 Features

- 🗣 Voice-activated with wake word **"Jarvis"**
- 🧠 Local language model (Qwen 3 via Ollama)
- 🔧 Tool-calling with LangChain
- 🔊 Text-to-speech responses via `pyttsx3`
- 🌍 Example tool: Get the current time in a given city
- 🔐 Optional support for OpenAI API integration
- 🧩 **A second brain that persists between sessions** — see below

---

## 🧩 The Second Brain

Jarvis remembers across sessions, and gets measurably better at it. The `brain/`
package implements one loop:

```
capture the source → turn it into linked, checkable knowledge
  → recall the right context → act with it → write the correction back
```

Everything lives in one SQLite file (`~/.jarvis/brain.db`). The package is **pure
standard library** — no new dependencies — and embeddings fall back to an offline
backend when Ollama is not running.

Six tools are wired into the agent, so by voice you can say *"remember that we
moved the deployment to the Pi"*, ask *"where are we deploying Jarvis?"* later,
and correct it with *"actually, that changed"* — and the correction sticks.

There is a web app for it:

```bash
python -m brain serve          # http://localhost:8787
```

Ask and read a cited answer with its five recall signals drawn per hit, capture a
source, verify the open claims, correct a note and see the before/after, and walk
the graph. Built on `http.server` — no dependency, no CDN, works offline. It binds
to localhost and has no authentication.

There is also a command line:

```bash
python -m brain learn ~/notes/vault          # capture + connect a folder
python -m brain ask "what did we decide about the index"
python -m brain why "index decision"         # the signals behind the ranking
python -m brain claims --open                # what still needs verifying
python -m brain status                       # sharpness, and the cheapest next move
python -m brain trend                        # did it actually get sharper?
python -m brain loop                          # the loop, and the 20 projects behind it
```

Sharpness is a real 0–100 measure of linkage, verification and earned trust — and
capturing raw material on its own *lowers* it, because unlinked, unchecked text
dilutes the brain.

```
$ python -m brain trend
+ monday research       0.0 ->   56.7  (+56.67)
+ tuesday triage       56.7 ->   61.2  (+4.53)
  net over 2 session(s): +61.20
```

Run the demo to watch a full loop end to end:

```bash
python examples/second_brain_demo.py
```

📖 **Full design, the 20-project mapping, the three builds, and the known limits:
[`docs/SECOND_BRAIN.md`](docs/SECOND_BRAIN.md)**

---


## ▶️ How It Works (`main.py`)

1. **Startup & local LLM Setup**
   - Initializes a local Ollama model (`qwen3:1.7b`) via `ChatOllama`
   - Registers tools (`get_time`) using LangChain

2. **Wake Word Listening**
   - Listens via microphone (e.g., `device_index=0`)
   - If it hears the word **"Jarvis"**, it enters "conversation mode"

3. **Voice Command Handling**
   - Records the user’s spoken command
   - Passes the command to the LLM, which may invoke tools
   - Responds using `pyttsx3` text-to-speech (with optional custom voice)

4. **Timeout**
   - If the user is inactive for more than 30 seconds in conversation mode, it resets to wait for the wake word again.

---

## 🤖 How To Start Jarvis

1. **Install Dependencies**  
   Make sure you have installed all required dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up the Local Model**  
   Ensure you have the `qwen3:1.7b` model available in Ollama.

3. **Run Jarvis**  
   Start the assistant by running:
   ```bash
   python main.py
   ```

4. **(Optional) Better semantic recall**  
   The second brain uses Ollama embeddings when available:
   ```bash
   ollama pull nomic-embed-text
   ```
   Without it, an offline fallback embedder is used automatically.

---

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

193 tests. The suite needs no network and no model server.

---

