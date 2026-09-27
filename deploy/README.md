# Deploying the second brain

The `brain/` package is pure standard library, so a deploy is a copy of a
directory and a systemd unit. No virtualenv, no package install, no build step.

## On the target machine

```bash
git clone https://github.com/hudsonrj/jarvis.git
cd jarvis
git checkout claude/github-second-brain-projects-owcwyo
sudo ./deploy/install.sh
```

That installs to `/root/apps/jarvis-brain`, writes
`/etc/systemd/system/jarvis-brain.service`, starts it, and waits until the app
answers before reporting success. With `tailscale` on the machine it binds to
that machine's Tailscale address automatically; otherwise it binds to localhost.

To pin the address yourself:

```bash
sudo ./deploy/install.sh --host 100.119.119.83 --port 8787
```

Options: `--dir` (default `/root/apps/jarvis-brain`), `--port` (8787),
`--host`, `--name` (the service name), `--no-start`.

Running it again updates the code in place and restarts the service. It never
touches the data directory, so the brain survives every redeploy.

## What ends up where

| Path | Holds |
|---|---|
| `/root/apps/jarvis-brain/brain/` | the code, replaced on each deploy |
| `/root/apps/jarvis-brain/data/brain.db` | the brain — notes, links, claims, history |
| `/etc/systemd/system/jarvis-brain.service` | the unit |

`BRAIN_DB` in the unit is what decides the database path, and the web app, the
command line and the Jarvis voice tools all read it, so they always open the
same file.

## Afterwards

```bash
journalctl -u jarvis-brain -f                    # logs
systemctl restart jarvis-brain                   # restart
cd /root/apps/jarvis-brain && BRAIN_DB=$PWD/data/brain.db python3 -m brain status
```

Feeding it something to start with:

```bash
cd /root/apps/jarvis-brain
export BRAIN_DB=$PWD/data/brain.db
python3 -m brain --session "first load" learn ~/notes      # a folder of markdown
python3 -m brain status
```

## Two things to know before exposing it

**The app has no authentication.** Anyone who can reach the port can read every
note and rewrite any of them. Binding to a Tailscale address means your tailnet
is the boundary, which is a reasonable place to draw it for a personal tool — but
every device on that tailnet is inside it. Do not bind it to `0.0.0.0` on a
machine with a public address; the installer warns when you try.

**Semantic recall wants Ollama.** The unit points `OLLAMA_HOST` at
`127.0.0.1:11434`. If Ollama is not running the brain falls back to its offline
embedder on its own and keeps working, with weaker recall. To get the better one:

```bash
ollama pull nomic-embed-text
systemctl restart jarvis-brain
```

Confirm which one it picked: the app's sidebar shows the embedder, and
`python3 -m brain status` prints it.
