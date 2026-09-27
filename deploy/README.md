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

## Logging in

The installer generates a token and prints it once. In the browser, leave the
**username blank** and paste the token as the **password**; browsers remember it.

```bash
curl -u ":$TOKEN" http://100.119.119.83:8787/api/state
```

The token lives in `data/env` with mode 600, and the unit reads it through
`EnvironmentFile`, so it is not in the unit and not in `ps` output. Reinstalling
keeps the existing token, so a browser that already remembers it keeps working.

```bash
sudo ./deploy/install.sh --token "$(openssl rand -base64 24)"   # set your own
sudo ./deploy/install.sh --no-token                             # turn it off
sed -n 's/^BRAIN_TOKEN=//p' /root/apps/jarvis-brain/data/env    # read it back
```

To rotate it: delete `data/env`, run the installer again, and it makes a new one.

The token is sent as HTTP Basic auth on every request, so on plain HTTP it is
only as private as the network. Over a tailnet the traffic is already encrypted
between devices, which is what this is for. Anywhere less private, put it behind
a reverse proxy with TLS.

## One thing to know before exposing it

Binding to a Tailscale address means your tailnet is the boundary — a reasonable
place to draw it for a personal tool, though every device on that tailnet is
inside it, which is why the token is on by default. Do not bind to `0.0.0.0` on a
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
