#!/usr/bin/env bash
# Install the Jarvis second brain as a systemd service.
#
#   sudo ./deploy/install.sh                       # bind to the Tailscale address
#   sudo ./deploy/install.sh --host 0.0.0.0        # every interface (read the warning)
#   sudo ./deploy/install.sh --port 9000 --dir /root/apps/brain
#
# Safe to run again: it updates the files in place and restarts the service.
set -euo pipefail

APP_DIR="/root/apps/jarvis-brain"
DATA_DIR=""
PORT="8787"
BIND_HOST=""
SERVICE="jarvis-brain"
NO_START=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)  APP_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) BIND_HOST="$2"; shift 2 ;;
    --name) SERVICE="$2"; shift 2 ;;
    --no-start) NO_START=1; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
DATA_DIR="${DATA_DIR:-$APP_DIR/data}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- checks ---------------------------------------------------------------

[ "$(id -u)" -eq 0 ] || die "run as root (the target directory is under /root)"
command -v python3 >/dev/null || die "python3 is not installed"

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 - <<'PY' || die "python3 $PYV is too old; the brain needs 3.9 or newer"
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
say "python3 $PYV"

# The brain package is pure standard library. If that ever stops being true,
# this deploy stops being a copy and a systemd unit, so it is worth checking.
python3 - "$SRC" <<'PY' || die "brain/ now imports something outside the standard library"
import ast, pathlib, sys
stdlib = set(sys.stdlib_module_names)
root = pathlib.Path(sys.argv[1]) / "brain"
bad = set()
for path in root.rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in stdlib and top != "brain":
                    bad.add(top)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            if top not in stdlib and top != "brain":
                bad.add(top)
raise SystemExit(1 if bad else 0)
PY

if [ -z "$BIND_HOST" ]; then
  if command -v tailscale >/dev/null; then
    BIND_HOST="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  fi
  [ -n "$BIND_HOST" ] || BIND_HOST="127.0.0.1"
fi
say "binding to $BIND_HOST:$PORT"
case "$BIND_HOST" in
  127.0.0.1|localhost) ;;
  100.*) say "reachable from every device on your tailnet, and the app has no login of its own" ;;
  *) printf '\033[33mwarning:\033[0m %s\n' \
       "$BIND_HOST is not a Tailscale address. The app has NO authentication — anyone who can reach this port can read and rewrite the brain." ;;
esac

# --- install --------------------------------------------------------------

say "installing to $APP_DIR"
mkdir -p "$APP_DIR" "$DATA_DIR"
# Replace the code wholesale, never the data directory.
rm -rf "$APP_DIR/brain" "$APP_DIR/tools" "$APP_DIR/docs" "$APP_DIR/examples"
cp -r "$SRC/brain" "$APP_DIR/brain"
cp -r "$SRC/docs" "$APP_DIR/docs" 2>/dev/null || true
cp -r "$SRC/examples" "$APP_DIR/examples" 2>/dev/null || true
find "$APP_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
chmod -R go-rwx "$APP_DIR"

say "writing /etc/systemd/system/$SERVICE.service"
sed -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__DATA_DIR__|$DATA_DIR|g" \
    -e "s|__BIND_HOST__|$BIND_HOST|g" \
    -e "s|__PORT__|$PORT|g" \
    "$SRC/deploy/jarvis-brain.service" > "/etc/systemd/system/$SERVICE.service"

# --- verify before starting ----------------------------------------------

say "checking the installed copy"
( cd "$APP_DIR" && python3 -c "
import brain
from brain.webapp import BrainService
print('   brain', brain.__version__, '| embedder', brain.get_embedder().name)
" ) || die "the installed copy does not import"

if [ "$NO_START" -eq 1 ]; then
  say "installed; not starting (--no-start)"
  exit 0
fi

systemctl daemon-reload
systemctl enable --now "$SERVICE" >/dev/null 2>&1 || systemctl restart "$SERVICE"
systemctl restart "$SERVICE"

say "waiting for it to answer"
for i in $(seq 1 30); do
  if python3 - "$BIND_HOST" "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1); s.close()
PY
  then
    say "up at http://$BIND_HOST:$PORT"
    echo
    echo "  database:  $DATA_DIR/brain.db"
    echo "  logs:      journalctl -u $SERVICE -f"
    echo "  restart:   systemctl restart $SERVICE"
    echo "  cli:       cd $APP_DIR && BRAIN_DB=$DATA_DIR/brain.db python3 -m brain status"
    exit 0
  fi
  sleep 0.5
done

printf '\033[31merror:\033[0m %s\n' "the service did not come up. Last log lines:" >&2
journalctl -u "$SERVICE" -n 30 --no-pager >&2 || true
exit 1
