#!/bin/bash
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}
DATA_DIR=${PCUR_DATA_DIR:-/data}

if ! getent group "$PGID" >/dev/null 2>&1; then
  groupadd -g "$PGID" photocure
fi

if ! getent passwd "$PUID" >/dev/null 2>&1; then
  useradd -u "$PUID" -g "$PGID" -s /usr/sbin/nologin -d "$DATA_DIR" photocure
fi

# Only walk the volume when its top level is not already owned by the target user.
if [ "$(stat -c '%u:%g' "$DATA_DIR")" != "$PUID:$PGID" ]; then
  chown -R "$PUID:$PGID" "$DATA_DIR"
fi

# gosu keeps the caller's HOME (/root), which the dropped user cannot write to.
export HOME="$DATA_DIR"

exec gosu "$PUID:$PGID" /usr/bin/photocure serve --host 0.0.0.0 --port 8080 "$@"
