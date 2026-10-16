#!/bin/bash
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

if ! id -g $PGID >/dev/null 2>&1; then
  groupadd -g $PGID photocure
fi

if ! id -u $PUID >/dev/null 2>&1; then
  useradd -u $PUID -g $PGID -s /usr/sbin/nologin -d /data photocure
fi

chown -R $PUID:$PGID /data

exec gosu $PUID:$PGID /usr/bin/photocure serve --host 0.0.0.0 --port 8080 "$@"
