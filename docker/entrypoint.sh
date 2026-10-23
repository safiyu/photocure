#!/bin/sh
# Container start: install the AI libraries for the detected hardware once, then serve.
set -e
DATA_DIR="${PCUR_DATA_DIR:-/data}"
# Libraries live in the data volume so they are installed once and survive restarts.
export PYTHONUSERBASE="${PYTHONUSERBASE:-$DATA_DIR/pylibs}"
if ! mkdir -p "$(python -c 'import site; print(site.getusersitepackages())')" 2>/dev/null || [ ! -w "$DATA_DIR" ]; then
  echo "photocure: $DATA_DIR is not writable by uid $(id -u). Chown it to that user (e.g. chown -R $(id -u) <host path>) or set 'user:' to match its owner." >&2
  exit 1
fi

# Volumes created before the app was renamed keep their database and settings.
if [ -f "$DATA_DIR/photo_curator.db" ] && [ ! -f "$DATA_DIR/photocure.db" ]; then
  for ext in "" -wal -shm; do
    [ -f "$DATA_DIR/photo_curator.db$ext" ] && mv "$DATA_DIR/photo_curator.db$ext" "$DATA_DIR/photocure.db$ext"
  done
  [ -f "$DATA_DIR/photo_curator.settings.json" ] && mv "$DATA_DIR/photo_curator.settings.json" "$DATA_DIR/photocure.settings.json"
fi

if [ "$1" = "auto" ]; then
  if [ "${PCUR_AUTO_SETUP:-1}" = "0" ]; then
    # Baked or offline image: no installing, just serve.
    exec photocure serve --db "$DATA_DIR/photocure.db" --host 0.0.0.0 --port 8080
  fi
  exec photocure setup --yes --no-open --host 0.0.0.0 --port 8080 --db "$DATA_DIR/photocure.db"
fi
exec "$@"
