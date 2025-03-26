#!/usr/bin/env bash
# Quick launcher for Photocure web dashboard
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -f "$DIR/.venv/bin/activate" ]; then
    . "$DIR/.venv/bin/activate"
fi

exec photocure start "$@"
