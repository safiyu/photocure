"""Tests for post-apply webhook and shell triggers."""
import json
import os
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import pytest

from photocure.config import CuratorConfig
from photocure.engine.hooks import dispatch_post_apply_hook
from photocure.services import apply_settings, save_settings, load_settings


def test_post_apply_hook_shell(tmp_path):
    out_file = tmp_path / "hook_out.txt"
    # Shell command that writes env vars to file
    cmd = f'echo "$PCUR_TRANSACTION_ID $PCUR_EXECUTED_COUNT" > "{out_file}"'
    payload = {
        "transaction_id": "TX_20261103_TEST",
        "organize_root": str(tmp_path),
        "executed": 42,
        "failures": 0,
    }
    dispatch_post_apply_hook(cmd, payload, block=True)
    assert out_file.exists()
    content = out_file.read_text().strip()
    assert content == "TX_20261103_TEST 42"


def test_post_apply_hook_http():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            received.append(json.loads(body.decode("utf-8")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *args):
            pass  # quiet test output

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    payload = {"transaction_id": "TX_HTTP_123", "executed": 5}
    dispatch_post_apply_hook(f"http://127.0.0.1:{port}/webhook", payload, block=True)
    server.server_close()

    assert len(received) == 1
    assert received[0]["transaction_id"] == "TX_HTTP_123"
    assert received[0]["executed"] == 5


def test_settings_save_and_apply_post_apply_hook(tmp_path):
    db_path = tmp_path / "test.db"
    cfg = CuratorConfig()
    assert cfg.post_apply_hook is None

    # Test saving via settings
    saved = save_settings(db_path, {"post_apply_hook": "https://example.com/webhook"})
    assert saved["post_apply_hook"] == "https://example.com/webhook"

    stored = load_settings(db_path)
    assert stored["post_apply_hook"] == "https://example.com/webhook"

    applied = apply_settings(cfg, stored)
    assert applied.post_apply_hook == "https://example.com/webhook"

