"""Authentication & authorization utilities (OTC, session cookies, host allowlist)."""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("photocure.auth")


class AuthManager:
    """Manages OTC, persistent token, and session state."""

    def __init__(self, cache_dir: Path | str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = self.cache_dir.parent / "auth_token"
        # In-memory OTC tracker (invalidated per-launch).
        self._otc = None
        self._session_tokens = {}

    def generate_otc(self) -> str:
        """Create a fresh OTC (32 bytes, URL-safe)."""
        self._otc = secrets.token_urlsafe(32)
        return self._otc

    def validate_and_consume_otc(self, code: str) -> str | None:
        """Validate OTC and return a persistent session token, or None."""
        if not self._otc or not secrets.compare_digest(code, self._otc):
            return None
        self._otc = None  # OTC is single-use.
        token = self.get_or_create_persistent_token()
        return token

    def get_or_create_persistent_token(self) -> str:
        """Return existing persistent token or create one and save."""
        if self.token_path.exists():
            try:
                token = self.token_path.read_text().strip()
                if token and len(token) >= 32:
                    return token
            except OSError:
                pass
        token = secrets.token_urlsafe(32)
        try:
            self.token_path.write_text(token)
            self.token_path.chmod(0o600)
        except OSError as e:
            log.warning("Failed to persist token: %s", e)
        return token

    def validate_token(self, token: str | None) -> bool:
        """Check if a token is valid (current session or persistent)."""
        if not token:
            return False
        if self._session_tokens.get(token):
            return True
        persistent = self.get_or_create_persistent_token()
        return secrets.compare_digest(token, persistent)

    def create_session_token(self) -> str:
        """Create a short-lived session token (stored in memory)."""
        token = secrets.token_urlsafe(32)
        self._session_tokens[token] = {"created_at": datetime.now(timezone.utc)}
        return token


def build_allowed_hosts(host: str, port: int, allowed_hosts_str: str = "") -> set[str]:
    """Build the set of allowed Host header values."""
    allowed = {
        "localhost",
        "127.0.0.1",
        "[::1]",
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        f"[::1]:{port}",
    }
    # Add configured bind address if non-loopback.
    if host not in ("127.0.0.1", "localhost", "::1", "[::1]"):
        allowed.add(host)
        allowed.add(f"{host}:{port}")
    # Add custom allows from env.
    if allowed_hosts_str:
        for h in allowed_hosts_str.split(","):
            h = h.strip()
            if h:
                allowed.add(h)
                allowed.add(f"{h}:{port}")
    return allowed
