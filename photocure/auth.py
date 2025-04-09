"""Authentication helpers: one-time code, persistent token, Host/Origin checks."""
from __future__ import annotations

import logging
import os
import secrets
import socket
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("photocure.auth")

COOKIE_NAME = "pcur_session"


def token_file_path(cache_dir: Path | str) -> Path:
    """Next to an absolute cache dir (Docker: /data/auth_token), else the XDG data dir.

    A relative cache dir is relative to wherever the user ran photocure, and the
    token must not be dropped into an arbitrary working directory.
    """
    cache = Path(cache_dir)
    if cache.is_absolute():
        return cache.parent / "auth_token"
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "photocure" / "auth_token"


class AuthManager:
    """Holds the one-time code (OTC) and the persistent session token."""

    def __init__(self, cache_dir: Path | str = ".cache", token: str | None = None):
        self.token_path = token_file_path(cache_dir)
        self._env_token = token or os.environ.get("PCUR_AUTH_TOKEN") or None
        # The CLI generates the OTC and hands it over via the environment, never via HTML.
        self._otc: str | None = os.environ.pop("PCUR_OTC", None) or None
        self.token = self._env_token or self._load_or_create()

    def _load_or_create(self) -> str:
        try:
            tok = self.token_path.read_text().strip()
            if len(tok) >= 32:
                return tok
        except OSError:
            pass
        tok = secrets.token_urlsafe(32)
        try:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(tok)
        except OSError as e:
            log.warning("could not persist auth token: %s", e)
        return tok

    def set_otc(self, code: str) -> None:
        self._otc = code

    def exchange(self, code: str | None) -> str | None:
        """Return the session token for a valid single-use OTC or the persistent token."""
        if not code:
            return None
        if self._otc and _same(code, self._otc):
            self._otc = None
            return self.token
        if _same(code, self.token):
            return self.token
        return None

    def validate(self, token: str | None) -> bool:
        return bool(token) and _same(token, self.token)


def _same(a: str, b: str) -> bool:
    # compare_digest raises TypeError on non-ASCII str, so compare bytes
    return secrets.compare_digest(a.encode("utf-8", "surrogatepass"),
                                  b.encode("utf-8", "surrogatepass"))


def new_otc() -> str:
    return secrets.token_urlsafe(32)


def _local_names() -> set[str]:
    """This machine's own hostname and addresses, used when binding to all interfaces."""
    names: set[str] = set()
    try:
        hostname = socket.gethostname()
        names.add(hostname)
        names.update(a for a in socket.gethostbyname_ex(hostname)[2] if not a.startswith("127."))
    except OSError:
        pass
    return {n for n in names if n}


def build_allowed_hosts(host: str, port: int, extra: str = "") -> set[str]:
    allowed = {"localhost", "127.0.0.1", "[::1]", "::1",
               f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
    if host and host not in ("0.0.0.0", "::"):
        allowed |= {host, f"{host}:{port}"}
    else:
        for h in _local_names():
            allowed |= {h, f"{h}:{port}"}
    for h in (x.strip() for x in extra.split(",")):
        if h:
            allowed |= {h, f"{h}:{port}"}
    return allowed


def origin_allowed(origin: str, allowed_hosts: set[str]) -> bool:
    """True when the Origin/Referer authority is in the Host allowlist."""
    if "*" in allowed_hosts:
        return True
    try:
        netloc = urlparse(origin).netloc
    except ValueError:
        return False
    if not netloc:
        return False
    netloc_no_port = netloc.split(":")[0]
    return netloc in allowed_hosts or netloc_no_port in allowed_hosts
