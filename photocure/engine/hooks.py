"""Post-apply event hooks (webhooks and shell triggers)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import urllib.request
from typing import Any

log = logging.getLogger("photocure.hooks")


def dispatch_post_apply_hook(
    hook: str | None,
    payload: dict[str, Any],
    block: bool = False,
    allow_shell: bool = True,
) -> None:
    """Dispatches a post-apply hook (HTTP POST or shell command).

    If ``block`` is False (default), the dispatch runs asynchronously in a daemon thread.
    Shell execution requires ``allow_shell=True``.
    """
    if not hook or not str(hook).strip():
        return

    hook_str = str(hook).strip()

    def _run() -> None:
        try:
            if hook_str.startswith(("http://", "https://")):
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    hook_str,
                    data=data,
                    headers={"Content-Type": "application/json", "User-Agent": "Photocure/2.0"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    status = getattr(resp, "status", getattr(resp, "code", 200))
                    log.info("post_apply_hook HTTP %s -> status %s", hook_str, status)
            elif not allow_shell:
                log.warning("post_apply_hook shell execution disabled for untrusted input: %s", hook_str)
            else:
                env = os.environ.copy()
                env["PCUR_TRANSACTION_ID"] = str(payload.get("transaction_id", ""))
                env["PCUR_ORGANIZE_ROOT"] = str(payload.get("organize_root", ""))
                env["PCUR_EXECUTED_COUNT"] = str(payload.get("executed", 0))
                env["PCUR_FAILURES_COUNT"] = str(payload.get("failures", 0))
                env["PCUR_COLLISIONS_COUNT"] = str(payload.get("collisions", 0))
                env["PCUR_PAYLOAD_JSON"] = json.dumps(payload)
                res = subprocess.run(
                    hook_str,
                    shell=True,
                    env=env,
                    timeout=60,
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    log.warning("post_apply_hook shell exited %d: %s", res.returncode, res.stderr.strip())
                else:
                    log.info("post_apply_hook shell completed: %s", hook_str)
        except Exception as e:
            log.warning("post_apply_hook failed (%s): %s", hook_str, e)

    if block:
        _run()
    else:
        t = threading.Thread(target=_run, name="post_apply_hook", daemon=True)
        t.start()

