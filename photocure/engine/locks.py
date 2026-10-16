"""Process-wide lock held while original photos or ledger rows are being mutated."""
import threading

LEDGER_WRITING = threading.Lock()
