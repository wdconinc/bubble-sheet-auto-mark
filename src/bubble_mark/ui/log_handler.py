"""In-memory logging handler that feeds the status-bar drawer."""
from __future__ import annotations

import collections
import logging
import threading
from typing import Callable

_MAX_LINES = 200


class StatusBarHandler(logging.Handler):
    """Logging handler that stores recent formatted log lines.

    Thread-safe: ``emit`` may be called from any thread.  A ``threading.Lock``
    guards the callbacks list so that ``add_callback`` / ``remove_callback``
    calls from other threads cannot corrupt concurrent iterations in ``emit``.
    Registered callbacks are invoked while the lock is *not* held to avoid
    deadlocks; callers are responsible for scheduling any UI update on the
    main thread.
    """

    def __init__(self, max_lines: int = _MAX_LINES) -> None:
        super().__init__()
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # logging.Handler interface
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._lines.append(line)
                callbacks = list(self._callbacks)
            for cb in callbacks:
                try:
                    cb(line)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def add_callback(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with each new formatted line."""
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb: Callable[[str], None]) -> None:
        """Unregister *cb* (no-op if not registered)."""
        with self._lock:
            try:
                self._callbacks.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lines(self) -> list[str]:
        """All retained log lines, oldest first."""
        return list(self._lines)

    @property
    def last_line(self) -> str:
        """The most recent log line, or an empty string if none."""
        return self._lines[-1] if self._lines else ""
