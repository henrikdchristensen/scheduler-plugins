"""Test utilities.

This module is imported by multiple test suites. Keep helpers stable and
backwards compatible.
"""

from __future__ import annotations

import io
import logging
import subprocess
import threading
from typing import IO, Optional, Tuple, Callable, Any, List


# ---------------------------------------------------------------------------
# Lock helpers (for testing thread-safe code)
# ---------------------------------------------------------------------------

# A no-op lock for testing that doesn't actually lock
class _NullLock:
    """A no-op context manager lock for testing"""
    def __call__(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def acquire(self):
        pass
    def release(self):
        pass

null_lock = _NullLock()


# ---------------------------------------------------------------------------
# Time mocking helpers
# ---------------------------------------------------------------------------

def time_sequence(values: List[float]) -> Callable[[], float]:
    """
    Return a callable that returns values from a sequence on each call.
    Used to mock time.time() in tests.
    
    Example:
        fake_time = time_sequence([0.0, 0.1, 0.2, 999.0])
        monkeypatch.setattr(module.time, "time", fake_time)
    """
    iterator = iter(values)
    def _next_time() -> float:
        return next(iterator)
    return _next_time


# ---------------------------------------------------------------------------
# Subprocess mocking helpers
# ---------------------------------------------------------------------------

def make_subprocess_run(
    returncode: int = 0,
    out_bytes: bytes = b"",
    assert_prefix: Optional[List[str]] = None,
) -> Callable[..., subprocess.CompletedProcess]:
    """
    Create a fake subprocess.run that returns a fixed result.
    Optionally asserts that the command starts with assert_prefix.
    
    Example:
        fake_run = make_subprocess_run(
            returncode=0,
            out_bytes=b"output",
            assert_prefix=["kubectl", "--context", "ctx1"],
        )
        monkeypatch.setattr(subprocess, "run", fake_run)
    """
    def _fake_run(cmd: List[str], *args, **kwargs) -> subprocess.CompletedProcess:
        if assert_prefix is not None:
            prefix_len = len(assert_prefix)
            assert cmd[:prefix_len] == assert_prefix, \
                f"Expected command to start with {assert_prefix}, got {cmd[:prefix_len]}"
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=returncode,
            stdout=out_bytes,
            stderr=b"",
        )
    return _fake_run


def make_subprocess_run_seq(
    returncodes: List[int],
    out_bytes: bytes = b"",
) -> Callable[..., subprocess.CompletedProcess]:
    """
    Create a fake subprocess.run that returns different returncodes on successive calls.
    Used for testing retry logic.
    
    Example:
        fake_run = make_subprocess_run_seq([1, 1, 0], out_bytes=b"")
        monkeypatch.setattr(subprocess, "run", fake_run)
    """
    iterator = iter(returncodes)
    def _fake_run(cmd: List[str], *args, **kwargs) -> subprocess.CompletedProcess:
        rc = next(iterator)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=rc,
            stdout=out_bytes,
            stderr=b"",
        )
    return _fake_run


def make_subprocess_check_output(
    out_bytes: bytes = b"",
    raises: bool = False,
) -> Callable[..., bytes]:
    """
    Create a fake subprocess.check_output that returns fixed output or raises.
    
    Example:
        fake_check_output = make_subprocess_check_output(
            out_bytes=b'{"items": []}',
        )
        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    """
    def _fake_check_output(cmd: List[str], *args, **kwargs) -> bytes:
        if raises:
            raise subprocess.CalledProcessError(1, cmd, output=out_bytes)
        return out_bytes
    return _fake_check_output


# ---------------------------------------------------------------------------
# Clock helpers (for testing time-dependent code)
# ---------------------------------------------------------------------------

class TimeController:
    """
    A simple fake clock for testing that doesn't actually sleep.
    Implements the Clock protocol from general_helpers.
    """
    def __init__(self, now: float = 0.0):
        self.current_time = now
    
    @property
    def now(self) -> float:
        """Alias for current_time for backward compatibility"""
        return self.current_time
    
    @now.setter
    def now(self, value: float) -> None:
        """Setter for backward compatibility"""
        self.current_time = value
    
    def time(self) -> float:
        return self.current_time
    
    def sleep(self, seconds: float) -> None:
        # Don't actually sleep in tests
        self.current_time += seconds
    
    def advance(self, seconds: float) -> None:
        """Manually advance the clock (for test control)"""
        self.current_time += seconds


# ---------------------------------------------------------------------------
# Logger helpers
# ---------------------------------------------------------------------------

def make_logger_stream(
    name: str = "test",
    level: int = logging.DEBUG,
) -> Tuple[logging.Logger, IO[str]]:
    """Create (or reuse) a logger configured with a single StreamHandler.

    Returns a tuple of (logger, string_stream) where string_stream is a
    text-mode stream (io.StringIO) attached to a StreamHandler.

    Backwards compatibility:
    - Historically, this helper existed multiple times in this file.
    - Some tests may call it repeatedly with the same logger name.
      We therefore clear existing handlers for that logger before adding a new
      one, and we disable propagation to avoid duplicate output.
    """

    stream = io.StringIO()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicate log lines when called
    # multiple times across tests.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)

    return logger, stream


def get_logger_stream(logger: logging.Logger) -> Optional[io.StringIO]:
    """Return the first attached StringIO stream for a logger, if present.

    This is a small compatibility helper used in some tests.
    """

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if isinstance(stream, io.StringIO):
                return stream
    return None
