"""Test utilities.

This module is imported by multiple test suites. Keep helpers stable and
backwards compatible.
"""

import io, logging, subprocess
from typing import IO, Optional, Tuple, Callable, List

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
    *,
    # Backward-compatible alias used by some tests.
    output: bytes | None = None,
    assert_prefix: Optional[List[str]] = None,
    raises: bool = False,
) -> Callable[..., bytes]:
    def _fake_check_output(cmd: List[str], *args, **kwargs) -> bytes:
        if assert_prefix is not None:
            prefix_len = len(assert_prefix)
            assert cmd[:prefix_len] == assert_prefix, \
                f"Expected command to start with {assert_prefix}, got {cmd[:prefix_len]}"

        effective = out_bytes if output is None else output
        if raises:
            raise subprocess.CalledProcessError(1, cmd, output=effective)
        return effective
    return _fake_check_output


# ---------------------------------------------------------------------------
# Clock helpers (for testing time-dependent code)
# ---------------------------------------------------------------------------

class TimeController:
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
        # Keep formatting minimal so tests can assert on message content.
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)

    return logger, stream


def get_logger_stream(logger: logging.Logger) -> Optional[io.StringIO]:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if isinstance(stream, io.StringIO):
                return stream
    return None
