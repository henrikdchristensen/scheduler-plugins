#!/usr/bin/env python3
#test_utils.py

import contextlib, io, logging, subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Tuple, Optional, Sequence

def make_logger_stream(name_prefix: str, level: int = logging.DEBUG) -> Tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger(f"test-{name_prefix}-{id(stream)}")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, stream

@contextlib.contextmanager
def null_lock() -> Iterator[None]:
    """A do-nothing context manager used to stub out file locks."""
    yield

@dataclass
class TimeController:
    now: float = 0.0
    sleeps: List[float] = field(default_factory=list)
    def time(self) -> float:
        return float(self.now)
    def sleep(self, dt: float) -> None:
        dt_f = float(dt)
        self.sleeps.append(dt_f)
        self.now += dt_f

def make_logger_stream(name: str, level: int = logging.INFO) -> Tuple[logging.Logger, io.StringIO]:
    """
    Create a logger writing to an in-memory StringIO.
    Returns (logger, stream).
    """
    stream = io.StringIO()
    logger = logging.getLogger(f"test.{name}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    h = logging.StreamHandler(stream)
    fmt = logging.Formatter("%(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger, stream

def time_sequence(values: Iterable[float]) -> Callable[[], float]:
    seq = list(values)
    if not seq:
        seq = [0.0]
    idx = {"i": 0}
    def _time() -> float:
        i = idx["i"]
        if i < len(seq) - 1:
            idx["i"] += 1
            return seq[i]
        return seq[-1]
    return _time

def make_subprocess_run(
    *,
    returncode: int = 0,
    out_bytes: bytes = b"",
    err_bytes: bytes = b"",
    assert_prefix: Optional[Sequence[str]] = None,
):
    def _run(cmd, input=None, stdout=None, stderr=None, check=False, **kwargs):
        if assert_prefix is not None:
            pref = list(assert_prefix)
            assert list(cmd)[: len(pref)] == pref
        cp = subprocess.CompletedProcess(cmd, returncode, stdout=out_bytes, stderr=err_bytes)
        if check and returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=returncode,
                cmd=cmd,
                output=out_bytes,
                stderr=err_bytes,
            )
        return cp
    return _run

def make_subprocess_run_seq(
    returncodes: Iterable[int],
    *,
    out_bytes: bytes = b"",
    err_bytes: bytes = b"",
    assert_prefix: Optional[Sequence[str]] = None,
):
    seq = list(returncodes)
    def _run(cmd, input=None, stdout=None, stderr=None, check=False, **kwargs):
        rc = seq.pop(0) if seq else 0
        if assert_prefix is not None:
            pref = list(assert_prefix)
            assert list(cmd)[: len(pref)] == pref
        cp = subprocess.CompletedProcess(cmd, rc, stdout=out_bytes, stderr=err_bytes)
        if check and rc != 0:
            raise subprocess.CalledProcessError(
                returncode=rc,
                cmd=cmd,
                output=out_bytes,
                stderr=err_bytes,
            )
        return cp
    return _run

def make_subprocess_check_output(
    *,
    output: bytes = b"",
    assert_prefix: Optional[Sequence[str]] = None,
):
    def _check_output(cmd, stderr=None, **kwargs):
        if assert_prefix is not None:
            pref = list(assert_prefix)
            assert list(cmd)[: len(pref)] == pref
        return output
    return _check_output