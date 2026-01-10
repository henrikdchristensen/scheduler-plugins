"""Test utilities.

This module is imported by multiple test suites. Keep helpers stable and
backwards compatible.
"""

from __future__ import annotations

import io
import logging
from typing import IO, Optional, Tuple


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
