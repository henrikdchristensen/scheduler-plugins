#!/usr/bin/env python3
# job_helpers.py

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from scripts.helpers.general_helpers import get_str, coerce_bool, parse_duration_to_seconds

UNSET = object()

@dataclass(frozen=True)
class JobField:
    job_key: str
    arg_attr: str
    parse: Optional[Callable[[Any], Any]] = None
    # If accept is None, we accept any value not in (None, "").
    accept: Optional[Callable[[Any], bool]] = None

def accept_default(v: Any) -> bool:
    """
    Default accept function: accepts any value not None or empty string.
    """
    return v is not None and v != ""

def parse_optional_str(v: Any) -> str | None:
    """
    Parse an optional string value.
    """
    return get_str(v)

def parse_optional_bool(v: Any) -> bool | None:
    """
    Parse an optional boolean value.
    """
    return coerce_bool(v, default=None)

def parse_optional_int(v: Any) -> int | None:
    """
    Parse an optional integer value.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None

def parse_optional_float(v: Any) -> float | None:
    """
    Parse an optional float value.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_optional_duration_seconds(v: Any) -> float | None:
    """Parse a duration-like value into seconds.
    Accepts numeric values (treated as seconds) and strings like "2h", "30m", "10s", "1d".
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(parse_duration_to_seconds(s))
    except Exception:
        return None

def merge_job_fields_into_args(args: argparse.Namespace, job: dict, fields: Iterable[JobField]) -> argparse.Namespace:
    """
    Merge job-file fields into args.
    """
    job = job or {}
    for f in fields:
        raw = job.get(f.job_key, UNSET)
        if raw is UNSET:
            continue

        val = f.parse(raw) if f.parse else raw
        accept = f.accept or accept_default

        if getattr(args, f.arg_attr, None) is None and accept(val):
            setattr(args, f.arg_attr, val)

    return args