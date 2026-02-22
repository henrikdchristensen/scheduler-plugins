#!/usr/bin/env python3
# trace_helpers.py

import re
from dataclasses import dataclass
import numpy as np
from scipy.stats import pareto as pareto_dist

# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------

@dataclass
class TraceRecord:
    id: int
    start_time: float
    end_time: float
    cpu: float
    mem: float
    priority: int
    replicas: int = 1

# -------------------------------------------------------------------
# Pod naming helpers
# -------------------------------------------------------------------

RS_PREFIX_RE = re.compile(r"^(rs-\d{6})(?:-.*)?$")

def rs_prefix_from_pod_name(pod_name: str) -> str:
    """
    Extract the ReplicaSet prefix (e.g. "rs-000001") from a pod name.
    """
    m = RS_PREFIX_RE.match(pod_name or "")
    if m:
        return m.group(1)
    if "-" in (pod_name or ""):
        return (pod_name or "").split("-", 1)[0]
    return pod_name or ""

# ----------------------------------------------------------------------
# Pareto I estimation helper
# ----------------------------------------------------------------------

def estimate_pareto_params(pos_data: np.ndarray) -> tuple[float, float] | None:
    """
    Estimate Pareto parameters via MLE using SciPy's pareto.
    """
    b_hat, _, scale_hat = pareto_dist.fit(pos_data, floc=0.0)
    if scale_hat <= 0 or b_hat <= 0:
        return None
    return float(b_hat), float(scale_hat)  # alpha, x_min