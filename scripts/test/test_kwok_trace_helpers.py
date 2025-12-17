#!/usr/bin/env python3
# test_kwok_trace_helpers.py

import pytest
import numpy as np

from scripts.kwok_trace_replayer import trace_helpers as th


# ---------------------------------------------------------------------------
# estimate_pareto_params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pos_data",
    [
        np.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0], dtype=float),
        np.array([0.5, 0.75, 1.0, 1.1, 1.25, 1.6, 2.2], dtype=float),
        np.array([10.0, 11.0, 12.0, 15.0, 20.0, 40.0], dtype=float),
    ],
)
def test_estimate_pareto_params_returns_positive_estimates(pos_data: np.ndarray):
    est = th.estimate_pareto_params(pos_data)
    assert est is not None
    alpha, x_min = est
    assert isinstance(alpha, float)
    assert isinstance(x_min, float)
    assert alpha > 0
    assert x_min > 0


@pytest.mark.parametrize(
    "b_hat,scale_hat",
    [
        (0.0, 1.0),     # invalid alpha
        (-1.0, 1.0),    # invalid alpha
        (1.0, 0.0),     # invalid x_min
        (1.0, -2.0),    # invalid x_min
    ],
)
def test_estimate_pareto_params_returns_none_on_invalid(monkeypatch, b_hat: float, scale_hat: float):
    calls = {"floc": None}

    def fake_fit(data, *args, **kwargs):
        calls["floc"] = kwargs.get("floc", None)
        # SciPy returns: (shape=b_hat, loc, scale)
        return (b_hat, 0.0, scale_hat)

    monkeypatch.setattr(th.pareto_dist, "fit", fake_fit)

    pos_data = np.array([1.0, 2.0, 3.0], dtype=float)
    assert th.estimate_pareto_params(pos_data) is None
    assert calls["floc"] == 0.0
