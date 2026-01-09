#!/usr/bin/env python3
# test_kwok_trace_generator.py

import argparse
import json
import math
import runpy
import sys
from pathlib import Path

import pytest

from scripts.kwok_trace_replayer import trace_generator as tg


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_required_args(tmp_path: Path, **overrides):
    """
    Matches current build_arg_parser() in production:
      required: target_util, mean_arrival, mean_req
      required-ish for tests: output_dir
    """
    base = dict(
        output_dir=str(tmp_path),
        seed=42,
        log_level="INFO",
        num_nodes=2,
        trace_time="5s",
        target_util=0.75,

        xmin_arrival=0.1,
        xmax_arrival=10.0,
        mean_arrival=1.0,

        xmin_life=0.1,
        xmax_life=10.0,
        mean_life=None,

        xmin_req=0.1,
        xmax_req=1.0,
        mean_req=0.2,

        priority_min=1,
        priority_max=2,
        priority_ratio=1.0,

        replicas_min=1,
        replicas_max=2,
        replicas_ratio=1.0,

        util_tol=0.01,
        calib_max_iter=10,
        initial_fill_tol=0.002,
        initial_max_pods=1000,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _set_alphas(gen: tg.TraceGenerator, alpha: float = 2.0) -> None:
    gen.alpha_req = alpha
    gen.alpha_arrival = alpha
    gen.alpha_life = alpha
    gen.args.alpha_req = alpha
    gen.args.alpha_arrival = alpha
    gen.args.alpha_life = alpha


def sample_for_alpha(_rng, alpha, x_min, x_max, size=1):
    # deterministic monotone-ish sample so alpha solver can converge in tests
    v = float(x_min + (1.0 / float(alpha)))
    if x_max is not None:
        v = min(v, float(x_max))
    return tg.np.full(size, v, dtype=float)


# =============================================================================
# build_arg_parser()
# =============================================================================

def test_build_arg_parser_defaults_and_required(tmp_path: Path):
    p = tg.build_arg_parser()
    args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--target-util", "0.9",
            "--mean-arrival", "1.0",
            "--mean-req", "0.2",
        ]
    )

    assert args.output_dir == str(tmp_path)
    assert args.seed == 42
    assert args.num_nodes == 8
    assert args.trace_time == "3600s"

    # defaults
    assert args.priority_min == 1
    assert args.priority_max == 3
    assert args.replicas_min == 1
    assert args.replicas_max == 1


@pytest.mark.parametrize(
    "missing_flag",
    ["--target-util", "--mean-arrival", "--mean-req"],
)
def test_build_arg_parser_missing_required_exits(tmp_path: Path, missing_flag: str):
    p = tg.build_arg_parser()
    argv = [
        "--output-dir", str(tmp_path),
        "--target-util", "0.9",
        "--mean-arrival", "1.0",
        "--mean-req", "0.2",
    ]
    i = argv.index(missing_flag)
    del argv[i:i + 2]
    with pytest.raises(SystemExit):
        p.parse_args(argv)


# =============================================================================
# round_float_args()
# =============================================================================

def test_round_float_args_rounds_only_floats():
    ns = argparse.Namespace(a=1.23456, b=2, c="x", d=3.14159)
    tg.round_float_args(ns, ndigits=2)
    assert ns.a == 1.23
    assert ns.b == 2
    assert ns.c == "x"
    assert ns.d == 3.14


# =============================================================================
# TraceGenerator.__init__()
# =============================================================================

def test_trace_generator_init_creates_paths(tmp_path: Path):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    assert gen.output_dir.exists()
    assert gen.figures_dir.exists()
    assert gen.initial_path.name == "initial.json"
    assert gen.trace_path.name == "trace.json"
    assert gen.info_path.name == "info_generate.yaml"


# =============================================================================
# TraceGenerator.log_args()
# =============================================================================

def test_log_args_calls_log_args_block_and_order(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    seen = {}

    def fake_log_args_block(logger, passed_args, title, include):
        seen["logger"] = logger
        seen["args"] = passed_args
        seen["title"] = title
        seen["include"] = include

    monkeypatch.setattr(tg, "log_args_block", fake_log_args_block)
    gen.log_args()

    assert seen["args"] is args
    assert seen["title"] == "ARGS"
    # first few should match production ordering
    assert seen["include"][:6] == ["output_dir", "seed", "log_level", "num_nodes", "trace_time", "xmin_arrival"]
    assert "target_util" in seen["include"]
    assert "mean_req" in seen["include"]
    assert "mean_arrival" in seen["include"]


# =============================================================================
# TraceGenerator._write_info_file()
# =============================================================================

def test_write_info_file_success_calls_write_info_file(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    called = {}

    monkeypatch.setattr(tg, "build_cli_cmd", lambda: ["python", "trace_generator.py", "--x"])

    def fake_write_info_file(out_path, inputs, logger):
        called["out_path"] = out_path
        called["inputs"] = inputs
        called["logger"] = logger

    monkeypatch.setattr(tg, "write_info_file", fake_write_info_file)

    gen._write_info_file(extra={"k": 1})

    assert str(called["out_path"]).endswith("info_generate.yaml")
    assert called["inputs"]["cli-cmd"] == ["python", "trace_generator.py", "--x"]
    assert "args" in called["inputs"]
    assert called["inputs"]["generated"] == {"k": 1}
    assert called["logger"] is tg.LOG


def test_write_info_file_exception_logs_warning(tmp_path: Path, monkeypatch, caplog):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(tg, "build_cli_cmd", lambda: ["x"])
    monkeypatch.setattr(tg, "write_info_file", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    caplog.set_level("WARNING", logger=tg.LOGGER_NAME)
    gen._write_info_file(extra={})

    assert any("failed to write info_generate.yaml" in rec.message for rec in caplog.records)


# =============================================================================
# TraceGenerator._sample_pareto()
# =============================================================================

def test_sample_pareto_validates_and_respects_xmax():
    rng = tg.np.random.default_rng(123)

    with pytest.raises(ValueError):
        tg.TraceGenerator._sample_pareto(rng, alpha=0.0, x_min=1.0, x_max=None, size=10)

    with pytest.raises(ValueError):
        tg.TraceGenerator._sample_pareto(rng, alpha=1.0, x_min=0.0, x_max=None, size=10)

    with pytest.raises(ValueError):
        tg.TraceGenerator._sample_pareto(rng, alpha=1.0, x_min=2.0, x_max=1.0, size=10)

    x = tg.TraceGenerator._sample_pareto(rng, alpha=2.0, x_min=1.0, x_max=2.0, size=1000)
    assert (x >= 1.0).all()
    assert (x <= 2.0 + 1e-9).all()

    # hit "no x_max" branch
    x2 = tg.TraceGenerator._sample_pareto(rng, alpha=2.0, x_min=1.0, x_max=None, size=10)
    assert (x2 >= 1.0).all()


# =============================================================================
# TraceGenerator._solve_alpha_for_mean()
# =============================================================================

def test_solve_alpha_validates_target_mean_ranges():
    rng = tg.np.random.default_rng(0)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_for_mean(rng, x_min=0.0, x_max=1.0, target_mean=0.5)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_for_mean(rng, x_min=1.0, x_max=None, target_mean=1.0)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_for_mean(rng, x_min=1.0, x_max=2.0, target_mean=3.0)


def test_solve_alpha_bracket_failure_raises(monkeypatch):
    rng = tg.np.random.default_rng(0)

    def constant_sample(_rng, _alpha, _x_min, _x_max, size=1):
        return tg.np.full(size, 1.0, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(constant_sample))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 10)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 10)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_for_mean(rng, x_min=0.5, x_max=2.0, target_mean=1.5)


def test_solve_alpha_converges_with_stubbed_sampling(monkeypatch):
    rng = tg.np.random.default_rng(0)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 200)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_LOWER_BOUND", 0.1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_UPPER_BOUND", 10.0)

    alpha = tg.TraceGenerator._solve_alpha_for_mean(
        rng,
        x_min=0.5,
        x_max=10.0,
        target_mean=0.7,
    )
    assert 0.1 <= alpha <= 10.0


def test_solve_alpha_unbounded_adjusts_alpha_low(monkeypatch):
    rng = tg.np.random.default_rng(0)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 50)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_LOWER_BOUND", 0.1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_UPPER_BOUND", 10.0)

    alpha = tg.TraceGenerator._solve_alpha_for_mean(
        rng,
        x_min=0.5,
        x_max=None,
        target_mean=0.7,
    )
    assert alpha >= 1.0


# =============================================================================
# TraceGenerator._build_geometric_support()
# =============================================================================

def test_build_geometric_support_validates_and_returns_probs():
    values, probs = tg.TraceGenerator._build_geometric_support(min_val=3, max_val=3, ratio=0.5)
    assert values.tolist() == [3]
    assert probs is None

    values, probs = tg.TraceGenerator._build_geometric_support(min_val=1, max_val=3, ratio=1.0)
    assert values.tolist() == [1, 2, 3]
    assert probs is None

    values, probs = tg.TraceGenerator._build_geometric_support(min_val=1, max_val=4, ratio=0.5)
    assert values.tolist() == [1, 2, 3, 4]
    assert probs is not None
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert probs[0] >= probs[-1]

    with pytest.raises(ValueError):
        tg.TraceGenerator._build_geometric_support(min_val=1, max_val=2, ratio=0.0)


# =============================================================================
# TraceGenerator._expected_value()
# =============================================================================

def test_expected_value_handles_uniform_and_weighted():
    vals = tg.np.array([1, 2, 3], dtype=int)
    assert tg.TraceGenerator._expected_value(vals, None) == pytest.approx(2.0)

    probs = tg.np.array([0.5, 0.25, 0.25], dtype=float)
    assert tg.TraceGenerator._expected_value(vals, probs) == pytest.approx(1.75)


# =============================================================================
# TraceGenerator._infer_mean_life_from_target_util()
# =============================================================================

def test_infer_mean_life_validates_ranges(tmp_path: Path):
    args = _make_required_args(tmp_path, target_util=0.5, mean_arrival=2.0, mean_req=0.25, num_nodes=2)
    gen = tg.TraceGenerator(args)

    mean_life = gen._infer_mean_life_from_target_util()
    assert mean_life > 0

    args2 = _make_required_args(tmp_path, target_util=0.0)
    gen2 = tg.TraceGenerator(args2)
    with pytest.raises(ValueError):
        gen2._infer_mean_life_from_target_util()


# =============================================================================
# TraceGenerator._fit_alphas()
# =============================================================================

def test_fit_alphas_sets_fields_and_attaches_to_args(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 100)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)

    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    # needs mean_life set (normally from _ensure_prepared)
    gen.args.mean_life = 2.0

    rng = tg.np.random.default_rng(0)
    gen._fit_alphas(rng)

    assert gen.alpha_req is not None
    assert gen.alpha_arrival is not None
    assert gen.alpha_life is not None
    assert hasattr(args, "alpha_req")
    assert hasattr(args, "alpha_arrival")
    assert hasattr(args, "alpha_life")


# =============================================================================
# TraceGenerator._time_avg_req_util()
# =============================================================================

def test_time_avg_req_util_basic(tmp_path: Path):
    args = _make_required_args(tmp_path, num_nodes=2, trace_time="10s")
    gen = tg.TraceGenerator(args)

    pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=10.0, cpu=0.5, mem=0.5, priority=1, replicas=1),
    ]
    # area = 0.5*10, divide by N*T = 2*10 => 0.25
    assert gen._time_avg_req_util(pods) == pytest.approx(0.25)


# =============================================================================
# TraceGenerator._ensure_prepared()
# =============================================================================

def test_ensure_prepared_sets_mean_life_once(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    calls = {"infer": 0}

    def fake_infer():
        calls["infer"] += 1
        return 3.0

    monkeypatch.setattr(gen, "_infer_mean_life_from_target_util", fake_infer)

    assert not gen._prepared
    gen._ensure_prepared()
    assert gen._prepared
    assert gen.args.mean_life == 3.0
    gen._ensure_prepared()
    assert calls["infer"] == 1


# =============================================================================
# TraceGenerator._write_json()
# =============================================================================

def test_write_json_writes_expected_shape(tmp_path: Path):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    pods = [
        tg.TraceRecord(id=1, start_time=0.1, end_time=0.2, cpu=0.3, mem=0.3, priority=1, replicas=2)
    ]
    out = tmp_path / "x.json"
    gen._write_json(out, pods)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["pods"]
    assert isinstance(data["pods"], list)
    assert data["pods"][0]["id"] == 1

# =============================================================================
# main()
# =============================================================================

def test_main_smoke_invokes_generator(monkeypatch, tmp_path: Path):
    args = _make_required_args(tmp_path)

    class DummyParser:
        def parse_args(self):
            return args

    monkeypatch.setattr(tg, "build_arg_parser", lambda: DummyParser())
    monkeypatch.setattr(tg, "round_float_args", lambda *_a, **_k: None)
    monkeypatch.setattr(tg, "setup_logging", lambda **_k: None)

    seen = {"ran": False}

    class FakeGen:
        def __init__(self, _args):
            self.args = _args

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(tg, "TraceGenerator", FakeGen)

    tg.main()
    assert seen["ran"] is True


def test_main_guard_executes_noop(monkeypatch):
    monkeypatch.setenv("TRACE_GENERATOR_NOOP", "1")

    old_argv = sys.argv[:]
    try:
        sys.argv = [tg.__file__]
        runpy.run_path(tg.__file__, run_name="__main__")
    finally:
        sys.argv = old_argv
