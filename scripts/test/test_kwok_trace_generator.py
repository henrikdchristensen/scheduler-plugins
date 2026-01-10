#!/usr/bin/env python3
# test_kwok_trace_generator.py

import argparse, json, math, runpy, sys
from pathlib import Path

import pytest

from scripts.kwok_trace_replayer import trace_generator as tg

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_required_args(tmp_path: Path, **overrides):
    base = dict(
        job_file=None,
        output_dir=str(tmp_path),
        seed=42,
        seed_file=None,
        log_level="INFO",
        show_plots=False,
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
    cli_args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--seed", "42",
            "--num-nodes", "8",
            "--trace-time", "3600s",
            "--target-util", "0.9",
            "--xmin-arrival", "0.01",
            "--mean-arrival", "1.0",
            "--xmin-life", "10.0",
            "--xmin-req", "0.01",
            "--xmax-req", "1.0",
            "--mean-req", "0.2",
            "--priority-min", "1",
            "--priority-max", "3",
            "--priority-ratio", "1.0",
            "--replicas-min", "1",
            "--replicas-max", "1",
            "--replicas-ratio", "1.0",
        ]
    )

    args = tg.resolve_effective_args(cli_args)

    assert args.output_dir == str(tmp_path)
    assert args.seed == 42
    assert args.num_nodes == 8
    assert args.trace_time == "3600s"
    assert args.log_level == "INFO"
    assert args.show_plots is False

@pytest.mark.parametrize(
    "missing_flag",
    [
        "--output-dir",
        "--seed",
        "--num-nodes",
        "--trace-time",
        "--target-util",
        "--xmin-arrival",
        "--mean-arrival",
        "--xmin-life",
        "--xmin-req",
        "--xmax-req",
        "--mean-req",
        "--priority-min",
        "--priority-max",
        "--priority-ratio",
        "--replicas-min",
        "--replicas-max",
        "--replicas-ratio",
    ],
)
def test_resolve_effective_args_missing_required_exits(tmp_path: Path, missing_flag: str):
    p = tg.build_arg_parser()
    argv = [
        "--output-dir", str(tmp_path),
        "--seed", "42",
        "--num-nodes", "8",
        "--trace-time", "3600s",
        "--target-util", "0.9",
        "--xmin-arrival", "0.01",
        "--mean-arrival", "1.0",
        "--xmin-life", "10.0",
        "--xmin-req", "0.01",
        "--xmax-req", "1.0",
        "--mean-req", "0.2",
        "--priority-min", "1",
        "--priority-max", "3",
        "--priority-ratio", "1.0",
        "--replicas-min", "1",
        "--replicas-max", "1",
        "--replicas-ratio", "1.0",
    ]
    i = argv.index(missing_flag)
    del argv[i:i + 2]
    cli_args = p.parse_args(argv)
    with pytest.raises(SystemExit):
        tg.resolve_effective_args(cli_args)

def test_job_file_merges_when_cli_missing_and_cli_wins(tmp_path: Path):
    job_path = tmp_path / "job.yaml"
    job_path.write_text(
        """
output-dir: ./ignored-by-cli
seed: 111
num-nodes: 5
trace-time: 7s
target-util: 0.9
xmin-arrival: 0.01
mean-arrival: 1.0
xmin-life: 10.0
xmin-req: 0.01
xmax-req: 1.0
mean-req: 0.2
priority-min: 1
priority-max: 3
priority-ratio: 1.0
replicas-min: 1
replicas-max: 1
replicas-ratio: 1.0
show-plots: true
""".lstrip(),
        encoding="utf-8",
    )

    p = tg.build_arg_parser()
    cli_args = p.parse_args(
        [
            "--job-file", str(job_path),
            "--output-dir", str(tmp_path),
            "--seed", "222",  # should override job seed
        ]
    )

    args = tg.resolve_effective_args(cli_args)
    assert args.seed == 222
    assert args.num_nodes == 5
    assert args.trace_time == "7s"
    assert args.show_plots is True
    assert args.output_dir == str(tmp_path)

def test_seed_file_expands_to_subdirs_under_output_dir(tmp_path: Path):
    seeds_path = tmp_path / "seeds.txt"
    seeds_path.write_text("1\n2\n#comment\n2\n", encoding="utf-8")

    p = tg.build_arg_parser()
    cli_args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--seed-file", str(seeds_path),
            "--num-nodes", "8",
            "--trace-time", "3600s",
            "--target-util", "0.9",
            "--xmin-arrival", "0.01",
            "--mean-arrival", "1.0",
            "--xmin-life", "10.0",
            "--xmin-req", "0.01",
            "--xmax-req", "1.0",
            "--mean-req", "0.2",
            "--priority-min", "1",
            "--priority-max", "3",
            "--priority-ratio", "1.0",
            "--replicas-min", "1",
            "--replicas-max", "1",
            "--replicas-ratio", "1.0",
        ]
    )
    args = tg.resolve_effective_args(cli_args)
    runs = tg.expand_seed_runs(args)

    assert [r.seed for r in runs] == [1, 2]
    assert Path(runs[0].output_dir).name == "1"
    assert Path(runs[1].output_dir).name == "2"

def test_seed_and_seed_file_mutual_exclusion(tmp_path: Path):
    seeds_path = tmp_path / "seeds.txt"
    seeds_path.write_text("1\n", encoding="utf-8")

    p = tg.build_arg_parser()
    cli_args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--seed", "123",
            "--seed-file", str(seeds_path),
            "--target-util", "0.9",
            "--mean-arrival", "1.0",
            "--mean-req", "0.2",
        ]
    )
    with pytest.raises(SystemExit):
        tg.resolve_effective_args(cli_args)

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
    assert seen["include"][:6] == ["job_file", "output_dir", "seed", "seed_file", "log_level", "show_plots"]
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

def test_expected_value_for_support_handles_uniform_and_weighted():
    vals = tg.np.array([1, 2, 3], dtype=int)
    assert tg.TraceGenerator._expected_value_for_support(vals, None) == pytest.approx(2.0)

    probs = tg.np.array([0.5, 0.25, 0.25], dtype=float)
    assert tg.TraceGenerator._expected_value_for_support(vals, probs) == pytest.approx(1.75)

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

    # needs mean_life set
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
# TraceGenerator._build_initial_load()
# =============================================================================

def test_build_initial_load_can_overshoot_then_pop(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, target_util=0.1)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    def const_sample(_rng, _alpha, _x_min, _x_max, size=1):
        return tg.np.full(size, 0.2, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(const_sample))

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, next_id = gen._build_initial_load(
        tg.np.random.default_rng(0),
        prio_vals,
        None,
        rep_vals,
        None,
        next_id=0,
    )

    assert pods == []
    assert next_id == 0

def test_build_initial_load_returns_some_pods(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, target_util=0.4)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    def const_sample(_rng, _alpha, _x_min, _x_max, size=1):
        return tg.np.full(size, 0.2, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(const_sample))

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, next_id = gen._build_initial_load(
        tg.np.random.default_rng(1),
        prio_vals,
        None,
        rep_vals,
        None,
        next_id=0,
    )

    assert len(pods) >= 1
    assert next_id == len(pods)
    assert all(p.start_time == 0.0 for p in pods)

# =============================================================================
# TraceGenerator._generate_trace_events()
# =============================================================================

def test_generate_trace_events_seeds_baseline_and_pops_end_heap(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, trace_time="2.5s")
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    # dt1, life1, req1, dt2, life2, req2, dt3...
    samples = [
        1.0, 0.5, 0.2,
        1.0, 0.5, 0.2,
        10.0,
    ]

    def seq_sample(_rng, _alpha, _x_min, _x_max, size=1):
        v = float(samples.pop(0))
        return tg.np.full(size, v, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(seq_sample))

    initial_pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=0.5, cpu=0.1, mem=0.1, priority=1, replicas=1)
    ]

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, _, times, u_hist, pods_hist = gen._generate_trace_events(
        tg.np.random.default_rng(0),
        prio_vals,
        None,
        rep_vals,
        None,
        next_id=1,
        initial_pods=initial_pods,
    )

    assert len(pods) == 2
    assert times == pytest.approx([0.0, 1.0, 2.0])
    assert len(u_hist) == len(times)
    assert pods_hist == [1, 1, 1]

# =============================================================================
# TraceGenerator._generate_tracedata_once()
# =============================================================================

def test_generate_tracedata_once_sets_series_and_metadata(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=2, trace_time="5s")
    args.mean_life = 2.0
    gen = tg.TraceGenerator(args)

    def fake_fit(_rng):
        _set_alphas(gen, alpha=2.0)

    monkeypatch.setattr(gen, "_fit_alphas", fake_fit)
    monkeypatch.setattr(gen, "_time_avg_req_util", lambda _pods: 0.55)

    initial_pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=2),
    ]
    trace_pods = [
        tg.TraceRecord(id=2, start_time=1.0, end_time=2.0, cpu=0.2, mem=0.2, priority=1, replicas=1),
    ]
    times = [0.0, 1.0]
    u_hist = [0.1, 0.2]
    pods_hist = [2, 1]

    monkeypatch.setattr(gen, "_build_initial_load", lambda *_a, **_k: (initial_pods, 1))
    monkeypatch.setattr(gen, "_generate_trace_events", lambda *_a, **_k: (trace_pods, 2, times, u_hist, pods_hist))

    got_initial, got_trace, stats, extra = gen._generate_tracedata_once(iter_seed=1234)

    assert got_initial == initial_pods
    assert got_trace == trace_pods
    assert stats["util_time_avg"] == pytest.approx(0.55)
    assert gen.times == times
    assert gen.u_req_hist == u_hist
    assert gen.pods_hist == pods_hist
    assert gen.initial_pods_count == 2
    assert extra["seed"] == int(args.seed)
    assert "rng" in extra and "derived" in extra["rng"]
    assert "files" in extra and "initial_json" in extra["files"]

# =============================================================================
# TraceGenerator._calibrate_mean_life()
# =============================================================================

def test_calibrate_mean_life_stops_within_tolerance(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5)
    args.mean_life = 10.0
    gen = tg.TraceGenerator(args)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = []
    extra = {"ok": True}

    monkeypatch.setattr(
        gen,
        "_generate_tracedata_once",
        lambda _seed: (initial, trace, {"util_time_avg": 0.5, "initial_count": 1.0, "trace_count": 0.0}, extra),
    )

    got_initial, got_trace, got_extra = gen._calibrate_mean_life()
    assert got_initial == initial
    assert got_trace == trace
    assert got_extra == extra

def test_calibrate_mean_life_updates_mean_life_and_clamps_to_xmax(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.9, xmin_life=1.0, xmax_life=15.0)
    args.mean_life = 10.0
    gen = tg.TraceGenerator(args)

    calls = {"n": 0}

    def fake_generate_tracedata_once(_seed):
        calls["n"] += 1
        if calls["n"] == 1:
            measured = 0.1
        else:
            measured = float(args.target_util)
        return [], [], {"util_time_avg": measured, "initial_count": 0.0, "trace_count": 0.0}, {"iter": calls["n"]}

    monkeypatch.setattr(gen, "_generate_tracedata_once", fake_generate_tracedata_once)
    monkeypatch.setattr(tg, "CALIB_MEAN_LIFE_MAX_ITER", 2)

    gen._calibrate_mean_life()

    # After first iteration: new_mean_life=10*(0.9/0.1)=90 -> clamp to 0.999*xmax
    assert gen.args.mean_life == pytest.approx(15.0 * 0.999)

def test_calibrate_mean_life_raises_on_zero_measured_util(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5)
    args.mean_life = 10.0
    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(
        gen,
        "_generate_tracedata_once",
        lambda _seed: ([], [], {"util_time_avg": 0.0, "initial_count": 0.0, "trace_count": 0.0}, {}),
    )

    with pytest.raises(RuntimeError):
        gen._calibrate_mean_life()

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
# TraceGenerator._write_outputs()
# =============================================================================

def test_write_outputs_writes_json_and_info(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5)
    gen = tg.TraceGenerator(args)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = [tg.TraceRecord(id=2, start_time=1.0, end_time=2.0, cpu=0.2, mem=0.2, priority=1, replicas=1)]
    extra = {"x": 1}

    calls = {"json": [], "info": 0}

    monkeypatch.setattr(gen, "_time_avg_req_util", lambda _pods: 0.6)

    def fake_write_json(path: Path, pods):
        calls["json"].append((path, len(pods)))

    monkeypatch.setattr(gen, "_write_json", fake_write_json)
    monkeypatch.setattr(gen, "_write_info_file", lambda extra: calls.__setitem__("info", calls["info"] + 1))

    gen._write_outputs(initial, trace, extra)

    assert calls["json"][0][0].name == "initial.json"
    assert calls["json"][1][0].name == "trace.json"
    assert calls["json"][0][1] == 1
    assert calls["json"][1][1] == 1
    assert calls["info"] == 1

# =============================================================================
# TraceGenerator.run()
# =============================================================================

def test_run_infers_mean_life_writes_and_calls_plot_helpers(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    args.mean_life = None
    gen = tg.TraceGenerator(args)

    _set_alphas(gen, alpha=2.0)

    monkeypatch.setattr(gen, "_infer_mean_life_from_target_util", lambda: 12.3)
    monkeypatch.setattr(gen, "log_args", lambda: None)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = []
    extra = {"ok": True}

    monkeypatch.setattr(gen, "_calibrate_mean_life", lambda: (initial, trace, extra))
    monkeypatch.setattr(gen, "_write_outputs", lambda *_a, **_k: None)

    gen.times = [0.0]
    gen.u_req_hist = [0.1]
    gen.pods_hist = [1]
    gen.initial_pods_count = 1

    called = {"util": 0, "hist": 0}

    def fake_util(**kwargs):
        called["util"] += 1
        assert kwargs["out_path"].endswith("utilization.png")

    def fake_hist(**kwargs):
        called["hist"] += 1
        assert kwargs["out_path"].endswith("histograms.png")

    monkeypatch.setattr(tg, "plot_utilization_time_series", fake_util)
    monkeypatch.setattr(tg, "plot_generator_histograms", fake_hist)

    gen.run()

    assert gen.args.mean_life == pytest.approx(12.3)
    assert called["util"] == 1
    assert called["hist"] == 1

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
