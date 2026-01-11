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
    Build a Namespace that passes TraceGenerator._validate_args().
    """
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

        # Inter-arrival
        xmin_arrival=0.1,
        xmax_arrival=10.0,
        mean_arrival=1.0,

        # Lifetime (bounded Pareto) -- keep mean_life feasible (< bounded_pareto_max_mean)
        xmin_life=2.0,
        xmax_life=100.0,
        mean_life=10.0,

        # Requests
        xmin_req=0.1,
        xmax_req=1.0,
        mean_req=0.2,

        # Priority + replicas
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
    gen.alpha_req = float(alpha)
    gen.alpha_arrival = float(alpha)
    gen.alpha_life = float(alpha)
    gen.args.alpha_req = float(alpha)
    gen.args.alpha_arrival = float(alpha)
    gen.args.alpha_life = float(alpha)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
            "--xmax-arrival", "10.0",
            "--mean-arrival", "1.0",

            "--xmin-life", "10.0",
            "--xmax-life", "100.0",

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

    args = tg.TraceGenerator.resolve_effective_args(cli_args)

    assert args.output_dir == str(tmp_path)
    assert args.seed == 42
    assert args.num_nodes == 8
    assert args.trace_time == "3600s"
    assert args.log_level == tg.DEFAULT_LOG_LEVEL
    assert args.show_plots is tg.DEFAULT_SHOW_PLOTS


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--output-dir",
        "--seed",
        "--num-nodes",
        "--trace-time",
        "--target-util",
        "--xmin-arrival",
        "--xmax-arrival",
        "--mean-arrival",
        "--xmin-life",
        "--xmax-life",
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
        "--xmax-arrival", "10.0",
        "--mean-arrival", "1.0",

        "--xmin-life", "10.0",
        "--xmax-life", "100.0",

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
        tg.TraceGenerator.resolve_effective_args(cli_args)


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
        tg.TraceGenerator.resolve_effective_args(cli_args)


# ---------------------------------------------------------------------------
# Job-file helpers (private)
# ---------------------------------------------------------------------------

def test_load_job_doc_missing_file_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        tg.TraceGenerator.load_job_doc(tmp_path / "nope.yaml")


def test_load_job_doc_non_mapping_exits(tmp_path: Path):
    p = tmp_path / "job.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")  # YAML list
    with pytest.raises(SystemExit):
        tg.TraceGenerator.load_job_doc(p)


def test_apply_defaults_sets_log_level_and_show_plots_when_none(tmp_path: Path):
    ns = argparse.Namespace(log_level=None, show_plots=None)
    out = tg.TraceGenerator.apply_defaults(ns)
    assert out.log_level == tg.DEFAULT_LOG_LEVEL
    assert out.show_plots is tg.DEFAULT_SHOW_PLOTS


def test_validate_args_enforces_min_lifetime_floor(tmp_path: Path):
    args = _make_required_args(tmp_path, xmin_life=1.0, xmax_life=10.0)
    with pytest.raises(SystemExit):
        tg.TraceGenerator.validate_args(args)


def test_validate_args_requires_xmax_gt_xmin(tmp_path: Path):
    args = _make_required_args(tmp_path, xmin_arrival=1.0, xmax_arrival=1.0)
    with pytest.raises(SystemExit):
        tg.TraceGenerator.validate_args(args)


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
xmax-arrival: 10.0
mean-arrival: 1.0

xmin-life: 10.0
xmax-life: 100.0

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

    args = tg.TraceGenerator.resolve_effective_args(cli_args)
    assert args.seed == 222
    assert args.num_nodes == 5
    assert args.trace_time == "7s"
    assert args.show_plots is True
    assert args.output_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# expand_seed_runs()
# ---------------------------------------------------------------------------

def test_expand_seed_runs_single_seed(tmp_path: Path):
    args = _make_required_args(tmp_path, seed=7, seed_file=None)
    runs = tg.TraceGenerator.expand_seed_runs(args)
    assert [r.seed for r in runs] == [7]
    assert str(Path(runs[0].output_dir).resolve()) == str(Path(tmp_path).resolve())


def test_expand_seed_runs_seed_file_expands_to_subdirs_under_output_dir(tmp_path: Path, monkeypatch):
    # Do not test read_seeds_file() (imported) — stub it.
    seeds_path = tmp_path / "seeds.txt"
    seeds_path.write_text("1\n2\n", encoding="utf-8")

    p = tg.build_arg_parser()
    cli_args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--seed-file", str(seeds_path),

            "--num-nodes", "8",
            "--trace-time", "3600s",
            "--target-util", "0.9",

            "--xmin-arrival", "0.01",
            "--xmax-arrival", "10.0",
            "--mean-arrival", "1.0",

            "--xmin-life", "10.0",
            "--xmax-life", "100.0",

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
    args = tg.TraceGenerator.resolve_effective_args(cli_args)

    monkeypatch.setattr(tg, "read_seeds_file", lambda _p, logger: [1, 2])

    runs = tg.TraceGenerator.expand_seed_runs(args)
    assert [r.seed for r in runs] == [1, 2]
    assert Path(runs[0].output_dir).name == "1"
    assert Path(runs[1].output_dir).name == "2"


# ---------------------------------------------------------------------------
# round_float_args()
# ---------------------------------------------------------------------------

def test_round_float_args_rounds_only_floats():
    ns = argparse.Namespace(a=1.23456, b=2, c="x", d=3.14159)
    tg.TraceGenerator.round_float_args(ns, ndigits=2)
    assert ns.a == 1.23
    assert ns.b == 2
    assert ns.c == "x"
    assert ns.d == 3.14


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_trace_generator_init_creates_paths(tmp_path: Path):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    assert gen.output_dir.exists()
    assert gen.figures_dir.exists()
    assert gen.initial_path.name == "initial.json"
    assert gen.trace_path.name == "trace.json"
    assert gen.info_path.name == "info_generate.yaml"


def test_from_run_args_creates_figures_dir_even_for_seed_runs(tmp_path: Path):
    run_args = _make_required_args(tmp_path, seed=1)

    gen = tg.TraceGenerator.__new__(tg.TraceGenerator)
    tg.setup_logging(name=tg.LOGGER_NAME, prefix=f"[{tg.LOGGER_NAME}] ", level=run_args.log_level)
    gen.init_from_args(run_args, create_figures_dir=True, log_args=False)

    assert gen.output_dir.exists()
    assert gen.figures_dir.exists()
    assert gen.util_plot_path.name == "utilization.png"
    assert gen.hist_plot_path.name == "histograms.png"
    assert gen.times == []
    assert gen.u_req_hist == []
    assert gen.pods_hist == []
    assert gen.initial_pods_count == 0


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
    assert "xmax_arrival" in seen["include"]
    assert "xmax_life" in seen["include"]
    assert "target_util" in seen["include"]


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

    gen.write_info_file(extra={"k": 1})

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

    old_propagate = tg.LOG.propagate
    tg.LOG.propagate = True
    try:
        caplog.set_level("WARNING", logger=tg.LOGGER_NAME)
        gen.write_info_file(extra={})
    finally:
        tg.LOG.propagate = old_propagate

    assert any("failed to write info_generate.yaml" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Bounded Pareto: sampling + mean + alpha solve
# ---------------------------------------------------------------------------

def test_sample_bounded_pareto_validates_and_respects_bounds():
    rng = tg.np.random.default_rng(123)

    with pytest.raises(ValueError):
        tg.TraceGenerator.sample_bounded_pareto(rng, alpha=0.0, x_min=1.0, x_max=2.0, size=10)
    with pytest.raises(ValueError):
        tg.TraceGenerator.sample_bounded_pareto(rng, alpha=1.0, x_min=0.0, x_max=2.0, size=10)
    with pytest.raises(ValueError):
        tg.TraceGenerator.sample_bounded_pareto(rng, alpha=1.0, x_min=2.0, x_max=1.0, size=10)

    x = tg.TraceGenerator.sample_bounded_pareto(rng, alpha=2.0, x_min=1.0, x_max=2.0, size=5000)
    assert x.shape == (5000,)
    assert (x >= 1.0).all()
    assert (x <= 2.0 + 1e-9).all()


def test_bounded_pareto_mean_validates_and_matches_alpha_one_formula():
    with pytest.raises(ValueError):
        tg.TraceGenerator.bounded_pareto_mean(alpha=1.0, x_min=0.0, x_max=2.0)
    with pytest.raises(ValueError):
        tg.TraceGenerator.bounded_pareto_mean(alpha=1.0, x_min=2.0, x_max=1.0)

    x_min, x_max = 2.0, 10.0
    m1 = tg.TraceGenerator.bounded_pareto_mean(alpha=1.0, x_min=x_min, x_max=x_max)
    expected = (x_max * x_min / (x_max - x_min)) * math.log(x_max / x_min)
    assert math.isclose(m1, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_bounded_pareto_max_mean_validates_and_is_between_bounds():
    with pytest.raises(ValueError):
        tg.TraceGenerator.bounded_pareto_max_mean(x_min=0.0, x_max=2.0)
    with pytest.raises(ValueError):
        tg.TraceGenerator.bounded_pareto_max_mean(x_min=2.0, x_max=2.0)

    mmax = tg.TraceGenerator.bounded_pareto_max_mean(x_min=2.0, x_max=10.0)
    assert 2.0 < mmax < 10.0


def test_solve_alpha_for_bounded_mean_validates_ranges_and_converges():
    with pytest.raises(ValueError):
        tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=0.0, x_max=2.0, target_mean=1.0)
    with pytest.raises(ValueError):
        tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=2.0, x_max=2.0, target_mean=1.0)
    with pytest.raises(ValueError):
        tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=1.0, x_max=2.0, target_mean=1.0)  # must be in (x_min, x_max)
    with pytest.raises(ValueError):
        tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=1.0, x_max=2.0, target_mean=3.0)

    x_min, x_max = 0.1, 1.0
    target = 0.2
    alpha = tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=x_min, x_max=x_max, target_mean=target)
    got = tg.TraceGenerator.bounded_pareto_mean(alpha, x_min, x_max)
    assert got == pytest.approx(target, rel=1e-8, abs=1e-10)


def test_solve_alpha_for_bounded_mean_rejects_target_above_max_mean():
    x_min, x_max = 2.0, 10.0
    mean_max = tg.TraceGenerator.bounded_pareto_max_mean(x_min, x_max)
    with pytest.raises(ValueError):
        tg.TraceGenerator.solve_alpha_for_bounded_mean(x_min=x_min, x_max=x_max, target_mean=mean_max)


# ---------------------------------------------------------------------------
# Discrete helpers (priority/replicas)
# ---------------------------------------------------------------------------

def test_build_trunc_geometric_support_validates_and_returns_probs():
    with pytest.raises(ValueError):
        tg.TraceGenerator.build_trunc_geometric_support(1, 2, ratio=0.0)

    vals, probs = tg.TraceGenerator.build_trunc_geometric_support(3, 3, ratio=0.5)
    assert vals.tolist() == [3]
    assert probs is None

    vals, probs = tg.TraceGenerator.build_trunc_geometric_support(1, 3, ratio=1.0)
    assert vals.tolist() == [1, 2, 3]
    assert probs is None

    vals, probs = tg.TraceGenerator.build_trunc_geometric_support(1, 4, ratio=0.5)
    assert vals.tolist() == [1, 2, 3, 4]
    assert probs is not None
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probs[0] >= probs[-1]

    # max < min => clamp to min only
    vals, probs = tg.TraceGenerator.build_trunc_geometric_support(5, 1, ratio=0.5)
    assert vals.tolist() == [5]
    assert probs is None


def test_expected_value_handles_uniform_and_weighted():
    vals = tg.np.array([1, 2, 3], dtype=int)
    assert tg.TraceGenerator.expected_value(vals, None) == pytest.approx(2.0)

    probs = tg.np.array([0.5, 0.25, 0.25], dtype=float)
    assert tg.TraceGenerator.expected_value(vals, probs) == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Mean-life inference and alpha fitting
# ---------------------------------------------------------------------------

def test_infer_mean_life_from_target_util_happy_path(tmp_path: Path):
    args = _make_required_args(
        tmp_path,
        num_nodes=2,
        target_util=0.5,
        mean_arrival=2.0,
        mean_req=0.25,
        replicas_min=1,
        replicas_max=1,
        replicas_ratio=1.0,
        xmin_life=2.0,
        xmax_life=100.0,
    )
    gen = tg.TraceGenerator(args)
    mean_life = gen.infer_mean_life_from_target_util()
    assert mean_life > 0.0
    assert float(args.xmin_life) < mean_life < float(args.xmax_life)


def test_infer_mean_life_from_target_util_raises_if_outside_bounds(tmp_path: Path):
    # Too small => <= xmin-life
    args_small = _make_required_args(
        tmp_path,
        num_nodes=1,
        target_util=0.01,
        mean_arrival=0.1,
        mean_req=1.0,
        replicas_min=2,
        replicas_max=2,
        replicas_ratio=1.0,
        xmin_life=2.0,
        xmax_life=100.0,
    )
    gen_small = tg.TraceGenerator(args_small)
    with pytest.raises(ValueError):
        gen_small.infer_mean_life_from_target_util()

    # Too large => >= xmax-life
    args_big = _make_required_args(
        tmp_path,
        num_nodes=50,
        target_util=1.0,
        mean_arrival=10.0,
        mean_req=0.1,
        replicas_min=1,
        replicas_max=1,
        replicas_ratio=1.0,
        xmin_life=2.0,
        xmax_life=100.0,
    )
    gen_big = tg.TraceGenerator(args_big)
    with pytest.raises(ValueError):
        gen_big.infer_mean_life_from_target_util()


def test_fit_alphas_sets_fields_and_attaches_to_args(tmp_path: Path):
    args = _make_required_args(tmp_path, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    gen.fit_alphas()

    assert gen.alpha_req is not None
    assert gen.alpha_arrival is not None
    assert gen.alpha_life is not None

    assert hasattr(args, "alpha_req")
    assert hasattr(args, "alpha_arrival")
    assert hasattr(args, "alpha_life")

    # If mean_life changes, only alpha_life should recompute (req/arrival cached)
    old_req = gen.alpha_req
    old_arr = gen.alpha_arrival
    old_life = gen.alpha_life

    gen.args.mean_life = 12.0
    gen.fit_alphas()

    assert gen.alpha_req == old_req
    assert gen.alpha_arrival == old_arr
    assert gen.alpha_life != old_life


# ---------------------------------------------------------------------------
# Util metric
# ---------------------------------------------------------------------------

def test_time_avg_req_util_basic(tmp_path: Path):
    args = _make_required_args(tmp_path, num_nodes=2, trace_time="10s")
    gen = tg.TraceGenerator(args)

    pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=10.0, cpu=0.5, mem=0.5, priority=1, replicas=1),
    ]
    # area = 0.5*10, divide by N*T = 2*10 => 0.25
    assert gen.time_avg_req_util(pods) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Initial snapshot
# ---------------------------------------------------------------------------

def test_build_initial_snapshot_can_overshoot_then_pop(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, target_util=0.1, mean_life=10.0)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    def const_sample(_rng, *, alpha, x_min, x_max, size=1):
        # Always return a value that causes overshoot in req and a valid lifetime.
        # We'll use x_min/x_max to decide which one is being requested by bounds.
        # Simpler: just return 0.2 for everything.
        return tg.np.full(size, 0.2, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "sample_bounded_pareto", staticmethod(const_sample))

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, next_id = gen.build_initial_snapshot(
        tg.np.random.default_rng(0),
        prio_vals=prio_vals,
        prio_probs=None,
        rep_vals=rep_vals,
        rep_probs=None,
        next_id=0,
    )

    assert pods == []
    assert next_id == 0


def test_build_initial_snapshot_returns_some_pods(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, target_util=0.4, mean_life=10.0)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    def const_sample(_rng, *, alpha, x_min, x_max, size=1):
        # Use a stable request and lifetime.
        return tg.np.full(size, 0.2, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "sample_bounded_pareto", staticmethod(const_sample))

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, next_id = gen.build_initial_snapshot(
        tg.np.random.default_rng(1),
        prio_vals=prio_vals,
        prio_probs=None,
        rep_vals=rep_vals,
        rep_probs=None,
        next_id=0,
    )

    assert len(pods) >= 1
    assert next_id == len(pods)
    assert all(p.start_time == 0.0 for p in pods)
    assert all(p.end_time > 0.0 for p in pods)


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------

def test_generate_trace_events_seeds_baseline_and_pops_end_heap(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=1, trace_time="2.5s", mean_life=10.0)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, alpha=2.0)

    # dt1, life1, req1, dt2, life2, req2, dt3...
    samples = [
        1.0, 0.5, 0.2,
        1.0, 0.5, 0.2,
        10.0,
    ]

    def seq_sample(_rng, *, alpha, x_min, x_max, size=1):
        v = float(samples.pop(0))
        return tg.np.full(size, v, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "sample_bounded_pareto", staticmethod(seq_sample))

    initial_pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=0.5, cpu=0.1, mem=0.1, priority=1, replicas=1)
    ]

    prio_vals = tg.np.array([1], dtype=int)
    rep_vals = tg.np.array([1], dtype=int)

    pods, _, times, u_hist, pods_hist = gen.generate_trace_events(
        tg.np.random.default_rng(0),
        prio_vals=prio_vals,
        prio_probs=None,
        rep_vals=rep_vals,
        rep_probs=None,
        next_id=1,
        initial_pods=initial_pods,
    )

    assert len(pods) == 2
    assert times == pytest.approx([0.0, 1.0, 2.0])
    assert len(u_hist) == len(times)
    assert pods_hist == [1, 1, 1]


# ---------------------------------------------------------------------------
# One generation pass
# ---------------------------------------------------------------------------

def test_generate_once_sets_series_and_extra_info(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, num_nodes=2, trace_time="5s", mean_life=10.0)
    gen = tg.TraceGenerator(args)

    # Avoid the real alpha solve here; just set values and attach to args.
    def fake_fit_alphas():
        _set_alphas(gen, alpha=2.0)

    monkeypatch.setattr(gen, "fit_alphas", fake_fit_alphas)
    monkeypatch.setattr(gen, "time_avg_req_util", lambda _pods: 0.55)

    initial_pods = [
        tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=2),
    ]
    trace_pods = [
        tg.TraceRecord(id=2, start_time=1.0, end_time=2.0, cpu=0.2, mem=0.2, priority=1, replicas=1),
    ]

    times = [0.0, 1.0]
    u_hist = [0.1, 0.2]
    pods_hist = [2, 1]

    monkeypatch.setattr(gen, "build_initial_snapshot", lambda *_a, **_k: (initial_pods, 1))
    monkeypatch.setattr(gen, "generate_trace_events", lambda *_a, **_k: (trace_pods, 2, times, u_hist, pods_hist))

    got_initial, got_trace, util, extra = gen.generate_once(iter_seed=1234)

    assert got_initial == initial_pods
    assert got_trace == trace_pods
    assert util == pytest.approx(0.55)

    assert gen.times == times
    assert gen.u_req_hist == u_hist
    assert gen.pods_hist == pods_hist
    assert gen.initial_pods_count == 2

    assert extra["seed"] == int(args.seed)
    assert extra["measured_util_time_avg"] == pytest.approx(0.55)
    assert "rng" in extra and "derived" in extra["rng"]
    assert "files" in extra and "initial_json" in extra["files"]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibrate_mean_life_stops_within_tolerance(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = []
    extra = {"ok": True}

    monkeypatch.setattr(gen, "generate_once", lambda _seed: (initial, trace, 0.5, extra))

    got_initial, got_trace, got_extra = gen.calibrate_mean_life()
    assert got_initial == initial
    assert got_trace == trace
    assert got_extra == extra


def test_calibrate_mean_life_updates_mean_life_and_clamps_to_xmax(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.9, xmin_life=2.0, xmax_life=15.0, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    calls = {"n": 0}

    def fake_generate_once(_seed):
        calls["n"] += 1
        measured = 0.1 if calls["n"] == 1 else float(args.target_util)
        return [], [], measured, {"iter": calls["n"]}

    monkeypatch.setattr(gen, "generate_once", fake_generate_once)
    monkeypatch.setattr(tg, "MEAN_LIFE_CALIBRATION_MAX_ITER", 2)

    gen.calibrate_mean_life()

    # After first iteration: new_mean=10*(0.9/0.1)=90 -> clamp to 0.999*xmax
    assert gen.args.mean_life == pytest.approx(15.0 * 0.999)


def test_calibrate_mean_life_raises_on_zero_measured_util(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(gen, "generate_once", lambda _seed: ([], [], 0.0, {}))

    with pytest.raises(RuntimeError):
        gen.calibrate_mean_life()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_write_json_writes_expected_shape(tmp_path: Path):
    pods = [
        tg.TraceRecord(id=1, start_time=0.1, end_time=0.2, cpu=0.3, mem=0.3, priority=1, replicas=2)
    ]
    out = tmp_path / "x.json"
    tg.TraceGenerator.write_json(out, pods)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["pods"]
    assert isinstance(data["pods"], list)
    assert data["pods"][0]["id"] == 1


def test_write_outputs_writes_json_and_info(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, target_util=0.5, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = [tg.TraceRecord(id=2, start_time=1.0, end_time=2.0, cpu=0.2, mem=0.2, priority=1, replicas=1)]
    extra = {"x": 1}

    calls = {"json": [], "info": 0}

    monkeypatch.setattr(gen, "time_avg_req_util", lambda _pods: 0.6)

    def fake_write_json(path: Path, pods_):
        calls["json"].append((path, len(pods_)))

    monkeypatch.setattr(gen, "write_json", fake_write_json)
    monkeypatch.setattr(gen, "write_info_file", lambda extra: calls.__setitem__("info", calls["info"] + 1))

    gen.write_outputs(initial, trace, extra)

    assert calls["json"][0][0].name == "initial.json"
    assert calls["json"][1][0].name == "trace.json"
    assert calls["json"][0][1] == 1
    assert calls["json"][1][1] == 1
    assert calls["info"] == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_run_seed_infers_mean_life_writes_and_calls_plot_helpers(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, mean_life=None)
    gen = tg.TraceGenerator(args)

    # Ensure args.alpha_* exist for plotting call sites.
    gen.args.alpha_arrival = 2.0
    gen.args.alpha_life = 2.0
    gen.args.alpha_req = 2.0

    monkeypatch.setattr(gen, "infer_mean_life_from_target_util", lambda: 12.3)

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    trace = []
    extra = {"ok": True}

    monkeypatch.setattr(gen, "calibrate_mean_life", lambda: (initial, trace, extra))
    monkeypatch.setattr(gen, "write_outputs", lambda *_a, **_k: None)

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

    monkeypatch.setattr(tg, "plot_utilization_and_num_pods", fake_util)
    monkeypatch.setattr(tg, "plot_generator_histograms", fake_hist)

    gen.run_seed()

    assert gen.args.mean_life == pytest.approx(12.3)
    assert called["util"] == 1
    assert called["hist"] == 1


def test_run_seed_skips_calibration_when_mean_life_provided(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(gen, "calibrate_mean_life", lambda: (_ for _ in ()).throw(AssertionError("should not calibrate")))

    seen = {"iter_seed": None, "called": 0}

    initial = [tg.TraceRecord(id=1, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=0, replicas=1)]
    trace = [tg.TraceRecord(id=2, start_time=0.5, end_time=1.5, cpu=0.2, mem=0.2, priority=0, replicas=1)]

    def fake_generate_once(iter_seed: int):
        seen["iter_seed"] = iter_seed
        seen["called"] += 1
        # generate_once() normally fits alphas and stores them on args for plotting call sites.
        gen.args.alpha_arrival = 2.0
        gen.args.alpha_life = 2.0
        gen.args.alpha_req = 2.0
        return initial, trace, 0.1234, {"k": "v"}

    monkeypatch.setattr(gen, "generate_once", fake_generate_once)
    monkeypatch.setattr(gen, "write_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(tg, "plot_utilization_and_num_pods", lambda **_k: None)
    monkeypatch.setattr(tg, "plot_generator_histograms", lambda **_k: None)

    gen.run_seed()

    assert seen["called"] == 1
    assert seen["iter_seed"] is not None
    assert gen.args.mean_life == pytest.approx(10.0)


def test_run_invokes_run_seed_per_expanded_seed(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path, seed=1, mean_life=10.0)
    gen = tg.TraceGenerator(args)

    r1 = _make_required_args(tmp_path / "out1", seed=1, mean_life=10.0)
    r2 = _make_required_args(tmp_path / "out2", seed=2, mean_life=10.0)

    monkeypatch.setattr(tg.TraceGenerator, "expand_seed_runs", staticmethod(lambda _a: [r1, r2]))
    monkeypatch.setattr(tg, "make_header_footer", lambda _t: ("H", "F"))

    called = {"runs": []}

    def fake_run_seed(self):
        called["runs"].append((int(self.args.seed), str(self.args.output_dir)))

    monkeypatch.setattr(tg.TraceGenerator, "run_seed", fake_run_seed)

    gen.run()
    assert called["runs"] == [(1, str(r1.output_dir)), (2, str(r2.output_dir))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_main_smoke_invokes_generator(monkeypatch, tmp_path: Path):
    args = _make_required_args(tmp_path)

    class DummyParser:
        def parse_args(self):
            return args

    monkeypatch.setattr(tg, "build_arg_parser", lambda: DummyParser())

    seen = {"ran": False}

    class FakeGen:
        def __init__(self, _cli_args):
            pass

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
