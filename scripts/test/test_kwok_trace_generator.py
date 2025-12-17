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
        output_dir=str(tmp_path),
        seed=42,
        log_level="INFO",
        num_nodes=2,
        trace_time="5s",
        xmin_arrival=0.1,
        xmax_arrival=10.0,
        mean_arrival=1.0,
        xmin_life=0.1,
        xmax_life=10.0,
        mean_life=2.0,
        xmin_cpu=0.1,
        xmax_cpu=1.0,
        mean_cpu=0.2,
        xmin_mem=0.1,
        xmax_mem=1.0,
        mean_mem=0.3,
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
    gen.alpha_cpu = alpha
    gen.alpha_mem = alpha
    gen.alpha_arrival = alpha
    gen.alpha_life = alpha
    gen.args.alpha_cpu = alpha
    gen.args.alpha_mem = alpha
    gen.args.alpha_arrival = alpha
    gen.args.alpha_life = alpha


def sample_for_alpha(_rng, alpha, x_min, x_max, size=1):
    v = float(x_min + (1.0 / float(alpha)))
    if x_max is not None:
        v = min(v, float(x_max))
    return tg.np.full(size, v, dtype=float)


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------

def test_build_arg_parser_defaults_and_required_means(tmp_path: Path):
    p = tg.build_arg_parser()
    args = p.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--mean-arrival", "1.0",
            "--mean-life", "10.0",
            "--mean-cpu", "0.2",
            "--mean-mem", "0.3",
        ]
    )

    assert args.output_dir == str(tmp_path)
    assert args.seed == 42
    assert args.num_nodes == 8
    assert args.trace_time == "3600s"
    assert args.priority_min == 1
    assert args.priority_max == 3
    assert args.replicas_min == 1
    assert args.replicas_max == 1


@pytest.mark.parametrize(
    "missing_flag",
    ["--mean-arrival", "--mean-life", "--mean-cpu", "--mean-mem"],
)
def test_build_arg_parser_missing_required_exits(tmp_path: Path, missing_flag: str):
    p = tg.build_arg_parser()
    argv = [
        "--output-dir", str(tmp_path),
        "--mean-arrival", "1.0",
        "--mean-life", "10.0",
        "--mean-cpu", "0.2",
        "--mean-mem", "0.3",
    ]
    i = argv.index(missing_flag)
    del argv[i:i + 2]

    with pytest.raises(SystemExit):
        p.parse_args(argv)


# ---------------------------------------------------------------------------
# round_float_args
# ---------------------------------------------------------------------------

def test_round_float_args_rounds_only_floats():
    ns = argparse.Namespace(a=1.23456, b=2, c="x", d=3.14159)
    tg.round_float_args(ns, ndigits=2)
    assert ns.a == 1.23
    assert ns.b == 2
    assert ns.c == "x"
    assert ns.d == 3.14


# ---------------------------------------------------------------------------
# TraceGenerator init/prepare + info/log helpers
# ---------------------------------------------------------------------------

def test_log_args_calls_log_args_block_and_preserves_order(tmp_path: Path, monkeypatch):
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
    
    assert seen["include"][:5] == ["output_dir", "seed", "log_level", "num_nodes", "trace_time"]
    assert "mean_cpu" in seen["include"]
    assert "mean_mem" in seen["include"]
    assert "mean_arrival" in seen["include"]
    assert "mean_life" in seen["include"]


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

    gen._write_info_file()

    assert str(called["out_path"]).endswith("info_generate.yaml")
    assert called["inputs"]["cli-cmd"] == ["python", "trace_generator.py", "--x"]
    assert "args" in called["inputs"]
    assert called["logger"] is tg.LOG


def test_write_info_file_exception_logs_warning(tmp_path: Path, monkeypatch, caplog):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(tg, "build_cli_cmd", lambda: ["x"])
    monkeypatch.setattr(tg, "write_info_file", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    caplog.set_level("WARNING", logger=tg.LOGGER_NAME)
    gen._write_info_file()

    assert any("failed to write info_generate.yaml" in rec.message for rec in caplog.records)


def test_generator_ensure_prepared_runs_once(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    counts = {"info": 0, "log": 0, "fit": 0}

    monkeypatch.setattr(gen, "_write_info_file", lambda: counts.__setitem__("info", counts["info"] + 1))
    monkeypatch.setattr(gen, "log_args", lambda: counts.__setitem__("log", counts["log"] + 1))
    monkeypatch.setattr(gen, "_fit_alphas", lambda: (counts.__setitem__("fit", counts["fit"] + 1), _set_alphas(gen, 2.0)))

    assert gen._prepared is False
    gen._ensure_prepared()
    assert gen._prepared is True
    gen._ensure_prepared()
    assert counts == {"info": 1, "log": 1, "fit": 1}


# ---------------------------------------------------------------------------
# TraceGenerator._sample_pareto
# ---------------------------------------------------------------------------

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

    # also hit the "no x_max" branch explicitly
    x2 = tg.TraceGenerator._sample_pareto(rng, alpha=2.0, x_min=1.0, x_max=None, size=10)
    assert (x2 >= 1.0).all()


# ---------------------------------------------------------------------------
# TraceGenerator._solve_alpha_of_pareto_for_mean
# ---------------------------------------------------------------------------

def test_solve_alpha_validates_target_mean_ranges():
    rng = tg.np.random.default_rng(0)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_of_pareto_for_mean(rng, x_min=0.0, x_max=1.0, target_mean=0.5)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_of_pareto_for_mean(rng, x_min=1.0, x_max=None, target_mean=1.0)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_of_pareto_for_mean(rng, x_min=1.0, x_max=2.0, target_mean=3.0)


def test_solve_alpha_bracket_failure_raises(monkeypatch):
    rng = tg.np.random.default_rng(0)

    def constant_sample(_rng, _alpha, _x_min, _x_max, size=1):
        return tg.np.full(size, 1.0, dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(constant_sample))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 10)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 10)

    with pytest.raises(ValueError):
        tg.TraceGenerator._solve_alpha_of_pareto_for_mean(rng, x_min=0.5, x_max=2.0, target_mean=1.5)


def test_solve_alpha_converges_quickly_with_stubbed_sampling(monkeypatch):
    rng = tg.np.random.default_rng(0)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 200)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_LOWER_BOUND", 0.1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_UPPER_BOUND", 10.0)

    alpha = tg.TraceGenerator._solve_alpha_of_pareto_for_mean(
        rng,
        x_min=0.5,
        x_max=10.0,
        target_mean=0.7,
    )
    assert 0.1 <= alpha <= 10.0


def test_solve_alpha_unbounded_hits_alpha_low_adjust(monkeypatch):
    rng = tg.np.random.default_rng(0)

    # Make sampling cheap + monotone
    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 50)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_LOWER_BOUND", 0.1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_UPPER_BOUND", 10.0)

    # x_max=None should force alpha_low to >= 1.0001
    alpha = tg.TraceGenerator._solve_alpha_of_pareto_for_mean(
        rng,
        x_min=0.5,
        x_max=None,
        target_mean=0.7,
    )
    assert alpha >= 1.0


# ---------------------------------------------------------------------------
# TraceGenerator._fit_alphas
# ---------------------------------------------------------------------------

def test_fit_alphas_sets_fields_and_attaches_to_args(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(sample_for_alpha))
    monkeypatch.setattr(tg, "SOLVE_ALPHA_SAMPLES", 1)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_MAX_ITERATIONS", 100)
    monkeypatch.setattr(tg, "SOLVE_ALPHA_TOLERANCE", 1e-6)

    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    gen._fit_alphas()

    assert gen.alpha_cpu is not None
    assert gen.alpha_mem is not None
    assert gen.alpha_arrival is not None
    assert gen.alpha_life is not None
    assert hasattr(args, "alpha_cpu")
    assert hasattr(args, "alpha_mem")
    assert hasattr(args, "alpha_arrival")
    assert hasattr(args, "alpha_life")


# ---------------------------------------------------------------------------
# TraceGenerator._build_geometric_support
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TraceGenerator._delete_completed_pods
# ---------------------------------------------------------------------------

def test_delete_completed_pods_updates_cluster_state():
    state = tg.ClusterState(num_nodes=1, live_cpu=3.0, live_mem=4.0, live_pods=3)

    end_heap = [
        tg.EndHeapEntry(end_time=1.0, cpu=1.0, mem=1.0, replicas=1),
        tg.EndHeapEntry(end_time=2.0, cpu=0.5, mem=0.5, replicas=2),
        tg.EndHeapEntry(end_time=5.0, cpu=1.0, mem=2.0, replicas=1),
    ]
    tg.heapq.heapify(end_heap)

    tg.TraceGenerator._delete_completed_pods(state, end_heap, t=2.0)

    assert len(end_heap) == 1
    assert state.live_cpu == pytest.approx(3.0 - 1.0 * 1 - 0.5 * 2)
    assert state.live_mem == pytest.approx(4.0 - 1.0 * 1 - 0.5 * 2)
    assert state.live_pods == 0


# ---------------------------------------------------------------------------
# TraceGenerator._summarize_utilization
# ---------------------------------------------------------------------------

def test_summarize_utilization_handles_empty_snapshots():
    gen = tg.TraceGenerator.__new__(tg.TraceGenerator)
    gen.times = []
    gen.u_cpu_hist = []
    gen.u_mem_hist = []
    tg.TraceGenerator._summarize_utilization(gen)


# ---------------------------------------------------------------------------
# TraceGenerator plotting helpers
# ---------------------------------------------------------------------------

def test_plot_trace_utilization_writes_png(tmp_path: Path):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    gen.times = [0.0, 0.5, 1.0, 2.0]
    gen.u_cpu_hist = [0.0, 0.1, 0.2, 0.15]
    gen.u_mem_hist = [0.0, 0.05, 0.25, 0.2]
    gen.max_runnable_pods_hist = [0, 1, 2, 1]
    gen.pods = [
        tg.TraceRecord(id=1, start_time=0.25, end_time=0.75, cpu=0.1, mem=0.1, priority=1, replicas=1),
        tg.TraceRecord(id=2, start_time=1.25, end_time=1.75, cpu=0.2, mem=0.2, priority=2, replicas=1),
    ]

    gen._plot_trace_utilization()
    assert gen.util_plot_path.exists()


def test_plot_generated_histograms_writes_png(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)
    _set_alphas(gen, 2.0)

    gen.pods = [
        tg.TraceRecord(id=1, start_time=0.25, end_time=1.25, cpu=0.1, mem=0.2, priority=1, replicas=1),
        tg.TraceRecord(id=2, start_time=0.75, end_time=2.75, cpu=0.2, mem=0.3, priority=2, replicas=2),
    ]

    def fake_hist(ax, data, **kwargs):
        ax.set_title(kwargs.get("title", "hist"))
        ax.set_xlabel(kwargs.get("x_label", "x"))
        ax.set_ylabel(kwargs.get("y_label", "y"))

    def fake_bar(ax, data, **kwargs):
        ax.set_title(kwargs.get("title", "bar"))
        ax.set_xlabel(kwargs.get("x_label", "x"))
        ax.set_ylabel(kwargs.get("y_label", "y"))

    monkeypatch.setattr(tg, "plot_histogram_with_pareto", fake_hist)
    monkeypatch.setattr(tg, "plot_bar_with_geometric", fake_bar)

    gen._plot_generated_histograms()
    assert gen.hist_plot_path.exists()


# ---------------------------------------------------------------------------
# TraceGenerator._generate_trace / _write_trace / run
# ---------------------------------------------------------------------------

def test_generate_trace_write_trace_and_run(tmp_path: Path, monkeypatch):
    args = _make_required_args(tmp_path)

    gen = tg.TraceGenerator(args)

    monkeypatch.setattr(gen, "_write_info_file", lambda: None)
    monkeypatch.setattr(gen, "log_args", lambda: None)

    def fake_fit():
        _set_alphas(gen, alpha=2.0)

    monkeypatch.setattr(gen, "_fit_alphas", fake_fit)

    def fixed_sample(_rng, _alpha, x_min, _x_max, size=1):
        return tg.np.full(size, float(x_min), dtype=float)

    monkeypatch.setattr(tg.TraceGenerator, "_sample_pareto", staticmethod(fixed_sample))

    # Avoid plotting (covered above in dedicated plotting tests)
    monkeypatch.setattr(tg.TraceGenerator, "_plot_trace_utilization", lambda self: None)
    monkeypatch.setattr(tg.TraceGenerator, "_plot_generated_histograms", lambda self: None)

    gen._generate_trace()
    assert gen.pods
    assert gen.times
    assert gen.u_cpu_hist
    assert gen.u_mem_hist
    assert gen.max_runnable_pods_hist

    gen._write_trace()
    assert (Path(args.output_dir) / "trace.json").exists()

    gen.run()
    assert (Path(args.output_dir) / "trace.json").exists()


def test_write_trace_writes_expected_json_shape(tmp_path: Path):
    args = _make_required_args(tmp_path)
    gen = tg.TraceGenerator(args)

    _set_alphas(gen, alpha=2.0)
    gen.pods = [
        tg.TraceRecord(id=1, start_time=0.1, end_time=0.2, cpu=0.3, mem=0.4, priority=1, replicas=2)
    ]
    gen.trace_time_s = 5.0

    gen._write_trace()
    data = json.loads((Path(args.output_dir) / "trace.json").read_text(encoding="utf-8"))

    assert "meta" in data
    assert "pods" in data
    assert isinstance(data["pods"], list)
    assert data["pods"][0]["id"] == 1
    assert data["meta"]["cpu_params"]["pareto_alpha"] == 2.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
