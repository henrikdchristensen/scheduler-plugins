#!/usr/bin/env python3
"""
Generate KWOK trace replayer job files for all combinations.

Usage:
  python make_kwok_trace_jobs.py --out-dir data/jobs/kwok_trace_replayer

Creates:
  <out-dir>/default/*.yaml
  <out-dir>/plugin/*.yaml
"""

#TODO: TO BE DELETED BEFORE SUBMISSION

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

# ----------------------------- Constants -----------------------------

NODES = [16, 32]
PRIOS = [1, 4]
ARRIVALS_S = [8, 16, 32]

BLOCKING_VALUES = [0, 1]
DEFPREEMPT_VALUES = [0, 1]

DEFAULT_KWOKCTL_CONFIG = "data/configs-kwokctl/default.yaml"
# NEW convention:
#   data/configs-kwokctl/plugin-scheduler-defpreempt=0.yaml
#   data/configs-kwokctl/plugin-scheduler-defpreempt=1.yaml
PLUGIN_KWOKCTL_CONFIG_TEMPLATE = "data/configs-kwokctl/plugin-scheduler-defpreempt={defpreempt}.yaml"

SOLVER_TIMEOUT = "16s"

# ----------------------------- YAML helpers -----------------------------

def _yaml_quote(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'

def render_job_yaml(
    trace_dir: str,
    result_dir: str,
    kwokctl_config_file: str,
    override_envs: List[Tuple[str, str]] | None = None,
) -> str:
    lines: List[str] = [
        f"trace-dir: {trace_dir}",
        f"result-dir: {result_dir}",
        f"kwokctl-config-file: {kwokctl_config_file}",
    ]

    if override_envs:
        lines.append("override-kwokctl-envs:")
        for name, value in override_envs:
            lines.append(f"  - name: {name}")
            lines.append(f"    value: {_yaml_quote(value)}")

    return "\n".join(lines) + "\n"

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# ----------------------------- Path conventions -----------------------------

def trace_dir(nodes: int, prio: int, arrival_s: int) -> str:
    return f"data/traces/nodes={nodes}_prio={prio}_arrival={arrival_s}s"

def default_result_dir(nodes: int, prio: int, arrival_s: int) -> str:
    return f"results/default/nodes={nodes}_prio={prio}_arrival={arrival_s}s"

def plugin_result_dir(base_filename_no_ext: str) -> str:
    return f"results/plugin/{base_filename_no_ext}"

def plugin_config_for(defpreempt: int) -> str:
    return PLUGIN_KWOKCTL_CONFIG_TEMPLATE.format(defpreempt=defpreempt)

# ----------------------------- Generators -----------------------------

def iter_default_jobs(out_dir: Path) -> Iterable[Path]:
    for n in NODES:
        for p in PRIOS:
            for a in ARRIVALS_S:
                fname = f"nodes={n}_prio={p}_arrival={a}s.yaml"
                path = out_dir / "default" / fname

                yml = render_job_yaml(
                    trace_dir=trace_dir(n, p, a),
                    result_dir=default_result_dir(n, p, a),
                    kwokctl_config_file=DEFAULT_KWOKCTL_CONFIG,
                    override_envs=None,
                )
                write_text(path, yml)
                yield path

def iter_plugin_jobs(out_dir: Path) -> Iterable[Path]:
    for n in NODES:
        for p in PRIOS:
            for a in ARRIVALS_S:
                # schedulingfailure (no blocking/defpreempt)
                # Keep using the non-defpreempt-specific config unless you want otherwise.
                base = f"mode=schedulingfailure_nodes={n}_prio={p}_arrival={a}s"
                path = out_dir / "plugin" / f"{base}.yaml"
                yml = render_job_yaml(
                    trace_dir=trace_dir(n, p, a),
                    result_dir=plugin_result_dir(base),
                    kwokctl_config_file="data/configs-kwokctl/plugin-scheduler.yaml",
                    override_envs=[
                        ("SOLVER_PYTHON_TIMEOUT", SOLVER_TIMEOUT),
                        ("OPTIMIZE_MODE", "scheduling_failure"),
                    ],
                )
                write_text(path, yml)
                yield path

                # periodic8s and stablequeue2s (with blocking + defpreempt)
                for blocking in BLOCKING_VALUES:
                    blocking_str = "true" if blocking == 1 else "false"

                    for defpreempt in DEFPREEMPT_VALUES:
                        kwokcfg = plugin_config_for(defpreempt)

                        # periodic8s
                        base = (
                            f"mode=periodic8s_blocking={blocking}_defpreempt={defpreempt}"
                            f"_nodes={n}_prio={p}_arrival={a}s"
                        )
                        path = out_dir / "plugin" / f"{base}.yaml"
                        yml = render_job_yaml(
                            trace_dir=trace_dir(n, p, a),
                            result_dir=plugin_result_dir(base),
                            kwokctl_config_file=kwokcfg,
                            override_envs=[
                                ("SOLVER_PYTHON_TIMEOUT", SOLVER_TIMEOUT),
                                ("OPTIMIZE_MODE", "periodic"),
                                ("OPTIMIZE_BLOCKING_SOLVING", blocking_str),
                                ("OPTIMIZE_PERIODIC_INTERVAL", "8s"),
                            ],
                        )
                        write_text(path, yml)
                        yield path

                        # stablequeue2s
                        base = (
                            f"mode=stablequeue2s_blocking={blocking}_defpreempt={defpreempt}"
                            f"_nodes={n}_prio={p}_arrival={a}s"
                        )
                        path = out_dir / "plugin" / f"{base}.yaml"
                        yml = render_job_yaml(
                            trace_dir=trace_dir(n, p, a),
                            result_dir=plugin_result_dir(base),
                            kwokctl_config_file=kwokcfg,
                            override_envs=[
                                ("SOLVER_PYTHON_TIMEOUT", SOLVER_TIMEOUT),
                                ("OPTIMIZE_MODE", "stable_queue"),
                                ("OPTIMIZE_BLOCKING_SOLVING", blocking_str),
                                ("OPTIMIZE_STABLE_QUEUE_DELAY", "2s"),
                            ],
                        )
                        write_text(path, yml)
                        yield path

# ----------------------------- CLI -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path,
        help="Output directory root, e.g. data/jobs/kwok_trace_replayer",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir

    written = 0
    for _ in iter_default_jobs(out_dir):
        written += 1
    for _ in iter_plugin_jobs(out_dir):
        written += 1

    print(f"Wrote {written} job files under: {out_dir}")
    print(f"  - {out_dir / 'default'}")
    print(f"  - {out_dir / 'plugin'}")

if __name__ == "__main__":
    main()
