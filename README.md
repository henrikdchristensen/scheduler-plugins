
# Kubernetes Plugin for Optimized Scheduling

- [Kubernetes Plugin for Optimized Scheduling](#kubernetes-plugin-for-optimized-scheduling)
  - [Overview](#overview)
  - [Code Structure](#code-structure)
  - [Scheduler Integration](#scheduler-integration)
  - [Building the Scheduler and OPSche](#building-the-scheduler-and-opsche)
  - [Requirements for Running on a KWOK Cluster](#requirements-for-running-on-a-kwok-cluster)
  - [Evaluation of OPSche on a KWOK Cluster](#evaluation-of-opsche-on-a-kwok-cluster)
    - [Experimental Setup Used for This Analysis](#experimental-setup-used-for-this-analysis)
    - [Scheduler Setups](#scheduler-setups)
    - [Generating Traces](#generating-traces)
    - [Replaying Traces](#replaying-traces)
    - [Reproducing Evaluation Results](#reproducing-evaluation-results)
    - [Faster Evaluation Through Parallelization (HPC/VM)](#faster-evaluation-through-parallelization-hpcvm)
  - [Unit and Integration Tests](#unit-and-integration-tests)
  - [Upstream Version](#upstream-version)

## Overview

This project is a fork of the Kubernetes-sigs project [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) (see [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/) for background).

It introduces **OPSche** (denoted **MyPriorityOptimizer** in the code), a Kubernetes scheduler plugin that improves pod-node placements by delegating placement planning to an external **solver**, aiming for **optimal placements** when feasible. Given a solver-produced plan, OPSche enforces it by *evicting* and *relocating* pods, potentially across multiple nodes (**cross-node preemption**). This differs from the default Kubernetes scheduler, whose built-in preemption (*DefaultPreemption*) is limited to a *single node* and can therefore cause more preemptions than necessary and lead to *suboptimal* placements.

Specifically, OPSche implements the following extension points (also called **hooks**):

- **PreEnqueue** – temporarily blocks new pods from entering the scheduling queue while a placement plan is being applied.
- **PreFilter** – steers a pod to the node selected by the solver.
- **PostFilter** – triggers optimization after a scheduling failure (i.e., when the default scheduler cannot place a pod).
- **Reserve/Unreserve** – reserves and releases node resources according to the solver plan.

Beyond these hooks, OPSche also includes a **background loop** that can trigger optimization in two additional ways: periodically at fixed time intervals, or during stable-queue windows (i.e., when no new pods are arriving). Together, this provides three **trigger modes**, all of which execute the same optimization flow.

- **SchedulingFailure** – optimize after each failed scheduling attempt (generally not recommended for large clusters).
- **Periodic** – optimize running and pending pods at fixed time intervals.
- **StableQueue** – optimize during stable-queue windows (i.e., when no new pods are arriving for a certain time).

Moreover, the solver can run in either **blocking** or **non-blocking** mode:

- **Blocking** – regular scheduling of new pods is paused while the solver runs and while the resulting plan is enforced.
- **Non-blocking** – regular scheduling continues while the solver runs and is paused only during plan enforcement.

In both modes, once the solver finishes, OPSche re-checks the cluster state before enforcing the plan. If the state has changed and the plan is no longer valid, it is discarded.

To detect **plan completion** and verify that pods end up on the intended nodes, OPSche also includes a **background watcher** that tracks plan completion and checks for any discrepancies between the intended and actual placements.

The solver implementation uses a **priority-aware** optimization model. The objective is to schedule as many *high-priority* pods as possible while *minimizing disruption* (i.e., reducing reallocations and evictions). It is implemented as a **CP-SAT** solver using the Python API of [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver).

For instructions on **reproducing the evaluation results**, see [Reproducing Evaluation Results](#reproducing-evaluation-results) below.

## Code Structure

The source code for OPSche is located in `pkg/mypriorityoptimizer/`. The main files and their roles are:

- `plugin.go` – main OPSche entry point and setup.
- `args.go` and `constants.go` – OPSche configuration arguments and constants (e.g., trigger mode, solver timeout).
- `optimization_flow.go` – core *optimization flow*, including solver invocation and plan application.
- `loop_helpers.go`, `loop_periodic.go`, `loop_stable_queue.go` – *background-loop* logic for *Periodic* and *StableQueue* triggering.
- `solver_external.go` and `solver_python.go` – external solver invocation and parsing of solver output.
- `hook_preenqueue.go` – implementation of the *PreEnqueue* hook, mainly used to block new pods while optimization is running or a plan is being enforced.
- `hook_prefilter.go` – implementation of the *PreFilter* hook, used to steer a pod to the node assigned by the solver plan.
- `hook_postfilter.go` – implementation of the *PostFilter* hook, used to detect failed scheduling attempts; in *SchedulingFailure* mode, it also triggers optimization.
- `hook_reserve_unreserve.go` – implementation of the *Reserve/Unreserve* hooks, used to reserve/release resources according to the plan.
- `plan_completion_watch.go` – background watcher that tracks plan completion and verifies that pods end up on the intended nodes.

The solver implementation is located in `scripts/python_solver/` and can also serve as a template for adding other solvers.

## Scheduler Integration

Plugins in the Kubernetes scheduler are enabled through a **scheduler configuration manifest** that selects the plugin and its settings. The manifests for OPSche are located in `manifests/mypriorityoptimizer/` (see also [Scheduler Configuration](https://kubernetes.io/docs/reference/scheduling/config/) for background).

OPSche is registered in `cmd/scheduler/main.go`, which ensures it is included in the scheduler binary at build time.

## Building the Scheduler and OPSche

The scheduler and OPSche can be built into a **binary** that can then be run in a cluster (e.g., with [KWOK](https://kwok.sigs.k8s.io/)). The following tools are required to build the scheduler and OPSche:

- `make` (tested with 4.3)
- `Go` (tested with 1.24.3)
- `python3` (tested with 3.10.12)
- `pip` (tested with 24.0)

Currently, the project has been tested on **amd64** and **arm64**. Running on other architectures should not be a problem but has not been verified.

To **build** the binary, run the following script:

```bash
./build.sh
```

The resulting scheduler binary is written to `bin/kube-scheduler`. The build process also downloads `kube-apiserver` and `kube-controller-manager` into `bin/`, since they are required for running KWOK clusters; keeping them there avoids re-downloading them for each cluster creation.

## Requirements for Running on a KWOK Cluster

To run the scheduler with OPSche on a **KWOK** cluster, a few additional tools are required (tools already listed in [Building the Scheduler and OPSche](#building-the-scheduler-and-opsche) are omitted):

- `kubectl` (tested with client v1.32.7)
- `kwok` and `kwokctl` (tested with v0.7.0)

Running the scheduler and OPSche on KWOK also requires a **KWOK cluster configuration file**. An example can be found at `data/configs-kwokctl/plugin-scheduler-defpreempt=1.yaml`.

**Note:** There may be version mismatches between the scheduler-plugins project and KWOK. If so, set a specific `VERSION` value when building the scheduler binary (sometimes this must be forced for a given scheduler-plugins release). For example, KWOK may report:

```bash
# Running the command
kwokctl logs kube-scheduler --name kwok1
# may report
E1210 13:54:11.815001   22220 run.go:72] "command failed" err="[emulation version 0.33 is not between [1.31, 0.33.0], minCompatibilityVersion version 0.32 is not between [1.31, 0.33]]"
```

## Evaluation of OPSche on a KWOK Cluster

The **Trace Replayer** script (found at `scripts/kwok_trace_replayer/`) is used to evaluate the performance of OPSche relative to the default scheduler. It simulates workload arrivals and removals over time in a cluster and continuously evaluates the scheduler with OPSche. The evaluation consists of two steps: **trace generation** and **trace replay**. Traces are first generated, then replayed with both the default scheduler and with OPSche integrated.

### Experimental Setup Used for This Analysis

The evaluations in this analysis were conducted on machines with the following specifications:

- **CPU:** 8 vCPUs (Intel Xeon Gold 6130)
- **Memory:** 48 GB RAM
- **Operating system:** Ubuntu 24.04
- **Solver library:** OR-Tools 9.14.6206

### Scheduler Setups

To compare the default scheduler against the OPSche approach, two cluster configurations are used:

1. **Default scheduler** – the default Kubernetes scheduler without OPSche, configured with `data/configs-kwokctl/default.yaml`.
2. **Scheduler with OPSche** – the scheduler with OPSche enabled and configured with `data/configs-kwokctl/plugin-scheduler-defpreempt=<1|0>.yaml` (1=default preemption enabled, 0=default preemption disabled).

### Generating Traces

Traces are generated with `scripts/kwok_trace_replayer/trace_generator.py` and stored under `data/traces/`.

The generator reads job files from `data/jobs/kwok_trace_generator/`, where each job file defines parameters such as the number of nodes, inter-arrival times, and other workload settings. From these job files, the script produces traces of workload arrivals and removals over time.

To generate traces, first install the required Python dependencies:

```bash
pip install -r scripts/kwok_trace_replayer/requirements.txt
```

Then generate traces from the job files:

```bash
python -m scripts.kwok_trace_replayer.trace_generator \
--job-dir data/jobs/kwok_trace_generator/
```

### Replaying Traces

After traces have been generated, they can be replayed with `scripts/kwok_trace_replayer/trace_replayer.py`. The replayer uses job files in `data/jobs/kwok_trace_replayer/`, which specify the trace to replay and how OPSche should be configured for that run.

To replay a trace, run:

```bash
python -m scripts.kwok_trace_replayer.trace_replayer \
--job-file data/jobs/kwok_trace_replayer/<job_file>.yaml
```

### Reproducing Evaluation Results

**Prerequisites:** Before proceeding, make sure you have completed the following steps:

1. Installed all required tools (see [Building the Scheduler and OPSche](#building-the-scheduler-and-opsche) and [Requirements for Running on a KWOK Cluster](#requirements-for-running-on-a-kwok-cluster)).
2. Built the scheduler binary with `./build.sh` (see [Building the Scheduler and OPSche](#building-the-scheduler-and-opsche)).

To **reproduce all evaluation results**, each job file under `data/jobs/kwok_trace_replayer/` must be executed using the `trace_replayer.py` script (see [Replaying Traces](#replaying-traces)). The job files are organized into subdirectories according to the scheduler configuration used for each run.

After all jobs have completed, organize the results in the `analysis/kwok_trace_replayer/` folder as follows.

```text
analysis/kwok_trace_replayer/
├── default/
│   ├── nodes=16_prio=1_arrival=1s/
│   │   ├── 1420052706459400740/
│   │   │   ├── general_stats.csv
│   │   │   ├── pod_stats.csv
│   │   │   └── info_replayer.yaml
│   │   └── ...
│   ├── nodes=16_prio=1_arrival=2s/
│   │   ├── ...
│   └── ...
└── plugin/
    ├── mode=periodic2s_blocking=0_defpreempt=0_nodes=16_prio=1_arrival=1s/
    │   ├── 1420052706459400740/
    │   │   ├── general_stats.csv
    │   │   ├── pod_stats.csv
    │   │   └── info_replayer.yaml
    │   └── ...
    ├── mode=stablequeue16s_blocking=1_defpreempt=1_nodes=32_prio=4_arrival=16s/
    │   ├── ...
    └── ...
```

Use the following scripts to aggregate the results and generate the figures and tables used in the analysis.

First, install the required Python dependencies:

```bash
pip install -r scripts/kwok_trace_replayer/requirements.txt
```

Next, combine the raw results into a single CSV file:

```bash
python -m scripts.kwok_trace_replayer.seal_results
```

Finally, generate the figures and tables:

```bash
python -m scripts.kwok_trace_replayer.plots_and_tables
```

The following scripts reproduce the aggregate optimality and statistical
analyses reported in the EuroSys paper:

```bash
python -m scripts.kwok_trace_replayer.optimality_stats
python -m scripts.kwok_trace_replayer.significance_test
```

Both scripts read `analysis/kwok_trace_replayer/results_seeds.csv`, generated by
`seal_results` above. The significance script uses each experimental run as the
unit of analysis and applies Holm correction for multiple comparisons.

### Faster Evaluation Through Parallelization (HPC/VM)

Because the evaluation consists of many jobs, it is practical to run them in parallel on HPC or VM resources. To support this, a bootstrap script is provided that prepares everything needed to execute the experiments on a KWOK cluster.

First, create a `bootstrap` folder containing the required artifacts, including built binaries, solver code, trace replayer script, job files, and configuration files:

```bash
./make_bootstrap_folder.sh
```

The generated bootstrap folder can then be uploaded to the HPC or VM environment. There, the bootstrap.sh script in `scripts/bootstrap/` can be used to set up the job runner and execute the experiments. The script installs the required prerequisites and runs the selected jobs. The bootstrap script accepts the same parameters as `trace_replayer.py`. The most important ones are:

- `--runner`: which runner to use (e.g., `trace_replayer`)
- `--content-dir`: path to the generated `bootstrap` folder
- `--job-file`: path to the job file to run (see `data/jobs/`)

## Unit and Integration Tests

Unit and integration tests are provided for both OPSche and the solver. The tests are located in `scripts/tests/` and can be run with `pytest` (tested with version `9.0.1`).

For convenience, the repository also includes a `run_tests.sh` script in the root directory that runs both Python and Go tests.

Before running, install the required Python dependencies:

```bash
# For integration tests
pip install -r scripts/kwok_integration_tests/requirements.txt
# For unit tests
pip install -r scripts/test/requirements.txt
```

Then run the tests with:

```bash
./run_tests.sh
# or run only unit tests
./run_tests.sh unit
# or run only integration tests
./run_tests.sh int
```

## Upstream Version

The latest upstream versions are listed in the [scheduler-plugins releases](https://github.com/kubernetes-sigs/scheduler-plugins/releases) and should follow [Kubernetes versioning](https://kubernetes.io/releases). After merging with upstream, always verify that everything works as intended, e.g., by running the integration tests (see [Unit and Integration Tests](#unit-and-integration-tests)).
