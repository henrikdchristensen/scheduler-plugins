
# Kubernetes Plugin for Optimized Scheduling

- [Kubernetes Plugin for Optimized Scheduling](#kubernetes-plugin-for-optimized-scheduling)
  - [Overview](#overview)
  - [Code Structure](#code-structure)
  - [Scheduler Integration](#scheduler-integration)
  - [Building the Scheduler and OptPlugin](#building-the-scheduler-and-optplugin)
  - [Requirements for Running on a KWOK Cluster](#requirements-for-running-on-a-kwok-cluster)
  - [Evaluation of OptPlugin on a KWOK Cluster](#evaluation-of-optplugin-on-a-kwok-cluster)
    - [Experimental Setup Used for This Analysis](#experimental-setup-used-for-this-analysis)
      - [Scheduler setups](#scheduler-setups)
    - [Workload Once Generator](#workload-once-generator)
      - [Running the generator](#running-the-generator)
      - [Gathering instances for evaluation](#gathering-instances-for-evaluation)
      - [Organizing workload once results](#organizing-workload-once-results)
    - [Trace Replayer](#trace-replayer)
      - [Generating traces](#generating-traces)
      - [Replaying traces](#replaying-traces)
      - [Organizing trace replay results](#organizing-trace-replay-results)
    - [Faster Evaluation Through Parallelization](#faster-evaluation-through-parallelization)
    - [Analysis of Results](#analysis-of-results)
  - [Unit and Integration Tests](#unit-and-integration-tests)
  - [Upstream Version](#upstream-version)

## Overview

This project is a fork of the Kubernetes-sigs project [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) (see [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/) for background).

It introduces **OptPlugin** (denoted **MyPriorityOptimizer** in the code), a Kubernetes scheduler plugin that improves pod-node placements by delegating placement planning to an external **solver**, aiming for **optimal placements** when feasible. Given a solver-produced plan, OptPlugin enforces it by *evicting* and *relocating* pods, potentially across multiple nodes (**cross-node preemption**). This differs from the default Kubernetes scheduler, whose built-in preemption (*DefaultPreemption*) is limited to a *single node* and can therefore cause more preemptions than necessary and lead to *suboptimal* placements.

Specifically, OptPlugin implements the following extension points (also called **hooks**):

- **PreEnqueue** – temporarily blocks new pods from entering the scheduling queue while a placement plan is being applied.
- **PreFilter** – steers a pod to the node selected by the solver.
- **PostFilter** – triggers optimization after a scheduling failure (i.e., when the default scheduler cannot place a pod).
- **Reserve/Unreserve** – reserves and releases node resources according to the solver plan.

Beyond these hooks, OptPlugin also includes a **background loop** that can trigger optimization in two additional ways: periodically at fixed time intervals, or during stable-queue windows (i.e., when no new pods are arriving). Together, this provides three **trigger modes**, all of which execute the same optimization flow.

- **SchedulingFailure** – optimize after each failed scheduling attempt (generally not recommended for large clusters).
- **Periodic** – optimize running and pending pods at fixed time intervals.
- **StableQueue** – optimize during stable-queue windows (i.e., when no new pods are arriving for a certain time).

Moreover, the solver can run in either **blocking** or **non-blocking** mode:

- **Blocking** – regular scheduling of new pods is paused while the solver runs and while the resulting plan is enforced.
- **Non-blocking** – regular scheduling continues while the solver runs and is paused only during plan enforcement.

In both modes, once the solver finishes, OptPlugin re-checks the cluster state before enforcing the plan. If the state has changed and the plan is no longer valid, it is discarded.

To detect **plan completion** and verify that pods end up on the intended nodes, OptPlugin also includes a **background watcher** that tracks plan completion and checks for any discrepancies between the intended and actual placements.

<!-- TODO: uncomment when Gurobi solver is included -->
<!-- The project provides two solver implementations, both based on the same priority-aware optimization model. The objective is to schedule as many *high-priority* pods as possible while *minimizing disruption* (i.e., reducing reallocations and evictions): -->
<!-- - a **CP-SAT** solver implemented with the Python API of [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver), and -->
<!-- - a **Mixed-Integer Programming (MIP)** solver implemented with the Python API of [Gurobi](https://docs.gurobi.com/current/). -->

The solver implementation uses a **priority-aware** optimization model. The objective is to schedule as many *high-priority* pods as possible while *minimizing disruption* (i.e., reducing reallocations and evictions). It is implemented as a **CP-SAT** solver using the Python API of [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver).

## Code Structure

The source code for OptPlugin is located in `pkg/mypriorityoptimizer/`. The main files and their roles are:

- `plugin.go` – main OptPlugin entry point and setup.
- `args.go` and `constants.go` – OptPlugin configuration arguments and constants (e.g., trigger mode, solver timeout).
- `optimization_flow.go` – core *optimization flow*, including solver invocation and plan application.
- `loop_helpers.go`, `loop_periodic.go`, `loop_stable_queue.go` – *background-loop* logic for *Periodic* and *StableQueue* triggering.
- `solver_external.go` and `solver_python.go` – external solver invocation and parsing of solver output.
- `hook_preenqueue.go` – implementation of the *PreEnqueue* hook, mainly used to block new pods while optimization is running or a plan is being enforced.
- `hook_prefilter.go` – implementation of the *PreFilter* hook, used to steer a pod to the node assigned by the solver plan.
- `hook_postfilter.go` – implementation of the *PostFilter* hook, used to detect failed scheduling attempts; in *SchedulingFailure* mode, it also triggers optimization.
- `hook_reserve_unreserve.go` – implementation of the *Reserve/Unreserve* hooks, used to reserve/release resources according to the plan.
- `plan_completion_watch.go` – background watcher that tracks plan completion and verifies that pods end up on the intended nodes.

The solver implementation is located in `scripts/python_solver/` and can also serve as a template for adding other solvers.

<!-- TODO: uncomment when Gurobi solver is included -->
<!-- The two solver implementations are located in `scripts/python_solver/` and can also serve as templates for adding other solvers. -->
<!-- **Note:** Since Gurobi is a commercial solver, valid license credentials must be provided in the script—specifically `GRB_WLSACCESSID`, `GRB_WLSSECRET`, and `GRB_LICENSEID`—which can be obtained from the Gurobi license file. In practice, the license setup allowed at most two parallel executions using the WLS (Web License Server) access. -->

## Scheduler Integration

Plugins in the Kubernetes scheduler are enabled through a **scheduler configuration manifest** that selects the plugin and its settings. The manifests for OptPlugin are located in `manifests/mypriorityoptimizer/` (see also [Scheduler Configuration](https://kubernetes.io/docs/reference/scheduling/config/) for background).

OptPlugin is registered in `cmd/scheduler/main.go`, which ensures it is included in the scheduler binary at build time.

## Building the Scheduler and OptPlugin

The scheduler and OptPlugin can be built into a **binary** that can then be run in a cluster (e.g., with [KWOK](https://kwok.sigs.k8s.io/)). The following tools are required to build the scheduler and OptPlugin:

- `git` (tested with 2.43.0)
- `make` (tested with 4.3)
- `python3` (tested with 3.10.12)
- `pip` (tested with 24.0)
- `Go` (tested with 1.24.3)

Currently, the project has been tested on **amd64** and **arm64**. Running on other architectures should not be a problem but has not been verified.

To **build** the binary, run the following script from the repo root:

```bash
./build.sh
```

**Note:** You may get prompted to install some required Go dependencies, which can be done with `go get`.

The resulting scheduler binary is written to `bin/kube-scheduler`. The build process also downloads `kube-apiserver` and `kube-controller-manager` into `bin/`, since they are required for running KWOK clusters; keeping them there avoids re-downloading them for each cluster creation.

## Requirements for Running on a KWOK Cluster

To run the scheduler with OptPlugin on a **KWOK** cluster, a few additional tools are required (tools already listed in [Building the Scheduler and OptPlugin](#building-the-scheduler-and-optplugin) are omitted):

- `kubectl` (tested with client v1.32.7)
- `kwok` and `kwokctl` (tested with v0.7.0)

Running the scheduler and OptPlugin on KWOK also requires a **KWOK cluster configuration file**. An example is `data/configs-kwokctl/plugin-scheduler-defpreempt=1.yaml`:

```yaml
kind: KwokctlConfiguration
apiVersion: config.kwok.x-k8s.io/v1alpha1
options:
  kubeSchedulerConfig: manifests/mypriorityoptimizer/plugin-scheduler-config.yaml
  kubeSchedulerBinary: bin/kube-scheduler
  kubeSchedulerImage: localhost:5000/scheduler-plugins/kube-scheduler:dev
componentsPatches:
  - name: kube-scheduler
    # uncomment for more verbose logging
    # extraArgs:
    #   - key: v
    #     value: "10"
    extraEnvs:
      - name: OPTIMIZE_MODE
        value: "periodic" # choices: scheduling_failure, periodic, stable_queue
      - name: OPTIMIZE_BLOCKING_SOLVING # whether to block scheduling while the solver is running
        value: "false"
      - name: OPTIMIZE_PERIODIC_INTERVAL # interval for 'Periodic' mode
        value: 8s
      - name: OPTIMIZE_STABLE_QUEUE_DELAY # stable-queue window for 'StableQueue' mode
        value: 8s
      - name: SOLVER_PYTHON_ENABLED
        value: "true"
      - name: SOLVER_PYTHON_TIMEOUT # timeout for the solver before it is killed and the plan is discarded
        value: 10s
```

**Note:** There may be version mismatches between `scheduler-plugins` and KWOK. If so, set a specific `VERSION` value when building the scheduler binary (sometimes this must be forced for a given `scheduler-plugins` release). For example, KWOK may report:

```bash
# Running the command
kwokctl logs kube-scheduler --name kwok1
# may report
E1210 13:54:11.815001   22220 run.go:72] "command failed" err="[emulation version 0.33 is not between [1.31, 0.33.0], minCompatibilityVersion version 0.32 is not between [1.31, 0.33]]"
```

## Evaluation of OptPlugin on a KWOK Cluster

Two evaluation approaches are provided:

1. **Workload-Once Generator** (*solver evaluation*) – generates an initial workload, invokes the solver once, and evaluates the quality of the resulting placement plan.
2. **Trace Replayer** (*OptPlugin evaluation*) – simulates workload arrivals and removals over time in a cluster and continuously evaluates the scheduler with OptPlugin.

### Experimental Setup Used for This Analysis

The evaluations in this analysis were executed on machines with the following specifications:

- **CPU:** 8 vCPUs (Intel Xeon Gold 6130)
- **Memory:** 48 GB RAM
- **Operating system:** Ubuntu 24.04
- **Optimizer library:** OR-Tools 9.14.6206
<!-- TODO: uncomment when Gurobi solver is included -->
<!-- - **Optimizer libraries:** OR-Tools 9.14.6206 and GurobiPy 13.0.1 -->

#### Scheduler setups

To compare the default scheduler against the OptPlugin approach, two cluster configurations are used:

1. **Default scheduler** – the default Kubernetes scheduler without OptPlugin, configured with `data/configs-kwokctl/default.yaml`.
2. **Scheduler with OptPlugin** – the scheduler with OptPlugin enabled and configured with `data/configs-kwokctl/plugin-scheduler-defpreempt=<1|0>.yaml` (1=default preemption enabled, 0=default preemption disabled).

### Workload Once Generator

The **Workload Once Generator** creates workload instances and evaluates them with both the default scheduler and the scheduler with OptPlugin.

For the lower utilization levels (90% and 95%), a deterministic seed pre-filtering step is first applied to exclude trivial cases where all pods are scheduled. This step uses `data/configs-kwokctl/default-deterministic.yaml`, which enables `MyDeterministicScore` (`pkg/mydeterministicscore/`) to break scoring ties by pod name, disables `DefaultPreemption`, and sets `parallelism=1` to make scheduling fully deterministic.

#### Running the generator

The evaluation script is `scripts/kwok_workload_once/test_runner.py`. It can read configuration from three sources (later overrides earlier):

1. workload config file (e.g., `data/configs-workload/base.yaml`)
2. job file (e.g., `data/jobs/kwok_workload_once/<job_file>.yaml`)
3. command-line arguments

First, install the required Python dependencies:

```bash
pip install -r scripts/kwok_workload_once/requirements.txt
```

Then run the generator:

```bash
python -m scripts.kwok_workload_once.test_runner \
  --job-file data/jobs/kwok_workload_once/<job_file>.yaml
```

#### Gathering instances for evaluation

For **90% and 95% utilization**, seed selection is performed in two steps to filter out instances where the default scheduler places all pods without any contention. At **100% and 105% utilization**, the cluster is most likely (always) oversubscribed, so all seeds from `data/seeds/kwok_workload_once/seeds_100.txt` are used directly.

1. **Deterministic pre-filtering**
   Run the deterministic default scheduler with:

   - `seed-file: data/seeds/kwok_workload_once/seeds_all.txt`

   This produces `seeds-not-all-running.txt` containing up to 100 seeds where not all pods are running.

2. **Evaluation on selected seeds**
   These seeds are saved as configuration-specific files (e.g.,
   `nodes4_pods16_prio1_util095.txt`) under `data/seeds/kwok_workload_once/` and used for both the default scheduler and OptPlugin runs.

#### Organizing workload once results

Once all jobs have completed, organize the results as follows for later analysis.

```text
analysis/kwok_workload_once/
├── default/
│   ├── nodes4_pods16_prio1_util090/
│   │   ├── results.csv
│   │   └── info.yaml
│   ├── ...
│   └── nodes32_pods256_prio4_util105/
└── plugin-cp_sat/
    ├── nodes4_pods16_prio1_util090_timeout01/
    │   ├── results.csv
    │   └── info.yaml
    ├── nodes4_pods16_prio1_util090_timeout10/
    ├── nodes4_pods16_prio1_util090_timeout20/
    ├── ...
    └── nodes32_pods256_prio4_util105_timeout20/
```

### Trace Replayer

The **Trace Replayer** workflow consists of two steps: **trace generation** and **trace replay**. Traces are first generated, then replayed with both the default scheduler and the scheduler with OptPlugin.

#### Generating traces

Traces are generated with `scripts/kwok_trace_replayer/trace_generator.py` and stored under `data/traces/`.

The generator reads job files from `data/jobs/kwok_trace_generator/`, where each job file defines parameters such as the number of nodes, inter-arrival times, and other workload settings. From these job files, the script produces traces of workload arrivals and removals over time.

First, install the required Python dependencies:

```bash
pip install -r scripts/kwok_trace_replayer/requirements.txt
```

Then generate traces from the job files:

```bash
python -m scripts.kwok_trace_replayer.trace_generator \
--job-dir data/jobs/kwok_trace_generator/
```

#### Replaying traces

After traces have been generated, they can be replayed with `scripts/kwok_trace_replayer/trace_replayer.py`. The replayer uses job files in `data/jobs/kwok_trace_replayer/`, which specify the trace to replay and how OptPlugin should be configured for that run.

To replay a trace, run:

```bash
python -m scripts.kwok_trace_replayer.trace_replayer \
--job-file data/jobs/kwok_trace_replayer/<job_file>.yaml
```

#### Organizing trace replay results

After all jobs have completed, organize the results as follows for later analysis.

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

### Faster Evaluation Through Parallelization

Because the evaluation includes many jobs, it is useful to run them in parallel on HPC or VM resources.

To prepare this, first create a `bootstrap` folder containing everything needed to run experiments on a KWOK cluster (built binaries, solver code, job files, and configuration files). From the repo root, run:

```bash
./make_bootstrap_folder.sh
```

The generated `bootstrap` folder can then be uploaded to the HPC/VM provider. There, use the `bootstrap.sh` script (located in `scripts/bootstrap/`) to set up a job runner and execute the experiments. The script installs required prerequisites and runs the selected jobs.

The bootstrap script supports the same parameters as `test_runner.py` and `trace_replayer.py`, but the main ones are:

- `--runner`: which runner to use (`test_runner` or `trace_replayer`)
- `--content-dir`: path to the generated `bootstrap` folder
- `--job-file`: path to the job file to run (see `data/jobs/`)

### Analysis of Results

Analyzed outputs are stored in the `analysis/` folder. The analysis scripts assume the directory layouts described in [Organizing workload once results](#organizing-workload-once-results) and [Organizing trace replay results](#organizing-trace-replay-results).

Analysis code is split by evaluation type:

- **Workload Once Generator**: `scripts/kwok_workload_once/`
- **Trace Replayer**: `scripts/kwok_trace_replayer/`

For both, the workflow is the same:

- `seal_results.py` merges raw outputs into one CSV file
- `plots_and_tables.py` generates the figures and tables used in the paper

To run the analysis:

1. Seal results:
   `python scripts/<workload_once_or_trace_replayer>/seal_results.py`
2. Generate figures and tables:
   `python scripts/<workload_once_or_trace_replayer>/plots_and_tables.py`

## Unit and Integration Tests

Unit and integration tests are provided for both OptPlugin and the solver. The tests are located in `scripts/tests/` and can be run with `pytest` (tested with version `9.0.1`).

<!-- TODO: uncomment when Gurobi solver is included -->
<!-- Unit and integration tests are provided for both OptPlugin and the solvers. -->

For convenience, the repository also includes a `run_tests.sh` script in the root directory that runs both Python and Go tests:

```bash
./run_tests.sh
# or run only unit tests
./run_tests.sh unit
# or run only integration tests
./run_tests.sh int
```

Before running, install the required Python dependencies:

```bash
# For integration tests
pip install -r scripts/kwok_integration_tests/requirements.txt
# For unit tests
pip install -r scripts/test/requirements.txt
```

## Upstream Version

The latest upstream versions are listed in the [scheduler-plugins releases](https://github.com/kubernetes-sigs/scheduler-plugins/releases) and should follow [Kubernetes versioning](https://kubernetes.io/releases). After merging with upstream, always verify that everything works as intended, e.g., by running the integration tests (see [Unit and Integration Tests](#unit-and-integration-tests)).
