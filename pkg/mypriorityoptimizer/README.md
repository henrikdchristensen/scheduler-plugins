
# Kubernetes Plugin for Optimized Scheduling

- [Kubernetes Plugin for Optimized Scheduling](#kubernetes-plugin-for-optimized-scheduling)
  - [Overview](#overview)
  - [Code Structure](#code-structure)
  - [Integrating with the scheduler](#integrating-with-the-scheduler)
  - [Building](#building)
  - [Requirements for Execution on a KWOK Cluster](#requirements-for-execution-on-a-kwok-cluster)
  - [Evaluation of the Plugin on a KWOK Cluster](#evaluation-of-the-plugin-on-a-kwok-cluster)
    - [Experimental Setup Used](#experimental-setup-used)
    - [Workload Once Generator](#workload-once-generator)
      - [Structuring Workload Once Generator Results](#structuring-workload-once-generator-results)
    - [Trace Replayer](#trace-replayer)
      - [Structuring Trace Replayer Results](#structuring-trace-replayer-results)
    - [Bootstrapping for parallel evaluation](#bootstrapping-for-parallel-evaluation)
      - [Using Vagrant for bootstrap script development](#using-vagrant-for-bootstrap-script-development)
    - [Analysis of Results](#analysis-of-results)
  - [Unit and Integration Tests](#unit-and-integration-tests)
  - [GitHub Actions](#github-actions)
  - [Useful kubectl/kwokctl commands](#useful-kubectlkwokctl-commands)
  - [Upstream version](#upstream-version)

## Overview

This project introduces an **priority-based** approach for improving (ideally **optimal**) placements of pods onto nodes via the **MyPriorityOptimizer** plugin for the Kubernetes scheduler. The project is a fork of the Kubernetes-sigs project [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) and extends it with a new plugin. For background see description of the [Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/).

<center><img src="./images/scheduling-framework.png" alt="Scheduling Framework" width="60%"/></center>

The goal is to schedule as many *high-priority* pods as possible (with improved resource utilization as a possible side effect). To support this, the plugin integrates an *external* solver to compute an improved—ideally optimal—placement plan that maximizes the number of scheduled high-priority pods while *minimizing disruption* (i.e., reducing reallocations and evictions).

In this work, we provide two solver implementations: a Python-based **CP-SAT** solver using [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver), and a **Gurobi**-based solver [Gurobi](https://www.gurobi.com/). Given a solver-produced plan, the plugin applies it by evicting and relocating pods, potentially across multiple nodes. This is an important difference from the default Kubernetes scheduler, whose built-in preemption is limited to a *single node*, which can lead to more preemptions than necessary.

The plugin can invoke optimization using three different **trigger modes**:

- *SchedulingFailure* – for every failed scheduling attempt by the default scheduler (not recommended for large clusters).
- *Periodic* – optimize pods (running and pending) at fixed intervals.
- *StableQueue* – optimize pods during stable queue windows (i.e. when no pods are arriving).
<!-- - *Manual* – runs normal scheduling like the ones above, but optimization is only triggered manually (via HTTP). Used for testing and evaluation.
- *ManualBlocking* – same as *Manual*, but all pods are blocked from entering the cluster until the solver is triggered via HTTP and completes. Used for testing and evaluation. -->

Moreover, the solver can run in either **blocking** or **non-blocking** mode:

- *Blocking* – regular scheduling of new pods is blocked while the solver is  running and while the plan is being applied.
- *Non-Blocking* – regular scheduling of new pods is not blocked while the solver is running, only while the plan is being applied.

In both cases, when the solver completes, the cluster state is re-checked to ensure the plan is still valid before applying it; otherwise, the plan is discarded.

The following sections describe how to **build, run, and test** the scheduler with the plugin.

For [result replication](#result-replication-and-running-test-jobs) from the paper, read the provided instructions.

## Code Structure

The source code for the **MyPriorityOptimizer** plugin is located under `pkg/mypriorityoptimizer/`.

Some of the main files and their purpose are described below:

- `plugin.go`: Main entry point for the plugin that sets up the plugin.
- `args.go` and `constants.go`: Contains the configuration arguments for the plugin (e.g. trigger mode, solver timeout, etc.).
- `optimization_flow.go`: Contains the main logic for running the optimization flow, including triggering the solver, applying the plan, etc.
- `loop_helpers.go`, `loop_periodic.go`, `loop_stable_queue.go`: Contains the logic for triggering the optimization flow based either fixed intervals (periodic) or during stable queue windows (stable_queue).
- `solver_external.go` and `solver_python.go`: Contains the logic for invoking the external solver and parsing its output.
- `hook_preenqueue.go`: Implements the PreEnqueue hook. This is mainly used for blocking new pods while an optimization is running or a plan is being applied.
- `hook_prefilter.go`: Implements the PreFilter hook. Its main purpose is targeting the pod onto the node assigned by the solver in the plan (if any).
- `hook_postfilter.go`: Implements the PostFilter hook. This is mainly used to mark a pod as unschedulable as the default scheduler failed to place it. If in mode *SchedulingFailure*, it also triggers the optimization for every new pod that arrives.
- `hook_reserve_unreserve.go`: Implements the Reserve/Unreserve hooks. It is used as we cannot rely on specific pod names (e.g. pods from a ReplicaSets), instead we count how many of each that should be placed on each node.
- `plan_completion_watch.go`: A background watcher for plan completion, checking that all pods in the plan are assigned to the correct nodes.

Finally, the solver code is located under `scripts/python_solver/` and can be used as a template for implementing other solvers. Two solvers are provided, one using the CP-SAT constraint programming solver and one using the Gurobi mixed integer programming solver, both using the same priority-based optimization approach.

## Integrating with the scheduler

To enable and use plugins in the Kubernetes scheduler, you must apply a **scheduler configuration manifest** that selects the plugins and their settings; the manifests used for this plugin is located in `manifests/mypriorityoptimizer/` (see also [Scheduler Configuration](https://kubernetes.io/docs/reference/scheduling/config/) for background).

This file enables the **MyPriorityOptimizer** plugin and the hooks, described above, that it implements.

The plugin is referenced and registered in `cmd/scheduler/main.go` such that the scheduler will include it at build time.

## Building

The scheduler and the plugin can be built as a **binary** and can then be run in a cluster (e.g. using KWOK).

The following tools are required (if Windows host, use WSL2 w/ e.g. Ubuntu) to build the scheduler+plugin:

- `git` (tested with 2.43.0)
- `make` (tested with 4.3)
- `python3` (tested with 3.10.12)
- `pip` (tested with 24.0)
- `Go` (tested with 1.24.3)

Currently, it is only tested on **amd64** architecture and some code may need to be modified to run on other architectures (should not be a problem).

To build the binary, run the following bash script in the root of the repo:

```bash
./build.sh
```

The built binary will be located in `bin/kube-scheduler`. We also download the `kube-apiserver`, `kube-controller-manager` binaries to `bin/`, as they are needed when running in KWOK clusters and to prevent them from downloading on each cluster creation we store them in the `bin/` folder.

## Requirements for Execution on a KWOK Cluster

To run the scheduler with the plugin on a **KWOK** cluster.

The following tools are required (tools already mentioned in [Building](#building) are omitted):

- `kubectl` (tested with client v.1.32.7)
- `kwok`+`kwokctl` (tested with v0.7.0)

To set up a KWOK cluster with the scheduler+plugin one needs to provide a configuration file to KWOK. An example of a configuration file is `data/configs-kwokctl/plugin-scheduler-defpreempt=1.yaml`.

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
        value: "periodic" # choices: scheduling_failure, periodic, stable_queue, manual, manual_blocking
      - name: OPTIMIZE_BLOCKING_SOLVING
        value: "true" # choices: true, false
      - name: OPTIMIZE_PERIODIC_INTERVAL
        value: 30s # e.g. 10s, 30s, 60s
      - name: SOLVER_PYTHON_ENABLED
        value: "true"
      - name: SOLVER_PYTHON_TIMEOUT
        value: 10s # e.g. 1s, 10s, 20s
```

**Note:** There may be version mismatches between `scheduler-plugins` and KWOK. If so, set a specific `VERSION` value when building the scheduler binary (sometimes this must be forced for a given `scheduler-plugins` release). For example, KWOK may report …

```bash
# Running the command
kwokctl logs kube-scheduler --name kwok1
# may report
E1210 13:54:11.815001   22220 run.go:72] "command failed" err="[emulation version 0.33 is not between [1.31, 0.33.0], minCompatibilityVersion version 0.32 is not between [1.31, 0.33]]"
```

## Evaluation of the Plugin on a KWOK Cluster

Two different approaches for evaluating have been made:

1) **Workload Once Generator** (solver evaluation): A script that generates an initial workload for the scheduler and the plugin to place and optimize, and then evaluates the resulting placement.
2) **Trace Replayer** (plugin evaluation): A script that simulates workload arrivals and removals in a cluster and continuously evaluates the scheduler and the plugin including different optimization modes.

### Experimental Setup Used

The evaluation is done using machines having the following specifications:

- TODO: HARDWARE
- Ubuntu v22.04

### Workload Once Generator

For the **Workload Once Generator**, we first run the default scheduler (deterministically) to find 100 seeds where not all pods are running using another plugin called **MyDeterministicScore**, located under `pkg/mydeterministicscore/`. This plugin breaks scoring ties by name, disables `DefaultPreemption` plugin and sets `parallelism=1`.

The scheduler configuration file for using this plugin is `data/configs-kwokctl/default-deterministic.yaml`. Hereafter, we run the default scheduler (as-is) using the configuration file `data/configs-kwokctl/default.yaml`, and the scheduler with the `MyPriorityOptimizer` plugin on these seeds.

The evaluation script for generating a initial workload and running the tests is `scripts/kwok_workload_once/test_runner.py`. The script can be setup by reading settings from three types of sources, with later ones overriding earlier ones:

1) a workload configuration file (e.g. `data/configs-workload/base.yaml`), containing the general configuration for the workload to generate;
2) a test job file (e.g. `data/jobs/kwok_workload_once/<job_file>.yaml`), containing the specific configuration for a job; and
3) command-line arguments to the script (highest priority)

After choosing one of more of these source, the script can be run to generate the workload and run the tests. An example of how to run the script from the root of the repo is:

```bash
python -m scripts.kwok_workload_once.test_runner \
--cluster-name my-cluster \
--kwok-runtime binary \
--job-file data/jobs/kwok_workload_once/<job_file>.yaml \
--workload-config-file data/configs-workload/<workload_config_file>.yaml \
--kwokctl-config-file data/configs-kwokctl/<kwokctl_config_file>.yaml
```

TODO: SKAL VÆRE klart at vi specificerer seeds fra deterministic plugin here, og at job files specify which seeds to use.

#### Structuring Workload Once Generator Results

Having downloaded the results folder containing results from all jobs - the job files ensures that the results is organized by job type, as follows:

```text
analysis/kwok_workload_once/
├── default/
│   ├── nodes4_pods16_prio1_util090/
│   │   ├── results.csv
│   │   ├── info.yaml
│   │   ├── seeds-all-running.txt        (if applicable)
│   │   └── seeds-not-all-running.txt    (if applicable)
│   ├── ...
│   └── nodes32_pods256_prio4_util105/
└── plugin-cp_sat/
    ├── nodes4_pods16_prio1_util090_timeout01/
    │   ├── results.csv
    │   ├── info.yaml
    │   ├── seeds-all-running.txt        (if applicable)
    │   ├── seeds-not-all-running.txt    (if applicable)
    │   ├── scheduler-logs/
    │   └── solver-stats/
    ├── nodes4_pods16_prio1_util090_timeout10/
    ├── nodes4_pods16_prio1_util090_timeout20/
    ├── nodes4_pods16_prio1_util095_timeout01/
    ├── nodes4_pods16_prio1_util095_timeout10/
    ├── nodes4_pods16_prio1_util095_timeout20/
    ├── nodes4_pods16_prio1_util100_timeout01/
    ├── nodes4_pods16_prio1_util100_timeout10/
    ├── nodes4_pods16_prio1_util100_timeout20/
    ├── nodes4_pods16_prio1_util105_timeout01/
    ├── nodes4_pods16_prio1_util105_timeout10/
    ├── nodes4_pods16_prio1_util105_timeout20/
    ├── ...
    └── nodes32_pods256_prio4_util105_timeout20/
```

The `results.csv` file contains the scheduling results for the job, while the `info.yaml` file contains the job configuration used. If applicable, the `seeds-all-running.txt` and `seeds-not-all-running.txt` files contain the seeds where all pods were running and where not all pods were running, respectively. For the plugin jobs, the `scheduler-logs/` folder contains the saved kube-scheduler logs for each seed, while the `solver-stats/` folder contains the saved solver statistics for each seed.

### Trace Replayer

For the **Trace Replayer**, we first generate traces, and then replay them in both the default scheduler and the scheduler with the plugin.

The generated traces are stored under `data/traces/` and can be generated using the `scripts/kwok_trace_replayer/trace_generator.py` script. Using job files specifying  number of nodes, inter-arrival times, and other parameters, located under `data/jobs/kwok_trace_generator/`, the script generates traces of workload arrivals and removals in a cluster. Using the job files, the script generate them using: 

```bash
python -m scripts.kwok_trace_replayer.trace_generator \
--job-dir data/jobs/kwok_trace_generator/
```

Having generated the traces, they can be replayed using the `scripts/kwok_trace_replayer/trace_replayer.py` script. Using job files specifying which traces to replay, and how the plugin should be configured, located under `data/jobs/kwok_trace_replayer/`, the script replays the traces in a cluster. To run the trace replayer, run:

```bash
python -m scripts.kwok_trace_replayer.trace_replayer \
--job-file data/jobs/kwok_trace_replayer/<job_file>.yaml \
```

#### Structuring Trace Replayer Results

Having downloaded the results folder containing results from all jobs - the job files ensures that the results is organized by job type, as follows:

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

### Bootstrapping for parallel evaluation

As there is many jobs to run, it is beneficial to parallelize the evaluation using HPC resources.
To create a bootstrap folder used for a job worker containing all content needed to run tests on a KWOK cluster, including built binaries, solver code, and test jobs and configuration files. From the root of the repo, run:

```bash
./make_bootstrap_folder.sh
```

The idea is then to upload the created `bootstrap` folder to the HPC/VM provider, and then run the tests using a `bootstrap.sh` script, located under `scripts/bootstrap/` that sets up a job runner and runs the tests. The script will ensure all prerequisites are installed and the tests are run.

The bootstrap script accepts parameters that both the `test_runner.py` and `trace_replayer.py` scripts accepts, but the two main parameters to provide are:

- `--runner`: the runner to use, either `test_runner` or `trace_replayer`.
- `--content-dir`: path to the `bootstrap` folder created using the `make_bootstrap_folder.sh` script.
- `--job-file`: path to the job file to run (e.g. see jobs under `data/jobs/`).

#### Using Vagrant for bootstrap script development

To develop and test the bootstrap script it can be beneficial to run it in a VM on a local machine. For that reason, a `Vagrantfile` is provided in the root of the repo. Ensure the `bootstrap` folder is created first by running the `make_bootstrap_folder.sh` script.
Using Vagrant, it will create an Ubuntu 22.04 VM with all prerequisites installed. To use it, install `Vagrant` (tested with v2.4.7) and `VirtualBox` (tested with v7.1.10), then run:

```bash
vagrant up
```

This will create a VM named `scheduler-plugins` that you can SSH into using:

```bash
vagrant ssh
```

To delete the VM, run:

```bash
vagrant destroy -f
```

### Analysis of Results

Analyzed results are placed under the `analysis` folder. They assume the layout shown in  [Expected folder structure after running all tests for Workload Once Generator](#expected-folder-structure-after-running-all-tests-for-workload-once-generator) and [Expected folder structure after running all tests for Trace Replayer](#expected-folder-structure-after-running-all-tests-for-trace-replayer) if yours differs, code changes may be needed.

For the **Workload Once Generator**, the analysis code is located under `scripts/kwok_workload_once/` and for the **Trace Replayer**, the analysis code is located under `scripts/kwok_trace_replayer/`. To run the analysis, follow these steps.

For the both there is a `seal_results.py` script that merge all results into CSV file(s), and a `plots_and_tables.py` script that produces all the figures and tables used in the report.

1. First, merge the output files using `python scripts/<workload_once_or_trace_replayer>/seal_results.py`.
2. Then run `python scripts/<workload_once_or_trace_replayer>/plots_and_tables.py` to produce every figure and table used in the paper.

## Unit and Integration Tests

To make running tests easier, a `run_tests.sh` script has been provided in the root of the repo that can be used to run all tests (both Python and Go tests), simply run:

```bash
./run_tests.sh
# or to run only unit tests, run:
./run_tests.sh unit
# or to run only integration tests, run:
./run_tests.sh int
```

## GitHub Actions

A GitHub Actions workflow is provided in `.github/workflows/opt-prio-ci.yml` that runs on every push and pull request to the default branch `opt-prio-main` branch. The workflow runs the unit and integration tests.

## Useful kubectl/kwokctl commands

- Get pods

  ```bash
  # get all pods
  kubectl get pods -A -n <namespace>
  # get pending pods
  kubectl get pods -A -n <namespace> --field-selector=status.phase=Pending
  # get full description of a pod
  kubectl describe pod <pod_name> -n <namespace>
  ```

- Get nodes

  ```bash
  # get all nodes
  kubectl get nodes
  # get capacity and allocatable for a node
  kubectl get node <node> -o jsonpath='{.status.capacity}{"\n"}{.status.allocatable}{"\n"}'
  ```

- Delete namespace

  ```bash
  kubectl delete ns <namespace>
  ```

- Show recent cluster events

  ```bash
  kubectl get events -A --sort-by=.lastTimestamp
  ```

- Get kube-scheduler logs from KWOK cluster

  ```bash
  kwokctl logs kube-scheduler --name <cluster_name>
  ```

- Getting plugin configuration from kube-scheduler

  ```bash
  kubectl -n kube-system get cm kube-scheduler -o jsonpath='{.data.config\.yaml}' | yq e .
  ```

- Getting saved solver plans from kube-scheduler

  ```bash
  kubectl -n kube-system get cm -l plan
  kubectl -n kube-system get cm <CM> -o jsonpath='{.data.plan\.json}' | jq .
  ```

- KWOK: list clusters / create / delete

  ```bash
  # get all clusters
  kwokctl get clusters
  # create / delete cluster
  kwokctl create cluster --name <cluster_name>
  kwokctl delete cluster --name <cluster_name>
  ```

- KWOK: Get reasons for pod(s) not scheduling

  ```bash
  kubectl --context <ctx> -n <namespace> get events --field-selector involvedObject.kind=Pod -o json | jq '.items[] | {name: .involvedObject.name, reason: .reason, message: .message}'
  ```

## Upstream version

Latest upstream version are listed in the [Releases](https://github.com/kubernetes-sigs/scheduler-plugins/releases), it should follow the [Kubernetes versioning](https://kubernetes.io/releases).
After merging with upstream, always verify that all works as intended, e.g. by running a small smoke test with the `test_runner.py` script (see below).
