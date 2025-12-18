
# Priority Optimizer Plugin

- [Priority Optimizer Plugin](#priority-optimizer-plugin)
  - [Overview](#overview)
  - [Building](#building)
  - [Running](#running)
    - [KWOK (recommended)](#kwok-recommended)
  - [Testing](#testing)
    - [Workload Once Generator](#workload-once-generator)
      - [Deterministic scheduling](#deterministic-scheduling)
      - [Workload configuration file](#workload-configuration-file)
      - [Job file](#job-file)
      - [Bootstrap script](#bootstrap-script)
        - [Deterministic jobs with default scheduler](#deterministic-jobs-with-default-scheduler)
        - [Default scheduler jobs](#default-scheduler-jobs)
        - [Python solver jobs](#python-solver-jobs)
      - [Result replication and running test jobs](#result-replication-and-running-test-jobs)
      - [Expected folder structure after running all jobs](#expected-folder-structure-after-running-all-jobs)
      - [Analysis](#analysis)

## Overview

This project introduces an **optimized, priority-based** approach for placing pods optimally onto nodes via the **MyPriorityOptimizer** plugin for the Kubernetes scheduler. The project is a fork of the Kubernetes-sigs project [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) and extends it with a new plugin.

Our goal is to schedule as many *high-priority* pods as possible, especially when the *default scheduler* fails to do so (a possibly side effect of our approach is better resource utilization). The plugin enables integration of an *external* solver—here, a Python solver using [Google's CP-SAT](https://developers.google.com/optimization/cp/cp_solver)—to compute an optimal placement plan that maximizes high-priority pods while *minimizing disruption* by reducing the number of preemptions (*movements* and *evictions*). Given the solver's solution, the plugin applies it by evicting and moving pods, possibly across multiple nodes in the cluster (Note: default scheduler can only preempt pods within a *single* node, which may lead to more preemptions than necessary).

The following sections describe how to **build, run, and test** the scheduler with the plugin.
For [result replication](#result-replication-and-running-test-jobs) from the paper, read the provided instructions.

The code for the **MyPriorityOptimizer** plugin is located under `pkg/mypriorityoptimizer/`.

## Building

The scheduler+plugin can be built as a **binary** and can then be run in a cluster (e.g. using KWOK).

The following tools are required (if Windows host, use WSL2 w/ e.g. Ubuntu) to build the scheduler+plugin:

- `git` (tested with 2.43.0)
- `make` (tested with 4.3)
- `python3` (tested with 3.10.12)
- `pip` (tested with 24.0)
- `Go` (tested with 1.24.3)

Currently, it is only tested on **amd64** architecture and some code may need to be modified to run on other architectures (should not be a problem).

To build the binary, run the following command in the root of the repo:

```bash
make build-scheduler GO_BUILD_ENV='CGO_ENABLED=0 GOOS=linux GOARCH=amd64'
```

The built binary will be located in `bin/kube-scheduler`.

## Running

To run the scheduler with the plugin, you can e.g. run it in a **KWOK** (recommended) cluster.

The following tools are required (tools already mentioned in [Building](#building) are omitted):

- `kubectl` (tested with client v.1.32.7)
- When running in a KWOK cluster:
  - `kwok`+`kwokctl` (tested with v0.7.0)

### KWOK (recommended)

To set up a KWOK cluster with the scheduler+plugin one needs to provide a configuration file to KWOK. An example of a configuration file is `data/configs-kwokctl/plugin-scheduler.yaml`.

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
        value: "periodic" # choices: per_pod, periodic, interlude, manual, manual_blocking
      - name: OPTIMIZE_SOLVE_SYNCH
        value: "true" # choices: true, false
      - name: OPTIMIZE_PERIODIC_INTERVAL
        value: 30s # e.g. 10s, 30s, 60s
      - name: SOLVER_PYTHON_ENABLED
        value: "true"
      - name: SOLVER_PYTHON_TIMEOUT
        value: 10s # e.g. 1s, 10s, 20s
```

As can be seen this file can be used to set environment variables to configure the plugin.

Having set up the KWOK cluster configuration file, ensure you have built the latest scheduler binary or Docker image (see [Building](#building)). Also if using the binary setup, ensure the the latest Python solver is available at `/opt/solver/main.py` and that the Python environment is set up, by following these steps:

- From the root of the repo, run the following commands to set up the Python environment with the required dependencies:

   ```bash
   sudo install -d -m 0755 /opt/venv/
   sudo python3 -m venv /opt/venv/
   sudo /opt/venv/bin/python -m pip install --upgrade pip
   sudo /opt/venv/bin/pip install --no-cache-dir -r scripts/python_solver/requirements.txt
   ```

- Copy the Python solver code to the location expected used by the plugin:

   ```bash
   sudo install -d -m 0755 /opt/solver/
   sudo cp -a scripts/python_solver/main.py /opt/solver/main.py
   ```

   NOTE: If you change the code of the Python solver, you *must* copy it again.

Finally, to create and run the KWOK cluster with the scheduler+plugin, run the following command:

```bash
kwokctl create cluster --name <cluster_name> --runtime <docker/binary> --config <path/to/cluster-config.yaml>
```

If you later want to delete the cluster, run:

```bash
kwokctl delete cluster --name <cluster_name>
```

## Testing

For testing the plugin, a script has been made to create a bootstrap folder containing all content needed to run the tests, including the built binary, the Python solver code, and all test jobs and configuration files. From the root of the repo, run:

```bash
./make_bootstrap_folder.sh
```

Then, a Workload Once Generator (used in the **paper**) can generate an initial workload on a KWOK cluster running the scheduler with the plugin. The is used for evaluating the optimization capabilities of the plugin under different workloads and cluster sizes.

### Workload Once Generator

For evaluating the plugin, we first run the default scheduler (deterministically) to find 100 seeds where not all pods are running.
Hereafter, we run the default scheduler (as-is) and the scheduler with the `MyPriorityOptimizer` plugin on these seeds.

The script for generating the initial workload and running the tests is `scripts/kwok_workload_once/test_runner.py`. The script can be setup by reading settings from three types of sources, with later ones overriding earlier ones:

1) a workload configuration file
2) a test job file
3) command-line arguments to the script (highest priority)

After choosing one of more of these source, the script can be run to generate the workload and run the tests. An example of how to run the script from the root of the repo or the `bootstrap` folder is:

```bash
python -m scripts.kwok_workload_once.test_runner \
--cluster-name my-cluster \
--kwok-runtime binary \
--job-file data/jobs/kwok_workload_once/<job_file>.yaml \
--workload-config-file data/configs-workload/<workload_config_file>.yaml \
--kwokctl-config-file data/configs-kwokctl/<kwokctl_config_file>.yaml
```

The idea is that `<job_file>.yaml` contains the specific configuration for a job, `<workload_config_file>.yaml` contains the workload configuration to use, and `<kwokctl_config_file>.yaml` contains the KWOK cluster configuration to use (see above [KWOK (recommended)](#kwok-recommended) for more details on this file). In the following sections the [job file](#job-file) and the [workload configuration file](#workload-configuration-file) are described, however, first we shortly describe the deterministic scheduling setup.

#### Deterministic scheduling

To be able to reproduce seeds where not all pods are running using the default scheduler, another plugin called **MyDeterministicScore** is created, located under `pkg/mydeterministicscore/`. This plugin breaks scoring ties by name, disables `DefaultPreemption` plugin and sets `parallelism=1`, helping making scheduling deterministic.

The scheduler configuration file for using this plugin is `data/configs-kwokctl/deterministic-scheduler-config.yaml`.

#### Workload configuration file

As mentioned, the `test_runner.py` script can read a workload configuration file to set up the workload to generate. The file can be used as a template reducing the need to specify all settings in every job file.

An example of a workload configuration file is `data/configs-workload/base.yaml`:

```yaml
kind: WorkloadConfiguration
namespace: test
cpu_per_pod: ["100m", "1000m"]
mem_per_pod: ["100MB","1000MB"]
num_replicas_per_rs: [1, 4]
wait_pod_mode: running
wait_pod_timeout: 2s
settle_timeout_min: 3s
settle_timeout_max: 10s
```

Here the settings, for example, specify that pods should request between 100m and 1000m CPU and between 100MB and 1000MB memory, that ReplicaSets should have between 1 and 4 replicas, the script should wait for pods to be in running state with a timeout of 2s, and that the script should wait between 3s and 10s for the cluster to settle before checking the scheduling results.

#### Job file

To specify the specific configuration for a test job, a job file can be used. The job file can also specify which workload configuration file to use as a template, which kwokctl configuration file to use, which seed file to use, and other settings specific to the job--actually all settings that is possible in the test script.

All the test jobs used previously to evaluate the plugin can be found under `data/jobs/kwok_workload_once`. An example of a job file is `data/jobs/kwok_workload_once/nodes4_pods16_prio4_util095_timeout10.yaml`:

```yaml
workload-config-file: data/configs-workload/base.yaml
kwokctl-config-file: data/configs-kwokctl/plugin-scheduler.yaml
seed-file: data/seeds/kwok_workload_once/nodes4_pods16_prio4_util095.txt
output-dir: results/plugin-scheduler/nodes4_pods16_prio4_util095_timeout10
save-scheduler-logs: true
save-solver-stats: true
solver-trigger: true
override-workload-config:
  num_nodes: 4
  num_pods: 16
  num_priorities: 4
  util: 0.95
override-kwokctl-envs:
- name: SOLVER_PYTHON_TIMEOUT
  value: 10s
```

Here the job file specifies which workload configuration file, kwokctl configuration file, and seed file to use. It also specifies the output directory to save the results to, that scheduler logs and solver stats should be saved, and that the solver should be triggered manually over HTTP. Finally, it overrides/adds some settings in the workload configuration file (number of nodes, pods, priorities, and target utilization) and in the kwokctl configuration file (the Python solver timeout).

#### Bootstrap script

To run the test jobs faster by parallelizing the evaluation using HPC resources, a `bootstrap.sh` script is provided under `scripts/bootstrap/` that can be used to set up a job runner (HPC or VM) and run the tests. The script will ensure all prerequisites are installed and the tests are run.

The bootstrap script accepts parameters that the `test_runner.py` script accepts, but the two main parameters to provide are:

- `--content-dir`: path to the `bootstrap` folder created using the `make_bootstrap_folder.sh` script.
- `--job-file`: path to the job file to run (e.g. see jobs under `data/jobs/`).

Note, that if the binary or docker image is not provided this script can also pull the repo and build the latest version before running the tests.

##### Deterministic jobs with default scheduler

```bash
python -m scripts.helpers.job_generator.py \
--out-dir data/jobs/kwok_workload_once/default-deterministic \
--output-dir results/kwok_workload_once/default-deterministic \
--workload-config-file data/configs-workload/base.yaml \
--kwokctl-config-file data/configs-kwokctl/default-deterministic.yaml \
--seed-file data/seeds/kwok_workload_once/seeds_all.txt \
--num-nodes 4 8 16 32 \
--avg-pods-per-node 4 8 \
--num-priorities 1 2 4 \
--utils 0.90 0.95 1.00 1.05 \
--seeds-not-all-running 100 \
--default-scheduler
```

##### Default scheduler jobs

0.90-0.95 utils runs on the seeds found using the deterministic job generation above.

```bash
python -m scripts.helpers.job_generator \
--out-dir data/jobs/kwok_workload_once/default \
--output-dir results/kwok_workload_once/default \
--workload-config-file data/configs-workload/base.yaml \
--kwokctl-config-file data/configs-kwokctl/default.yaml \
--seed-file data/seeds/kwok_workload_once/ \
--num-nodes 4 8 16 32 \
--avg-pods-per-node 4 8 \
--num-priorities 1 2 4 \
--utils 0.90 0.95 \
--default-scheduler
```

1.00-1.05 utils runs on a fixed seed file with 100 seeds.

```bash
python -m scripts.helpers.job_generator \
--out-dir data/jobs/kwok_workload_once/default \
--output-dir results/kwok_workload_once/default \
--workload-config-file data/configs-workload/base.yaml \
--kwokctl-config-file data/configs-kwokctl/default.yaml \
--seed-file data/seeds/kwok_workload_once/seeds_100.txt \
--num-nodes 4 8 16 32 \
--avg-pods-per-node 4 8 \
--num-priorities 1 2 4 \
--utils 1.00 1.05 \
--default-scheduler
```

##### Python solver jobs

0.90-0.95 utils runs on the seeds found using the deterministic job generation above.

```bash
python -m scripts.helpers.job_generator \
--out-dir data/jobs/kwok_workload_once/plugin-scheduler \
--output-dir results/kwok_workload_once/plugin-scheduler \
--workload-config-file data/configs-workload/base.yaml \
--kwokctl-config-file data/configs-kwokctl/plugin-scheduler.yaml \
--seed-file data/seeds/kwok_workload_once/ \
--num-nodes 4 8 16 32 \
--avg-pods-per-node 4 8 \
--num-priorities 1 2 4 \
--utils 0.90 0.95 \
--timeouts 1 10 20 \
--save-scheduler-logs \
--save-solver-stats \
--solver-trigger
```

1.00-1.05 utils runs on a fixed seed file with 100 seeds.

```bash
python -m scripts.helpers.job_generator \
--out-dir data/jobs/kwok_workload_once/plugin-scheduler \
--output-dir results/kwok_workload_once/plugin-scheduler \
--workload-config-file data/configs-workload/base.yaml \
--kwokctl-config-file data/configs-kwokctl/plugin-scheduler.yaml \
--seed-file data/seeds/kwok_workload_once/seeds_100.txt \
--num-nodes 4 8 16 32 \
--avg-pods-per-node 4 8 \
--num-priorities 1 2 4 \
--utils 1.00 1.05 \
--timeouts 1 10 20 \
--save-scheduler-logs \
--save-solver-stats \
--solver-trigger
```

#### Result replication and running test jobs

After generating the jobs, they are ready to run. For faster evaluation, run them in parallel via the bootstrap script on HPC or VM resources.

To run a test jobs, follow these steps:

  1) Run the provided `make_bootstrap_folder.sh` script from the root of the repo to create the `bootstrap` folder containing the bootstrap script, the built binary, the Python solver code, and all content needed to run the tests (incl. the job files and configuration files, etc.):
  
      ```bash
      ./make_bootstrap_folder.sh
      ```
  
  2) Upload the `bootstrap` folder to HPC/VM provider where it can be found (can be renamed if needed). This folder contains the bootstrap script, the built binary, the Python solver code, and all content needed to run the tests (incl. the job files and configuration files, etc.).
  3) (Optional) Enable SSH access, by adding your public SSH key.
  4) Create an Ubuntu 22.04 instance (no GUI needed) for every job
  5) Once all jobs are done, download the folder containing the results to your local machine.

#### Expected folder structure after running all jobs

Having downloaded the results folder containing results from all jobs - the job files ensures that the results is organized by job type, as follows:

```results/
results/
├── default-deterministic/
│   ├── nodes4_pods16_prio1_util090/
│   │   ├── results.csv
│   │   ├── info.yaml
│   │   ├── seeds-all-running.txt        (if applicable)
│   │   └── seeds-not-all-running.txt    (if applicable)
│   ├── nodes4_pods16_prio1_util095/
│   ├── nodes4_pods16_prio1_util100/
│   ├── nodes4_pods16_prio1_util105/
│   ├── ...
│   └── nodes32_pods256_prio4_util105/
├── default/
│   ├── nodes4_pods16_prio1_util090/
│   │   ├── results.csv
│   │   ├── info.yaml
│   │   ├── seeds-all-running.txt        (if applicable)
│   │   └── seeds-not-all-running.txt    (if applicable)
│   ├── ...
│   └── nodes32_pods256_prio4_util105/
└── plugin-scheduler/
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

#### Analysis

Analyzed results are placed under `analysis/kwok_workload_once/`. They assume the layout shown in [Expected folder structure after running all jobs](#expected-folder-structure-after-running-all-jobs); if yours differs, code changes may be needed.

1. First, merge all job outputs into a single CSV by running `python scripts/kwok_workload_once/combine_results.py` (it writes `analysis/kwok_workload_once/per_combo_results.csv`; adjust input/output paths in the script if needed).
2. Then run `python scripts/kwok_workload_once/plots_and_tables.py` to produce every figure and table used in the report.
   Outputs are saved under `analysis/kwok_workload_once/figures/` and `analysis/kwok_workload_once/tables/`.