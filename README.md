# RooFit Multi-CPU Benchmark

This repository contains a small, reproducible benchmark to study how RooFit’s `NumCPU` option uses multiple CPU cores and how the observed scaling relates to **Amdahl’s law**.

The benchmark:
- Builds a fixed RooFit model (Breit–Wigner ⊗ Gaussian + exponential background),
- Generates a toy dataset with a chosen number of events and a fixed random seed,
- Repeats the fit for several `NumCPU` values,
- Measures wall time and CPU times,
- Compares the observed speedup to an Amdahl-type expectation and produces summary plots.

A more detailed explanation of the physics/computing context and of Amdahl’s law will be provided later in a separate PDF document in this repository.

---

## Repository contents

- `env_check.py`  
  Environment checker for the benchmark. Verifies that the required Python modules and external tools are available and prints clear diagnostics if something is missing.

- `roofit-multicpu-benchmark.py`  
  Main benchmark script.  
  It:
  - detects basic system information (OS, CPU model, core/thread layout),
  - sets up the RooFit model,
  - generates the dataset,
  - runs fits with different `NumCPU` settings,
  - measures timing information and computes derived quantities (e.g. speedups, Amdahl-based expectations),
  - writes all results to the `results/` directory.

- `results/`  
  Example output from a completed run, organised by event size and run label.  
  Inside each event-size directory you will find:
  - `data_dir/csv/`  – timing results,
  - `data_dir/json/` – configuration and system info,
  - `data_dir/logs/` – full benchmark log,
  - `plot_dir/`      – PNG/SVG plots (wall time, system time, speedup, Amdahl comparison, topology).

---

## How to get the repository

From a terminal:

```bash
git clone https://github.com/usozbilir/roofit-multicpu-benchmark.git
cd roofit-multicpu-benchmark
```

You now have the scripts and an example `results/` tree locally.

---

## Recommended usage

1. **Check the environment**

   Before running the benchmark, verify that your environment is ready:

   ```bash
   python env_check.py
   ```

   If everything is OK, the script will report success and exit.  
   If not, it will list what is missing and give hints on how to fix it.  
   Please follow the guidance printed by `env_check.py` and re-run it until it reports success.

2. **Run the benchmark**

   Once the environment check passes, run:

   ```bash
   python roofit-multicpu-benchmark.py
   ```


---

## Interpreting the results

The benchmark is intended to be used mainly for **educational and exploratory purposes**:

- to see how RooFit’s `NumCPU` setting behaves on different machines,
- to observe that speedup is limited by a non-parallel fraction of the code,
- to compare real measurements to an Amdahl-type model.

The details of Amdahl’s law and its interpretation in this context will be documented in a dedicated slide deck or PDF that will be added to this repository in the future.
