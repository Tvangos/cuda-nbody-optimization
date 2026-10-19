# CUDA N-Body Simulation — GPU Performance Engineering

![Language](https://img.shields.io/badge/Language-C%20%7C%20CUDA%20C%2B%2B-2475ba) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Tesla%20K80-76b900) ![Architecture](https://img.shields.io/badge/Architecture-x86__64%20%7C%20CUDA-6f42c1) ![Framework](https://img.shields.io/badge/Framework-CUDA%20Toolkit-008b8b) ![Parallelism](https://img.shields.io/badge/Parallelism-OpenMP%20%7C%20Multi--GPU-238636)

A performance-engineering study of a direct, all-pairs N-body simulation, progressing from sequential and OpenMP CPU implementations to an optimized CUDA implementation. The project explores how memory layout, data reuse, kernel organization, and workload distribution affect throughput in an **O(N²)** computational workload.

> **Best recorded result:** **423.34 billion interactions/second** for `opt10`, compared with **33.42 billion interactions/second** for the initial CUDA implementation — a **12.67× ratio of reported throughputs**. These are *not* directly comparable end-to-end speedups: the baseline includes GPU transfer time, while `opt10` reports compute/synchronization time and can use multiple GPUs. See [Benchmark methodology and limitations](#benchmark-methodology-and-limitations).

## Performance at a glance

The repository contains measured results for each optimization stage. Numbers below come from the checked-in `results_*.csv` files and represent trimmed means (12 runs, dropping each metric's minimum and maximum).

| Version / optimization technique | Avg. time (s) | Throughput (billion interactions/s) | Reported throughput vs. initial CUDA |
|:--|--:|--:|--:|
| `GPU` — Baseline CUDA: one block per galaxy | 1.2852 | 33.42 | 1.00× |
| `opt1` — Data-layout redesign: split positions and velocities | 1.3762 | 31.21 | 0.93× |
| `opt2` — Shared-memory tiling: reuse cached body coordinates | 0.6283 | 68.36 | 2.05× |
| `opt3` — Loop unrolling: improve instruction-level parallelism | 0.6248 | 68.76 | 2.06× |
| `opt4` — CUDA streams and multi-block kernels: concurrent galaxies | 0.4457 | 96.38 | 2.88× |
| `opt5` — Read-only caching and fast math: `__ldg`, `rsqrtf`, FMA | 0.4295 | 100.04 | 2.99× |
| `opt6` — Pinned-memory/asynchronous-transfer experiments | 0.4280 | 100.35 | 3.00× |
| `opt7` — Alternative SoA/texture-based force-kernel experiment | 0.4260 | 100.84 | 3.02× |
| `opt8` — Stream scheduling and transfer-overlap variant | 0.4277 | 100.49 | 3.01× |
| `opt9` — Fused timestep kernel with double-buffered positions | 0.4187 | 102.65 | 3.07× |
| **`opt10` — Full SoA, shared-memory tiling and multi-GPU partitioning** | **0.1016** | **423.34** | **12.67×*** |

\* **Interpret the final ratio cautiously.** `opt10` measures a different timing scope and dynamically uses all visible CUDA devices. The original experiment report specifies **2 × NVIDIA Tesla K80** GPUs on its test node, but the CSV files do not attach per-run hardware/device-count metadata or establish a controlled single-device comparison. `opt2.2` is excluded because its saved CSV has a different schema. The written report groups experiments by *conceptual optimization stages*, while the GitHub directories include additional experimental variants; from `opt7` onward, the descriptions above follow the **actual checked-in implementations** rather than assuming the report chapter number equals the source-directory number.

## What I optimized

| Technique | Implementation idea | Performance rationale |
|:--|:--|:--|
| **Shared-memory tiling** | Cache batches of particle positions in CUDA shared memory before evaluating pairwise interactions. | Reuse loaded positions across threads instead of repeatedly fetching them from device memory. |
| **Data layout** | Move from an array of full body records toward separate coordinates/velocities, ultimately using six float arrays in `opt10` (SoA). | Improve contiguous access patterns for adjacent GPU threads. |
| **Kernel decomposition** | Split velocity updates and position integration into separate kernels. | Enforce a device-wide ordering between computation phases rather than relying on block-local synchronization. |
| **Parallel granularity** | Progress from one CUDA block per independent system to multiple blocks per system. | Expose more parallel work to the GPU scheduler. |
| **Arithmetic optimization** | Use `rsqrtf`, fused multiply-add (`fmaf`), and loop-unrolling directives in the final tiled kernel. | Reduce costly arithmetic operations and improve instruction throughput, subject to floating-point accuracy trade-offs. |
| **Streams and transfers** | Explore asynchronous transfers and multiple CUDA streams in intermediate versions. | Investigate opportunities for overlapping transfers and work on independent systems. |
| **Multi-GPU distribution** | In `opt10`, detect available CUDA devices and partition independent systems among them, with per-device buffers and work streams. | Run separate subsets of the simulation on multiple devices when available. |

The numbered implementations are preserved to show the optimization process, including experiments that did **not** improve performance. The final implementation uses a **256-thread block / 256-body shared-memory tile** for its force kernel and launches separate velocity and position kernels for each time step. Although it allocates multiple streams per device, the final compute path uses **one simulation stream per device**; it should not be described as multi-stream overlap within each GPU.

## Workload and implementation

The default benchmark consists of:

- **32** independent systems (called *galaxies* in the original lab)
- **8,192** bodies per system (**262,144** total)
- **20** simulation steps with `dt = 0.01`
- **42,949,672,960** evaluated pairwise interactions, counting all pairs in each system at each time step

Each body accumulates a softened inverse-distance interaction from all bodies in its system, updates velocity, and then updates position. This implementation uses **unit-weight interactions**; the dataset generator's central-mass parameter is used when constructing initial orbital velocities, **not** as a per-body mass in the force kernel. It is a computational N-body benchmark rather than a full astrophysical gravity solver.

## Build and run

**Requirements:** NVIDIA CUDA-capable GPU(s), CUDA Toolkit (`nvcc`), `make`, Python 3, Bash, and `bc` for the benchmark scripts. The OpenMP target uses Intel `icx` with `-qopenmp`.

Run the following from the **repository root**:

```bash
# Generate the binary input dataset
python3 generate_dataset.py

# Build and run the final CUDA version
make -f Makefile_cuda DIR=opt10
./opt10/nbody
```

To benchmark a specific CUDA version, for example the initial implementation or final version:

```bash
./run_cuda.sh GPU 1 0
./run_cuda.sh opt10 1 0
```

The arguments are `<version> <OMP_NUM_THREADS> <DEBUG_FLAG>`. The script rebuilds the selected implementation, regenerates a dataset on each run, runs it 12 times, and writes `VERSION/results_VERSION.csv`. The thread count primarily affects the optional OpenMP reference run, not GPU block sizing.

For the CPU OpenMP implementation:

```bash
make -f Makefile_omp
./run_omp.sh omp 8
```

For an optional CPU/GPU comparison on the final version:

```bash
DEBUG=1 ./opt10/nbody
```

This executes the OpenMP reference and reports coordinate differences. It is computationally expensive; inspect the reported mismatches rather than assuming the benchmark script automatically enforces correctness.

## Benchmark methodology and limitations

The recorded CUDA results use **12 runs** per version. `run_cuda.sh` computes a trimmed mean of time and throughput after removing the lowest and highest measured value independently. The Python generator creates a new randomized dataset for every run and does **not** fix a random seed, although the problem dimensions remain constant.

**Comparability matters:**

1. The initial `GPU` timer covers host-to-device transfer, CUDA execution, and device-to-host transfer.
2. The final `opt10` timer starts **after** input transfers and stops **before** output transfers; it covers launching and synchronizing the GPU computation.
3. `opt10` distributes systems across all visible GPUs, so its result may reflect additional hardware as well as software optimization.
4. The accompanying lab report identifies the test node as two Intel Xeon E5-2695 v3 CPUs and two NVIDIA Tesla K80 GPUs, but the individual CSV files do not identify which devices were used for each run. A reproducible performance claim should record the GPU model and count, CPU, CUDA compiler version, and the exact same timing boundaries for every version.

Accordingly, the table is a transparent record of **reported benchmark outcomes**, not a claim of controlled single-GPU or end-to-end scaling. A next step is to repeat the benchmarks with fixed input data, device selection, and separate compute-only and end-to-end timings.

## Repository layout

```text
.
├── Code/                  # Sequential C baseline
│   └── nbody.c
├── omp/                   # OpenMP CPU version
│   └── nbody.c
├── GPU/                   # Initial CUDA version + recorded results
├── opt1/ ... opt10/      # Named optimization experiments; see performance table
├── opt2.2/                # Additional exploratory experiment
├── Makefile_cuda          # CUDA builds (DIR=<version>)
├── Makefile_omp           # Intel OpenMP build
├── generate_dataset.py    # Binary workload generator
├── run_cuda.sh            # Repeated CUDA performance measurements
├── run_omp.sh             # Repeated OpenMP CPU measurements
├── run_once.sh            # Single CUDA build/run
└── timer.h                # Wall-clock timing helper
```

**Project context:** Developed as part of a High-Performance Computing lab. The focus is on **profiling-oriented optimization, GPU programming, and experimentally evaluating performance trade-offs**, rather than on visualization or astrophysical fidelity.
