# 🚀 CUDA Softmax Benchmark Suite

This repository provides a **comprehensive CUDA Softmax microbenchmark** comparing three variants of GPU implementations:

- A **baseline** version using global memory and explicit `cudaMemcpy`,
- A **GPU-resident optimized** version where data stays entirely on-device,
- A **Unified Memory + Prefetch** version that leverages `cudaMallocManaged` for simplified memory management.

It demonstrates how **memory transfer strategies** and **kernel chaining** impact performance on modern GPUs.

![SoftMax Function](results/softmax_func.png)

---

## 🧩 Overview

| Variant       | File                   | Memory Strategy                      | Description                                                                    |
| ------------- | ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------ |
| **Baseline**  | `softmax_base.cu`      | Global Memory (`cudaMalloc`)         | Copies data between host and device for each run                               |
| **Optimized** | `softmax_optimized.cu` | GPU-Resident (`cudaMalloc`)          | Keeps data on GPU, performs reduction + normalization in-device                |
| **Unified**   | `softmax_unified.cu`   | Unified Memory (`cudaMallocManaged`) | Uses page migration + `cudaMemPrefetchAsync()` for seamless host–device access |

---

# 🚀 CUDA Softmax Benchmark Suite

This repository provides a **comprehensive CUDA Softmax microbenchmark** comparing three variants of GPU implementations:

- A **baseline** version using global memory and explicit `cudaMemcpy`,
- A **GPU-resident optimized** version where data stays entirely on-device,
- A **Unified Memory + Prefetch** version that leverages `cudaMallocManaged` for simplified memory management.

It demonstrates how **memory transfer strategies** and **kernel chaining** impact performance on modern GPUs.

---

## 🧩 Overview

| Variant       | File                   | Memory Strategy                      | Description                                                                    |
| ------------- | ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------ |
| **Baseline**  | `softmax_base.cu`      | Global Memory (`cudaMalloc`)         | Copies data between host and device for each run                               |
| **Optimized** | `softmax_optimized.cu` | GPU-Resident (`cudaMalloc`)          | Keeps data on GPU, performs reduction + normalization in-device                |
| **Unified**   | `softmax_unified.cu`   | Unified Memory (`cudaMallocManaged`) | Uses page migration + `cudaMemPrefetchAsync()` for seamless host–device access |

---

## ⚙️ Build Instructions

Ensure CUDA Toolkit ≥ **12.0** and Nsight Systems are installed.  
Then compile each variant:

```bash
nvcc -O3 --use_fast_math softmax_base.cu -Icuda-samples/Common -o softmax_base
nvcc -O3 --use_fast_math softmax_optimized.cu -Icuda-samples/Common -o softmax_opt
nvcc -O3 --use_fast_math softmax_unified.cu -Icuda-samples/Common -o softmax_unified
```

## 📊 Performance Summary (RTX 3060, 1 B Elements — End-to-End Runtime)

| Variant | Total Runtime (ms) | Speedup vs Baseline | Key Highlights |
|----------|-------------------:|--------------------:|----------------|
| `softmax_base` | **12011 ms** | 1.00× | Includes explicit CPU→GPU memcpy, normalization, and full reduction passes |
| `softmax_opt` | **15000 ms** | 0.80× | GPU-resident compute, but limited by separate allocations and no page overlap |
| `softmax_unified` | **679 ms** | **≈ 17.7× faster** | Unified Memory + Prefetch eliminates memcpy overhead and overlaps page migration with compute |

*(Data from [base_1b.csv](base_1b.csv) ,[opt_1b.csv](opt_1b.csv) and  [unified_1b.csv](unified_1b.csv); 1 billion float inputs on RTX 3060 12 GB.)*

**Observation:**  
Even though `softmax_opt` avoids host transfers, its separate GPU allocation still causes synchronization stalls and slower initialization.  
`softmax_unified` dramatically reduces runtime by prefetching pages into GPU memory, enabling near-full bandwidth utilization during compute.


### 🖼️ Shmoo Runtime Comparison

![Shmoo Runtime Graph](results/shmoo_runtime.png)

The figure above shows **per-variant softmax kernel performance** across input sizes (from 1 K → 1 billion elements).  
Each curve corresponds to one of **six GPU kernel variants** tested within each implementation:

| Variant ID | Kernel Name (Conceptual)           | Description                                                     |
| ---------- | ---------------------------------- | --------------------------------------------------------------- |
| 0          | **Naïve Kernel**                   | Direct exponential + sum reduction using global memory          |
| 1          | **Shared Memory Kernel**           | Per-block reduction in shared memory                            |
| 2          | **Warp Reduction Kernel**          | Uses warp shuffle (`__shfl_down_sync`) for intra-warp summation |
| 3          | **Warp + Shared Kernel**           | Combines warp shuffle and block shared reduction                |
| 4          | **Warp + Double Precision Kernel** | Uses higher-precision accumulation for numerical stability      |
| 5          | **Warp + Vectorized Kernel**       | Vectorized memory loads (`float4`) to improve coalescing        |

## 🚀 1 Billion-Element Benchmark (RTX 3060 12 GB)

The following tables summarize execution times for each CUDA Softmax kernel variant when processing **1 billion (1e9) FP32 elements**.  
Both **Baseline** (explicit CPU→GPU memory copies) and **Unified Memory + Prefetch** configurations were tested.

### 🔹 Execution Time & Speedup Summary

| Kernel ID | Kernel Variant Name | Baseline Mode (ms) | Unified Memory (ms) | Unified vs Baseline Speedup | Comments |
|:--:|:--|--:|--:|--:|--|
| 0 | Naïve (Global Memory) | 62.70 | 32.62 | 1.92 × | Unified memory nearly halves total runtime by avoiding explicit H2D copies |
| 1 | Shared Memory Kernel | 62.29 | 32.61 | 1.91 × | Copy overhead dominates baseline; Unified mode overlaps page migration |
| 2 | Warp Reduction Kernel | 53.40 | 26.99 | 1.98 × | Better arithmetic intensity; Unified mode sustains higher bandwidth |
| 3 | Warp + Shared Reduction | 54.62 | 27.24 | 2.01 × | Shared reuse + prefetch = best H2D latency hiding |
| 4 | Warp + Double Precision Accumulate | 53.40 | 26.73 | 2.00 × | Higher-precision accumulation benefits from prefetched pages |
| 5 | **Warp + Vectorized Kernel (Best)** | 51.61 | **25.82** | **2.00 × faster** | Lowest register pressure and highest SM occupancy; prefetch fully hides paging |

*(Data from [base_1b.csv](base_1b.csv) and [unified_1b.csv](unified_1b.csv); 1 billion float inputs on RTX 3060 12 GB.)*

---

### 🧠 Profiling Insights
- **Unified Memory Prefetch** nearly doubles throughput by eliminating manual memory copies and overlapping page migration with computation.  
- The **Warp + Vectorized Kernel** remains the fastest overall (≈ 25.8 ms end-to-end).  
- Nsight Compute shows ~36 registers/thread and ~70 % SM occupancy for this kernel, allowing more warps per SM and superior latency hiding.  
- Beyond ≈ 1 B elements the workload becomes **bandwidth-bound**, so Unified Memory’s page migration overlap is crucial to maintaining high throughput.  

---


## 🚀 Running the Benchmark

Once compiled, you can run each binary with a command-line argument specifying the **input vector length (N)**.  
Each benchmark sweeps sizes from 1 K up to `Nmax`, doubling per step.

### 🔹 Example runs

Run with **10 million elements**:

```bash
./softmax_base 10000000      > shmoo_base_10m.csv
./softmax_opt   10000000     > shmoo_opt_10m.csv
./softmax_unified 10000000   > shmoo_unified_10m.csv
```

Run with **100 million elements**:

```bash
./softmax_base 100000000     > shmoo_base_100m.csv
./softmax_opt   100000000    > shmoo_opt_100m.csv
./softmax_unified 100000000  > shmoo_unified_100m.csv
```

Run with **1 billion elements** (works with GPU memory >= 12GB):

```bash
./softmax_base 1000000000     > shmoo_base_1b.csv
./softmax_opt   1000000000    > shmoo_opt_1b.csv
./softmax_unified 1000000000  > shmoo_unified_1b.csv
```
## 🔍 Profiling Analysis (Nsight Compute)

Based on the Nsight Compute profiling results, the Unified Memory implementation demonstrates faster initialization because both CPU and GPU share a common memory space.
This allows all input data to be directly initialized on the GPU without explicit data transfers.

In contrast, the Softmax Base implementation allocates separate memory regions for the CPU and GPU.
As a result, the CPU must explicitly copy data to GPU memory before execution, introducing additional transfer overhead and increasing total runtime.

![Nsight Compute Profiling](results/ncu_profiling.png)

### Kernel Performance Summary
![CUDA Kernel Performance Summary](results/kernel_report.png)

🧠 Why softmax_warp_vectorized_kernel Achieves Better Occupancy

The softmax_warp_vectorized_kernel achieves the highest SM occupancy (~70–72%) and lowest register pressure (~36 registers/thread) among all variants.
This improvement comes from reduced live variable usage per thread — vectorized memory loads and fused arithmetic reduce the number of temporary variables that must stay resident in registers.

With fewer live registers per thread:

- More warps can be scheduled concurrently on each SM.

- The GPU hides latency more effectively.

- Shared memory pressure remains low since intermediate values are reused efficiently.

As a result, this kernel attains the best balance between compute utilization and resource footprint, leading to the highest throughput and smooth scaling across input sizes.
---
### 🔹 Triton vs CUDA C++ — Execution Time & TFLOPS Summary (RTX 3060 — Full Benchmark Sweep)

| Kernel ID | Implementation | Time (1 B Elements) (ms) | Total Run Time (ms) | TFLOPS (Per Kernel) | Comments |
|:--:|:--|--:|--:|--:|--|
| 0 | **CUDA C++ Unified** | **25.8** | ≈ 679 | 0.12 | `softmax_warp_vectorized_kernel` — Unified Memory + Prefetch hides page-migration costs; benchmark includes six CUDA kernels across 21 sizes. |
| 1 | **Triton Warp-Reduce** | **25.7** | ≈ 1703 | 0.12 | `softmax_warp_reduce_kernel` — Fused GPU kernel implemented in Triton; same six variants × 21 sizes; Python launch overhead adds latency. |

*(Both implementations executed the same six softmax kernel variants up to 1 billion elements.  
Per-kernel timing represents 1 B-element throughput; total time reflects the full multi-variant benchmark sweep.)*

*(Data from [unified_1b.csv](unified_1b.csv) and [softmax_triton_shmoo.csv](triton/softmax_triton_shmoo.csv); 1 billion float inputs on RTX 3060 12 GB.)*
