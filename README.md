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
nvcc -O3 --use_fast_math softmax_base.cu -o softmax_base
nvcc -O3 --use_fast_math softmax_optimized.cu -o softmax_opt
nvcc -O3 --use_fast_math softmax_unified.cu -o softmax_unified
```

## 📊 Performance Summary (RTX 3060 Example)

| Variant           | Total Runtime | Speedup         | Highlights                                  |
| ----------------- | ------------- | --------------- | ------------------------------------------- |
| `softmax_base`    | **~317 ms**   | 1×              | Includes repeated memcpy and normalization  |
| `softmax_opt`     | **~326 ms**   | ≈1×             | Fully GPU-resident, no host transfers       |
| `softmax_unified` | **~147 ms**   | **2.1× faster** | Unified Memory + Prefetch hides page faults |


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
nvcc -O3 --use_fast_math softmax_base.cu -o softmax_base
nvcc -O3 --use_fast_math softmax_optimized.cu -o softmax_opt
nvcc -O3 --use_fast_math softmax_unified.cu -o softmax_unified
```

## 📊 Performance Summary (RTX 3060 Example)

| Variant           | Total Runtime | Speedup         | Highlights                                  |
| ----------------- | ------------- | --------------- | ------------------------------------------- |
| `softmax_base`    | **~317 ms**   | 1×              | Includes repeated memcpy and normalization  |
| `softmax_opt`     | **~326 ms**   | ≈1×             | Fully GPU-resident, no host transfers       |
| `softmax_unified` | **~147 ms**   | **2.1× faster** | Unified Memory + Prefetch hides page faults |


### 🖼️ Shmoo Runtime Comparison

![Shmoo Runtime Graph](results/shmoo_runtime.png)

The figure above shows **per-variant softmax kernel performance** across input sizes (from 1 K → 16 M elements).  
Each curve corresponds to one of **six GPU kernel variants** tested within each implementation:

| Variant ID | Kernel Name (Conceptual) | Description |
|-------------|--------------------------|--------------|
| 0 | **Naïve Kernel** | Direct exponential + sum reduction using global memory |
| 1 | **Shared Memory Kernel** | Per-block reduction in shared memory |
| 2 | **Warp Reduction Kernel** | Uses warp shuffle (`__shfl_down_sync`) for intra-warp summation |
| 3 | **Warp + Shared Kernel** | Combines warp shuffle and block shared reduction |
| 4 | **Warp + Double Precision Kernel** | Uses higher-precision accumulation for numerical stability |
| 5 | **Warp + Vectorized Kernel** | Vectorized memory loads (`float4`) to improve coalescing |


---

### 📈 Detailed BenchMark Results (Excel)

For complete per-kernel, per-size timing data, open the benchmark spreadsheet:

👉 [**softmax_cuda_analysis.xlsx**](softmax_cuda_analysis.xlsx)

It includes:
- Execution times for all six kernel variants under each memory strategy  
- Derived speedups (Unified vs. Base / Optimized)  
- Aggregated averages and runtime trends used to generate the Shmoo plot

```

```

