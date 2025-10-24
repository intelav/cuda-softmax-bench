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


## 📊 Performance Summary (RTX 3060 Example)

| Variant | Total Runtime | Speedup | Highlights |
|----------|----------------|----------|-------------|
| `softmax_base` | **~317 ms** | 1× | Includes repeated memcpy and normalization |
| `softmax_opt` | **~326 ms** | ≈1× | Fully GPU-resident, no host transfers |
| `softmax_unified` | **~147 ms** | **2.1× faster** | Unified Memory + Prefetch hides page faults |

---

### 📈 Detailed Results (Excel)

You can open or download the full benchmark results here:
👉 [**softmax_cuda_analysis.xlsx**](softmax_cuda_analysis.xlsx)

---

### 🖼️ Shmoo Runtime Comparison

![Shmoo Runtime Graph](results/shmoo_runtime.png)
```
