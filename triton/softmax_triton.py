import torch
import triton
import triton.language as tl
import time
import csv

# ============================================================
# 🧠 Triton Softmax Kernel Variants
# ============================================================

@triton.jit
def softmax_naive_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0)
    ex = tl.exp(x)
    denom = tl.sum(ex, axis=0)
    y = ex / denom
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_block_shared_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    denom = tl.sum(ex, axis=0)
    y = ex / denom
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_warp_reduce_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    partial = tl.sum(ex, axis=0)
    y = ex / partial
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_warp_shared_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    max_val = tl.max(x, axis=0)
    ex = tl.exp(x - max_val)
    block_sum = tl.sum(ex, axis=0)
    y = ex / block_sum
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_double_accum_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x).to(tl.float64)
    denom = tl.sum(ex, axis=0)
    y = (ex / denom).to(tl.float32)
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_vectorized_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    denom = tl.sum(ex, axis=0)
    y = ex / denom
    tl.store(Y_ptr + offs, y, mask=mask)

# ============================================================
# 🧪 Benchmark helper
# ============================================================

def benchmark_softmax_triton(kernel, x, BLOCK_SIZE=1024, vec4=False):
    y = torch.empty_like(x)
    N = x.numel()
    if vec4:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE'] * 4),)
    else:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    torch.cuda.synchronize()
    start = time.time()
    kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000  # ms


def compute_tflops(N, ms):
    # Softmax ≈ 3N FLOPs (exp + sum + div)
    flops = 3 * N
    return (flops / (ms / 1e3)) / 1e12


# ============================================================
# 🚀 Main benchmark
# ============================================================
if __name__ == "__main__":
    # Detect safe upper limit based on free GPU memory
    free_mem, total_mem = torch.cuda.mem_get_info()
    safe_bytes = int(free_mem * 0.8)  # keep 20% margin
    max_elems = safe_bytes // (4 * 2)  # two float32 arrays (x,y)

    sizes = [n for n in [1<<i for i in range(10, 31)] if n <= max_elems]

    variants = [
        ("naive", softmax_naive_kernel),
        ("block_shared", softmax_block_shared_kernel),
        ("warp_reduce", softmax_warp_reduce_kernel),
        ("warp_shared", softmax_warp_shared_kernel),
        ("double_accum", softmax_double_accum_kernel),
        ("vectorized", softmax_vectorized_kernel),
    ]

    print(f"[INFO] Limiting maximum N to {max_elems:,} elements (~{safe_bytes / 1e9:.2f} GB usable)")

    csv_file = "softmax_triton_shmoo.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Full Kernel Name", "Elements (N)", "Runtime (ms)", "TFLOPS"])

        total_start = time.time()
        total_runtime_ms = 0.0

        for N in sizes:
            x = torch.randn(N, device='cuda', dtype=torch.float32)
            for name, kernel in variants:
                full_name = f"softmax_{name}_kernel"
                vec4 = (name == "vectorized")
                t = benchmark_softmax_triton(kernel, x, vec4=vec4)
                tflops = compute_tflops(N, t)
                total_runtime_ms += t
                print(f"N={N:10d}, Variant={full_name:25s}, Runtime={t:8.3f} ms, TFLOPS={tflops:6.3f}")
                writer.writerow([full_name, N, f"{t:.3f}", f"{tflops:.3f}"])

        total_end = time.time()
        total_elapsed_ms = (total_end - total_start) * 1000
        print(f"\nTotal benchmark runtime: {total_elapsed_ms:.3f} ms")

        # Write total runtime summary to CSV
        writer.writerow([])
        writer.writerow(["Total Benchmark Runtime (ms)", f"{total_elapsed_ms:.3f}"])
        writer.writerow(["Sum of Kernel Runtimes (ms)", f"{total_runtime_ms:.3f}"])

    print(f"Results saved to {csv_file}")
