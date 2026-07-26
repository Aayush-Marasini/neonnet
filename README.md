# NeonNet

**A zero-allocation, NEON-vectorized ML inference engine in pure C — hand-written kernels for the ARM Cortex-A76 (Raspberry Pi 5), no PyTorch, no BLAS, no OpenVINO.**

> **Status: active development.** The correct baseline engine, the benchmarking
> infrastructure, and the first optimization rung are complete and measured. The
> remaining optimization ladder and the roofline analysis are in progress — see
> [Current State](#current-state) and [Remaining Work](#remaining-work) below.
> Numbers reported here are real measurements on the target hardware, not projections.

---

## What this is

A from-scratch implementation of a neural network's **forward pass** (inference),
written to answer one question:

> **How close to the hardware's theoretical peak can a hand-written inference kernel
> get on an ARM edge device, and which specific optimizations close the gap?**



The model under test is a small MLP trained on MNIST:

```
784 inputs  →  Linear(784, 128)  →  ReLU  →  Linear(128, 10)  →  10 logits
```

Training happens in Python/PyTorch (`python/train.py`); the trained weights are
exported to a documented binary format and loaded by the C engine. **No machine
learning library is used at inference time** — every matrix multiply, activation, and
file-format parse is hand-written C.

---

## Target hardware

- **Board:** Raspberry Pi 5, ARM Cortex-A76 (aarch64), with the official active fan cooler.
- **Toolchain:** `gcc -O3 -mcpu=cortex-a76`, `perf`, `taskset`, `cpupower`, `valgrind`, `gdb`.

---

## Current state

| Phase | Description | Status |
|---|---|---|
| 0 | Environment & scaffolding | ✅ Done |
| 1 | Matrix library & memory discipline | ✅ Done — valgrind-clean |
| 2 | Model load + correctness gate | ✅ Done — C matches PyTorch |
| 3 | Measurement infrastructure | ✅ Done — baseline recorded |
| 4 | Optimization ladder | 🔨 In progress — rung 1 of 4 complete |
| 5 | Roofline analysis | ⏳ Not started |

### Correctness gate

The C forward pass is validated against PyTorch's own outputs on 700 held-out MNIST
test samples. The gate passes when every logit agrees with PyTorch to within 1e-4 and
no prediction disagrees:

```
Total Samples Processed: 700
Max Logit Difference:    1.14e-05
Prediction Mismatches:   0
STATUS: PASS
```

(The residual ~1e-5 difference is float32 non-associativity — the C accumulation order
and PyTorch's differ, and fused multiply-add rounds once where a separate multiply-add
rounds twice. This is expected and bounded, not a logic error; the gate checks bounded
drift and stable argmax, not bit-exactness.)

Trained model test accuracy: **97.84%**.

### Measured results so far
`benc/results.md`
Benchmarking conditions: `performance` CPU governor, process pinned to one core with
`taskset -c 3`, 100 warmup iterations discarded, 10,000 measured iterations, mean
reported with a 95% confidence interval. Hardware counters via `perf stat`.

| Rung | Latency (µs) | Speedup vs naive | L1 miss rate | Notes |
|---|---|---|---|---|
| Naive baseline | 175.50 ±0.03 | 1.00× | 49.49% | Textbook triple loop |
| `i-k-j` reorder | 21.59 ±0.01 | **8.13×** | 0.54% | Stride-1 inner loop |

The naive baseline is heavily **memory-bound**: a 49.49% cache-miss rate from walking
down columns of the weight matrix (a 512-byte stride that defeats the 64-byte cache
line on every step), with instruction throughput suppressed to 1.49 IPC on a core
capable of far more — the execution units stall waiting on RAM.

Reordering the loops to `i-k-j` makes the innermost loop walk memory with stride 1.
The measured 8.13× speedup factors cleanly into its two independent causes:

- **Cache misses collapse** (49.49% → 0.54%), lifting IPC from 1.49 to 3.90 — the
  pipeline stops stalling on memory.
- **Instruction count drops ~3×** for the identical arithmetic — evidence that GCC's
  auto-vectorizer, previously blocked by the column stride, now emits NEON. Confirmed
  by inspecting the disassembly (`fmla v6.4s, v7.4s, v30.4s` — a 4-lane fused
  multiply-add) and the `-fopt-info-vec` report.

A consequence worth stating plainly: because the compiler auto-vectorizes the clean
stride-1 loop, the hand-written NEON kernel (rung 4, below) will be racing **compiler
NEON, not scalar code**. Matching or beating the compiler's own auto-vectorizer at
identical optimization flags is the honest headline comparison this project is building
toward — not beating an artificially weak `-O0` baseline.

---

## Remaining work

The optimization ladder continues, one rung at a time. After **every** rung the
correctness gate is re-run and the benchmark re-measured — an optimization that breaks
the output is not an optimization.

- **Cache blocking (tiling).** Being implemented and measured; at batch size N=1 this
  forward pass is a streaming vector–matrix product with no temporal reuse of the
  weight matrix, so tiling is expected to be near-neutral here and pays off only at
  N>1. The result and that analysis will be recorded rather than assumed.
- **Memory alignment.** `aligned_alloc(64, ...)` to the cache-line boundary for aligned
  SIMD loads.
- **Hand-written NEON kernel.** An explicit `arm_neon.h` inner kernel with a scalar tail
  loop for non-multiple-of-4 remainders, benchmarked against the compiler's
  auto-vectorized build at the same flags.
- **Zero-allocation hot path.** Removing the per-call result-matrix allocation (the
  project's namesake goal) by reusing pre-allocated buffers.
- **Roofline analysis.** Placing the naive and optimized kernels against the A76's
  measured compute and memory-bandwidth ceilings to determine, with evidence, whether
  each kernel is compute- or memory-bound.
- **Narrative writeup & reproduction guide.** Full problem → approach → results → how to
  reproduce on a Pi 5.

---

## Repository layout

```
neonnet/
  src/
    matrix.{h,c}     # core Matrix type + naive/reordered ops
    layers.{h,c}     # linear, relu, softmax, argmax
    model.{h,c}      # weight loading + forward pass
    main.c           # correctness harness (the Phase 2 gate)
  bench/
    benchmark.c      # timing harness with confidence intervals
    results.md       # per-rung measured results
  python/
    train.py         # train the MLP, export weights + test data
    verify_export.py # pure-NumPy re-check of the export
    export_format.md # binary weight-file specification
  bench.sh           # reproducible governor + pin + perf wrapper
  Makefile
```

---

## Methodology (the rules this project holds itself to)

1. **Correctness before speed.** The correctness gate is sacred; no optimization ships
   until it passes.
2. **Measure every change.** A change with no measured number attached taught nothing.
3. **Fair comparisons.** Every version is compiled at identical optimization flags. The
   target to beat is the compiler's own auto-vectorization, not a strawman.
4. **Re-validate after every optimization.** SIMD remainder handling is the classic
   place output silently breaks.
5. **Commit at every milestone.** The incremental git history is part of the record.

---

## Reproducing

Requires a Raspberry Pi 5 (or another aarch64 Cortex-A76 target) with the toolchain
listed above.

```bash
# 1. Train the model and export weights (Python side)
python3 python/train.py

# 2. Build the engine and benchmark binaries
make clean && make

# 3. Run the correctness gate
./neonnet          # expect STATUS: PASS

# 4. Run the full benchmark suite (governor + core pin + perf counters)
./bench.sh
```

*This README will grow into the full narrative writeup as the remaining phases land.*
