# NeonNet

**A from-scratch ML inference engine in pure C — hand-written kernels for the ARM Cortex-A76 (Raspberry Pi 5). No PyTorch, no BLAS, no OpenVINO at inference time.**

> **On the name.** "Zero-allocation, NEON-vectorized" is the *destination* this project is climbing toward. The correct engine, measurement infrastructure, and the first five optimization rungs (including the hand-written NEON kernel and zero-allocation hot path) are complete and measured. This README states exactly what is and isn't done, and every number in it is a real measurement on the target board — not a projection.
>
> **Full per-rung results, conditions, and interpretation live in [`bench/results.md`](bench/results.md).**

---

## What this is

A from-scratch implementation of a neural network's **forward pass** (inference), built to answer one question:

> **How close to the hardware's theoretical peak can a hand-written inference kernel get on an actively-cooled ARM edge device, and which specific optimizations close the gap?**

The model under test is a small MLP trained on MNIST:

784 inputs  ->  Linear(784, 128)  ->  ReLU  ->  Linear(128, 10)  ->  10 logits


Training happens in Python/PyTorch (`python/train.py`); the trained weights are exported to a documented binary format and loaded by the C engine. **No machine-learning library is used at inference time** — every matrix multiply, activation, and file-format parse is hand-written C.

This is a **systems-understanding** project, not a code-delivery one. A correct engine with no benchmarks and no analysis would be a *failed* project. The measurements and the explanations are the deliverable.

---

## Target hardware

- **Board:** Raspberry Pi 5, ARM Cortex-A76 (aarch64), with the official active fan cooler.
- **Toolchain:** `gcc -O3 -mcpu=cortex-a76`, `perf`, `taskset`, `cpupower`, `valgrind`, `gdb`.

---

## Current state

| Phase | Description | Status |
|---|---|---|
| 0 | Environment & scaffolding | Done |
| 1 | Matrix library & memory discipline | Done — valgrind-clean |
| 2 | Model load + correctness gate | Done — C matches PyTorch |
| 3 | Measurement infrastructure | Done — baseline recorded |
| 4 | Optimization ladder | In progress — 5 of 6 rungs done (reorder, tiling, aligned, NEON, zero-alloc) |
| 5 | Roofline analysis | Not started |

### Correctness gate

The C forward pass is validated against PyTorch's own outputs on 700 held-out MNIST test samples. It passes when every logit agrees with PyTorch to within 1e-4 and no prediction disagrees:

Total Samples Processed: 700
Max Logit Difference:    1.14e-05
Prediction Mismatches:   0
STATUS: PASS


The residual ~1e-5 is float32 non-associativity, not a logic error: the C accumulation order and PyTorch's differ, and fused multiply-add rounds once where a separate multiply-then-add rounds twice. The gate checks *bounded* drift and stable argmax, not bit-exactness — which is why the bar is 1e-4 and not zero. This gate is re-run after **every** optimization rung.

Trained model test accuracy: **97.84%**.

---

## Measured results so far

Conditions: `performance` CPU governor, process pinned to one core (`taskset -c 3`), 100 warmup iterations discarded, 10,000 measured iterations, single-image input (batch N=1). Mean latency reported with a 95% confidence interval (the CI reflects precision of the mean, not the spread of individual runs). Hardware counters via `perf stat`.

| Rung | Latency (us) | vs naive | vs prev | L1 miss | IPC | Notes |
|---|---|---|---|---|---|---|
| Naive baseline | 175.50 +/-0.03 | 1.00x | — | 49.49% | 1.49 | Textbook triple loop |
| `i-k-j` reorder | 21.59 +/-0.01 | **8.13x** | 8.13x | 0.54% | 3.90 | Stride-1 inner loop |
| Cache tiling | 27.09 +/-0.01 | 6.48x | **0.80x** | 3.08% | 3.27 | Regression at N=1 (see below) |
| Aligned (Rung 3 peak)| 16.66 +/-0.03 | **10.54x** | 1.30x | 6.0%* | 3.23 | Unstable compiler peak (see below) |
| NEON (hand, 1-acc) | 24.96 +/-0.04 | 7.03x | 0.67x | 0.4% | 3.75 | Lost to GCC auto-vectorizer |
| Zero-alloc scalar | 24.30 +/-0.03 | 7.22x | **0.69x** | 0.53% | 3.39 | Regression from Rung 3 peak |
| Zero-alloc NEON | 20.77 +/-0.03 | **8.45x**| 1.20x | 0.53% | 3.87 | Becomes fastest stable path |

<sub>*Aligned miss-rate/IPC from a slightly noisier profiling session; latency is clean. Full numbers and per-rung interpretation in [`bench/results.md`](bench/results.md).</sub>

### Rung 1 — loop reorder (the big one)

The naive baseline is heavily **memory-bound**: a 49.49% cache-miss rate from walking *down* columns of the weight matrix — a 512-byte stride that defeats the 64-byte cache line on every step — with instruction throughput suppressed to 1.49 IPC on a core capable of far more. The execution units stall waiting on RAM.

Reordering the loops to `i-k-j` makes the innermost loop walk memory with stride 1. The 8.13x speedup factors cleanly into two independent causes: cache misses collapse (49.49% -> 0.54%), lifting IPC from 1.49 to 3.90; **and** instruction count drops ~3x for the identical arithmetic — the fingerprint of GCC's auto-vectorizer, previously blocked by the column stride, now emitting NEON. Confirmed in the disassembly (`fmla v6.4s, v7.4s, v30.4s` — a 4-lane fused multiply-add).

### The Surprises: Findings that shifted the project

The point of this project is measuring rather than assuming. Several rungs came back the *opposite* of what was predicted and fundamentally changed the optimization strategy.

**Cache tiling made it *slower* (0.80x), not neutral.** The prediction was near-neutral: at N=1 the forward pass is a streaming vector-matrix product with each weight read exactly once, so cache blocking has no temporal reuse to capture. Instead of neutral, tiling *regressed* — 21.59 -> 27.09 us, with the miss rate rising to 3.08%. Mechanism: splitting one contiguous stride-1 row-scan into blocked sub-passes broke the pattern the hardware prefetcher was exploiting and added loop-bookkeeping overhead, with zero reuse to offset it. 

**The 16.66 µs peak was a compiler ghost, not alignment.** Rung 3 produced an unpredicted 16.66 µs peak (initially attributed to 64-byte `__builtin_assume_aligned` hints). Disassembly showed GCC emitted a 2x unrolled inner loop. However, Rung 5 falsified the alignment hypothesis: when we stripped heap allocation out of the hot path, the compiler spooked (likely due to `memset` breaking escape analysis) and refused to unroll the loop, dropping scalar performance back down to 24.30 µs. The 16.66 µs peak was a fragile compiler state, not a reliable optimization.

**Hand-written NEON is slower than the peak, but structurally resilient.** Our 1-accumulator NEON kernel initially *lost* to GCC's auto-vectorizer (24.96 µs vs 16.66 µs) because it lacked the 2x unroll. However, when the zero-allocation hot path broke the compiler's optimizations in Rung 5, the hand-written NEON kernel (20.77 µs) overtook the scalar code (24.30 µs). It won not because the NEON code improved, but because hand-written intrinsics are immune to the fragile memory and escape analysis traps that break GCC's auto-vectorizer. It provides a deterministic, unbreakable baseline.

---

## Remaining work

The ladder continues one rung at a time; the correctness gate is re-run after each.

- **Multiple accumulators / unrolling (Rung 6)** — manually unrolling the hand-written NEON kernel to use independent accumulators. This is required to reclaim the 16.66 µs peak structurally, without relying on fragile compiler states.
- **Roofline analysis (Phase 5)** — writing a custom C memory stream benchmark to place the naive and optimized kernels against the A76's *measured* compute and memory-bandwidth ceilings, proving mathematically whether the final kernel is compute- or memory-bound.
- **Narrative writeup & reproduction guide** — full problem -> approach -> results -> how to reproduce on a Pi 5.

---

## Repository layout

neonnet/
src/
matrix.{h,c}     # core Matrix type + naive/reordered/aligned ops
layers.{h,c}     # linear, relu, softmax, argmax
model.{h,c}      # weight loading + forward pass
main.c           # correctness harness (the Phase 2 gate)
bench/
benchmark.c      # timing harness with confidence intervals
results.md       # per-rung measured results + interpretation  <-- read this
python/
train.py         # train the MLP, export weights + test data
verify_export.py # pure-NumPy re-check of the export
export_format.md # binary weight-file specification
bench.sh           # reproducible governor + pin + perf wrapper
Makefile


---

## Methodology (the rules this project holds itself to)

1. **Correctness before speed.** The gate is sacred; no optimization ships until it passes.
2. **Measure every change.** A change with no measured number attached taught nothing.
3. **Fair comparisons.** Every version compiled at identical flags. The target to beat is the compiler's own auto-vectorization, not a strawman.
4. **Re-validate after every optimization.** SIMD remainder handling is the classic place output silently breaks.
5. **Commit at every milestone.** The incremental git history is part of the record.

---

## Reproducing

Requires a Raspberry Pi 5 (or another aarch64 Cortex-A76 target) with the toolchain above.

```bash
# 1. Train the model and export weights (Python side)
python3 python/train.py

# 2. Build the engine and benchmark binaries
make clean && make

# 3. Run the correctness gate
./neonnet          # expect STATUS: PASS

# 4. Run the full benchmark suite (governor + core pin + perf counters)
./bench.sh
