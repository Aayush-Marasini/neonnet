# NeonNet

**A from-scratch ML inference engine in pure C — hand-written kernels for the ARM Cortex-A76 (Raspberry Pi 5). No PyTorch, no BLAS, no OpenVINO at inference time.**

> **On the name.** "Zero-allocation, NEON-vectorized" is the *destination* this project is climbing toward, not where it is today. The correct engine, the measurement infrastructure, and the first three optimization rungs are done and measured; the hand-written NEON kernel and the zero-allocation hot path are later, scheduled rungs. This README states exactly what is and isn't done, and every number in it is a real measurement on the target board — not a projection.
>
> **Full per-rung results, conditions, and interpretation live in [`bench/results.md`](bench/results.md).**

---

## What this is

A from-scratch implementation of a neural network's **forward pass** (inference), built to answer one question:

> **How close to the hardware's theoretical peak can a hand-written inference kernel get on an actively-cooled ARM edge device, and which specific optimizations close the gap?**

The model under test is a small MLP trained on MNIST:

```
784 inputs  ->  Linear(784, 128)  ->  ReLU  ->  Linear(128, 10)  ->  10 logits
```

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
| 4 | Optimization ladder | In progress — 3 of 6 rungs done (reorder, tiling, alignment) |
| 5 | Roofline analysis | Not started |

### Correctness gate

The C forward pass is validated against PyTorch's own outputs on 700 held-out MNIST test samples. It passes when every logit agrees with PyTorch to within 1e-4 and no prediction disagrees:

```
Total Samples Processed: 700
Max Logit Difference:    1.14e-05
Prediction Mismatches:   0
STATUS: PASS
```

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
| 64-byte alignment | 16.66 +/-0.03 | **10.53x** | 1.30x | 6.01%* | 3.23 | Compiler re-vectorized (see below) |

<sub>*Aligned miss-rate/IPC from a slightly noisier profiling session; latency is clean. Full numbers and per-rung interpretation in [`bench/results.md`](bench/results.md).</sub>

### Rung 1 — loop reorder (the big one)

The naive baseline is heavily **memory-bound**: a 49.49% cache-miss rate from walking *down* columns of the weight matrix — a 512-byte stride that defeats the 64-byte cache line on every step — with instruction throughput suppressed to 1.49 IPC on a core capable of far more. The execution units stall waiting on RAM.

Reordering the loops to `i-k-j` makes the innermost loop walk memory with stride 1. The 8.13x speedup factors cleanly into two independent causes: cache misses collapse (49.49% -> 0.54%), lifting IPC from 1.49 to 3.90; **and** instruction count drops ~3x for the identical arithmetic — the fingerprint of GCC's auto-vectorizer, previously blocked by the column stride, now emitting NEON. Confirmed in the disassembly (`fmla v6.4s, v7.4s, v30.4s` — a 4-lane fused multiply-add) and the `-fopt-info-vec` report.

> **Consequence:** because the compiler auto-vectorizes the clean stride-1 loop, the hand-written NEON kernel (a later rung) will race **compiler NEON, not scalar code**. Matching or beating the compiler's own auto-vectorizer at identical flags is this project's honest headline comparison — not beating a strawman `-O0` build.

### Two results that contradicted the prediction

The point of this project is measuring rather than assuming — and two rungs came back the *opposite* of what was predicted. Both are documented as findings, not buried.

**Cache tiling made it *slower* (0.80x), not neutral.** The prediction was near-neutral: at N=1 the forward pass is a streaming vector-matrix product with each weight read exactly once, so cache blocking has no temporal reuse to capture. Instead of neutral, tiling *regressed* — 21.59 -> 27.09 us, with the miss rate rising to 3.08%. Mechanism: splitting one contiguous stride-1 row-scan into blocked sub-passes broke the pattern the hardware prefetcher was exploiting and added loop-bookkeeping overhead, with zero reuse to offset it. Cache blocking is a technique for M>1 (batched) workloads; this quantifies exactly why it backfires at M=1.

**Alignment gave a 23% speedup that was predicted to be zero.** The prediction was a no-op: plain `malloc` already returns 16-byte-aligned memory on aarch64, and a 16-byte NEON load on a 16-aligned address can never straddle a 64-byte cache line — so there should have been nothing to fix. It sped up anyway, 21.55 -> 16.66 us. Disassembly killed the obvious explanation ("dropped unaligned scaffolding"): the aligned `mat_mul` is *longer*, not shorter. The real cause is a **different vectorization strategy** — with alignment guaranteed, GCC emitted a 2x-unrolled inner loop with two independent `fmla` accumulator chains (two `dup` + two `fmla` per iteration), versus the single-accumulator kernel it chose without the guarantee. The dual accumulators hide FMA latency (the second multiply-add doesn't wait on the first), which is why cycles fell even though the loop body grew, and why global instruction count dropped (2.03B -> 1.30B) from far fewer loop iterations. The precise cost-model trigger is compiler-internal and wasn't chased further.

> **Consequence:** GCC already applies the multiple-accumulator optimization that was planned as a *later hand-written rung*. Like the NEON rung, that rung is now "race the compiler's version," not "add it from scratch."

---

## Remaining work

The ladder continues one rung at a time; the correctness gate is re-run after each.

- **Hand-written NEON kernel** — an explicit `arm_neon.h` inner kernel with a scalar tail loop for non-multiple-of-4 remainders, benchmarked against the compiler's auto-vectorized build at identical flags.
- **Zero-allocation hot path** — removing the per-call result-matrix allocation (the project's namesake goal) by reusing pre-allocated buffers.
- **Multiple accumulators / unrolling** — hand-written, benchmarked against the dual-accumulator kernel GCC *already* emits (see above).
- **Roofline analysis (Phase 5)** — placing the naive and optimized kernels against the A76's *measured* compute and memory-bandwidth ceilings, to determine with evidence whether each kernel is compute- or memory-bound. The baseline profile (49% miss, 1.49 IPC) already predicts the naive kernel is memory-bound — the roofline is the proof.
- **Narrative writeup & reproduction guide** — full problem -> approach -> results -> how to reproduce on a Pi 5.

---

## Repository layout

```
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
```

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
```

*This README grows into the full narrative writeup as the remaining phases land. For the detailed per-rung measurements and analysis, see [`bench/results.md`](bench/results.md).*
