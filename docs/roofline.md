# Phase 5: Empirical Roofline Analysis

## Hardware Baselines (Raspberry Pi 5 / Cortex-A76)
- **Derived Clock Speed:** 2.40 GHz (16.94B cycles / 7.06s)
- **Peak Compute (FP32):** 38.4 GFLOP/s (2 pipes × 4 lanes × 2 ops/cycle × 2.40 GHz)
- **L2 Bandwidth Ceiling:** 44.55 GB/s (Measured via 4-accumulator 256KB footprint benchmark)

## Rung 5 (Zero-Alloc NEON) Achieved Performance
- **Execution Time:** 20.77 µs / pass
- **Achieved Compute:** 9.79 GFLOP/s (25.5% of Compute Roof)
- **Achieved L2 Bandwidth:** 20.12 GB/s (45.1% of L2 Roof)
- **Limiter Diagnosis:** NOT Bandwidth Bound. The kernel is bottlenecked by instruction issue capacity and serial `fmla` dependency chains.

## Rung 6 Prediction (Loop Unrolling & Multi-Accumulator)
**Prediction:** Massive improvement. Latency will drop from ~20.77 µs down to ~10–12 µs.
**Mechanism:** 
By unrolling the loop and accumulating into multiple independent NEON registers concurrently, we break the serial FADD/FMLA dependency chain that is currently capping throughput. We measured a 2.3× bandwidth multiplier in the streaming benchmark simply by utilizing four independent accumulators. Additionally, processing multiple elements per loop iteration dilutes the non-math instruction overhead (pointer increments, branches, and loop counters), freeing up front-end issue slots for pure math.

