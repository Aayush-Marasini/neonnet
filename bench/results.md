# NeonNet — Benchmark Results

## Baseline (v0.3-baseline)

### Conditions
- Device:    Raspberry Pi 5, Cortex-A76 (aarch64), official active fan cooler
- Compiler:  gcc -O3 -mcpu=cortex-a76
- Governor:  performance
- Core pin:  taskset -c 3
- Harness:   N = 10000 measured iterations, 100 warmup, input = 784 zeros (single image, held in place across runs)

### Numbers
- Mean latency:      175.50 µs   (95% CI ±0.03 µs)
- IPC:               1.49 instructions/cycle
- Cache references:  2,060,922,472
- Cache misses:      1,019,866,630  →  49.49% miss rate
- Hotspot:            mat_mul — 99.85% self time (perf report), linear — 0.04% self
- Correctness at this build: max logit diff 1.14e-5, 0 prediction mismatches (700 samples)

### Interpretation
The baseline implementation is heavily memory-bound, a hypothesis evidenced independently by both cache utilization and instruction throughput. The severe 49.49% cache miss rate demonstrates that the non-contiguous column stride in the inner loop is constantly missing the L1/L2 caches, forcing high-latency data fetches from main RAM. Concurrently, the pipeline is restricted to an IPC of just 1.49; because the Cortex-A76 is a wide, superscalar core capable of dispatching several instructions per clock cycle, this suppressed throughput indicates the execution units are continuously stalling while waiting for those memory fetches to resolve. While Phase 5's roofline plot will strictly prove this memory bottleneck by calculating the empirical arithmetic intensity, these metrics confirm that the engine is currently starved for data, meaning memory access patterns must be restructured before any instruction-level compute optimizations can take effect.

## Optimization Ladder (Phase 4 — filled per rung)

| Rung          | Latency µs (±CI) | Speedup vs naive | vs prev | Miss % | Max logit diff | Mismatches |
|---------------|------------------|------------------|---------|--------|-----------------|------------|
| naive (v0.3)  | 175.50 ±0.03     | 1.00×            | —       | 49.49% | 1.14e-5         | 0          |
| i-k-j reorder |                  |                  |         |        |                 |            |
| tiled         |                  |                  |         |        |                 |            |
| aligned       |                  |                  |         |        |                 |            |
| NEON          |                  |                  |         |        |                 |            |
