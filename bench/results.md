| Rung          | Latency µs (±CI) | Speedup vs naive | vs prev | Miss % | Max logit diff | Mismatches |
|---------------|------------------|------------------|---------|--------|-----------------|------------|
| naive (v0.3)  | 175.50 ±0.03     | 1.00×          | —   | 49.49% | 1.14e-5         | 0          |
| i-k-j reorder | 21.59 �0.01      | 8.13×          | 8.13× | 0.54%  | 1.144409e-05    |          0 |
| tiled         |                  |                  |         |        |                 |            |
| aligned       |                  |                  |         |        |                 |            |
| NEON          |                  |                  |         |        |                 |            |
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


i-k-j
 The reorder made the access stride-1. The 8x jump was because of two factors. the jump in IPC from 1.49 to 3.90. (2    .6 x) higher. and also the instructions fell from 6.19 B to 2.034 B (3x fewer). 3 * 2.6 = 7.8. That explains some of th    e speedup. Fewer cycles per instruction  , pipeline is no longer stalling on cache misses. And fewer instructions for t    he identical arithmetic means each instruction is now doing more work. -fopt-info-vec and fmla were used to get these n    umbers.

tiled
The implementation of block tiling resulted in a performance regression, increasing mean latency to 27.09 µs and elevating the cache miss rate to 3.08% compared to the un-tiled i-k-j reorder. This degradation confirms that for single-sample edge inference (batch size M=1), loop blocking provides no computational benefit. Because the input vector is a single row, there is no spatial or temporal reuse of the weight matrix; each parameter is fetched exactly once per forward pass. It also  failed to capture any data reuse and instead actively disrupted the Cortex-A76 hardware prefetcher, which was previously optimizing the continuous linear memory access pattern of the i-k-j layout. The introduced loop mechanics acted  as instruction overhead, proving that cache tiling is counterproductive when the foundational math lacks dimensional reuse. 

aligned
Aligning to 64 bytes produced an unprecedinted 23% speedup (21.55->16.66us).Disassembly showed the aligned "mat_mul" is longer, real cause of this is a different vectorization strtegy. With guaranteed alignment, GCC emitted a 2x unrolled inner loop with two indepedent fmla accumulator chains, versus rung 1's single accumulator kernel. The dual accumulators hide FMA latency and halve loop-overhead instructions, which explains the cycle drop and the global instruction count fall.(2.03B -> 1.30B) despite a heavier more bloated loop body. The precise cost model trigger is compiler internal and not further investigated.NOTE: This means, GCC already applies multi accumulator optimization that i planned to add by hand. 

## Optimization Ladder (Phase 4 — filled per rung)

| Rung          | Latency µs (±CI) | Speedup vs naive | vs prev | Miss % | Max logit diff | Mismatches |
|---------------|------------------|------------------|---------|--------|-----------------|------------|
| naive (v0.3)  | 175.50 ±0.03     | 1.00×          | —   | 49.49% | 1.14e-5         | 0          |
| i-k-j reorder | 21.59 �0.01      | 8.13×          | 8.13× | 0.54%  | 1.144409e-05    | 0          |
| tiled         | 27.09 ±0.1       | 6.48×          | 0.80× | 3.08%  | 1.144409e-05    | 0          |
| aligned       | 16.66 ±0.3       | 10.54×         | 1.30× |6.0%    | 1.144409e-0 5   | 0          |
| NEON          |                  |                  |         |        |                 |            |
