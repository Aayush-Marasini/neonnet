
## Baseline (v0.3-baseline)

### Conditions
- Device:    Raspberry Pi 5, Cortex-A76 (aarch64), official active fan cooler
- Compiler:  gcc -O3 -mcpu=cortex-a76
- Governor:  performance
- Core pin:  taskset -c 3
- Harness:   N = 10000 measured iterations, 100 warmup, input = 784 zeros (single image, held in place across runs)

### Numbers
- Mean latency:      175.50 Âµs   (95% CI Â±0.03 Âµs)
- IPC:               1.49 instructions/cycle
- Cache references:  2,060,922,472
- Cache misses:      1,019,866,630  â†’  49.49% miss rate
- Hotspot:            mat_mul â€” 99.85% self time (perf report), linear â€” 0.04% self
- Correctness at this build: max logit diff 1.14e-5, 0 prediction mismatches (700 samples)

### Interpretation
The baseline implementation is heavily memory-bound, a hypothesis evidenced independently by both cache utilization and instruction throughput. The severe 49.49% cache miss rate demonstrates that the non-contiguous column stride in the inner loop is constantly missing the L1/L2 caches, forcing high-latency data fetches from main RAM. Concurrently, the pipeline is restricted to an IPC of just 1.49; because the Cortex-A76 is a wide, superscalar core capable of dispatching several instructions per clock cycle, this suppressed throughput indicates the execution units are continuously stalling while waiting for those memory fetches to resolve. While Phase 5's roofline plot will strictly prove this memory bottleneck by calculating the empirical arithmetic intensity, these metrics confirm that the engine is currently starved for data, meaning memory access patterns must be restructured before any instruction-level compute optimizations can take effect.


i-k-j
 The reorder made the access stride-1. The 8x jump was because of two factors. the jump in IPC from 1.49 to 3.90. (2    .6 x) higher. and also the instructions fell from 6.19 B to 2.034 B (3x fewer). 3 * 2.6 = 7.8. That explains some of th    e speedup. Fewer cycles per instruction  , pipeline is no longer stalling on cache misses. And fewer instructions for t    he identical arithmetic means each instruction is now doing more work. -fopt-info-vec and fmla were used to get these n    umbers.

tiled
The implementation of block tiling resulted in a performance regression, increasing mean latency to 27.09 Âµs and elevating the cache miss rate to 3.08% compared to the un-tiled i-k-j reorder. This degradation confirms that for single-sample edge inference (batch size M=1), loop blocking provides no computational benefit. Because the input vector is a single row, there is no spatial or temporal reuse of the weight matrix; each parameter is fetched exactly once per forward pass. It also  failed to capture any data reuse and instead actively disrupted the Cortex-A76 hardware prefetcher, which was previously optimizing the continuous linear memory access pattern of the i-k-j layout. The introduced loop mechanics acted  as instruction overhead, proving that cache tiling is counterproductive when the foundational math lacks dimensional reuse. 

aligned
Aligning to 64 bytes produced an unpredicted 23% speedup (21.55 µs $\rightarrow$ 16.66 µs) resulting from the compiler safely emitting a 2x unrolled inner loop. Disassembly proves this unrolling does not use independent accumulators to hide FMA latency via instruction-level parallelism (ILP), as successive fmla instructions write sequentially to the exact same destination register (v16). Instead, the massive reduction in global instructions (2.03B $\rightarrow$ 1.30B) and cycles is driven primarily by the elimination of intermediate memory round-trips for the accumulator. Unrolling shortens the serial dependency chain for a given output block across two $k$-steps from fmla $\rightarrow$ str $\rightarrow$ ldr $\rightarrow$ fmla to a direct fmla $\rightarrow$ fmla sequence, which drops cycles without parallelizing the math, while simultaneously stripping out redundant pointer increments and bounds checks. This lack of ILP and adherence to sequential execution is corroborated by two metrics: the max logit difference remained exactly 1.144409e-05, implying an unaltered summation tree, and the IPC fell from 3.90 to 3.23—a direction consistent with the absence of independent execution chains, though confounded by the deletion of cheap, high-throughput loop overhead instructions.NOTE: Because the compiler's auto-vectorizer failed to use independent accumulators, the hand-written multi-accumulator implementation (Rung 6) remains an uncompleted optimization. However, because our innermost $j$-loop steps are independent for a fixed $k$, the hardware may already find sufficient work to fill FMA latency slots, meaning manual multi-accumulator unrolling is not yet attempted, and not yet shown to be needed.

NEON
3.75 ins / cycle
rather than reproducing GCC's pre-unroll shape, we actively regressed past it to measure 24.96 µs. Stated plainly, our hand-written NEON implementation lost to GCC's very first optimization on the ladder, coming in slower than the naive i-k-j scalar reorder from Rung 1 (21.55 µs). As revealed by objdump, the gap is driven by iteration count and a single, un-hidden FMA dependency chain—not a memory-resident accumulator bottleneck, which the i-k-j layout forces identically on both kernels. Our inner vector loop executes exactly one fmla per body, paying the full add/cmp/b.ne loop overhead every four columns, whereas GCC's aligned scalar kernel 2×-unrolled the loop to pack two fmla instructions per body. On the correctness front, the logit drift held exactly at 1.144409e-05 as predicted, because the per-column summation order remains perfectly preserved across the SIMD lanes, and memory remains Valgrind-clean at N=2, 10, and 128. The generalized lesson is that intrinsics are not automatically fast: by forcing a strict single-accumulator block, we revoked the compiler's scheduling freedom without replacing it with better scheduling of our own
## Optimization Ladder (Phase 4 â€” filled per rung)

| Rung          | Latency Âµs (Â±CI) | Speedup vs naive | vs prev | Miss % | Max logit diff | Mismatches |
|---------------|------------------|------------------|---------|--------|-----------------|------------|
| naive (v0.3)  | 175.50 Â±0.03     | 1.00Ã—          | â€”   | 49.49% | 1.14e-5         | 0          | i-k-j reorder | 21.59 Â0.01      | 8.13Ã—          | 8.13Ã— | 0.54%  | 1.144409e-05    | 0          |
| tiled         | 27.09 Â±0.1       | 6.48Ã—          | 0.80Ã— | 3.08%  | 1.144409e-05    | 0          |
| aligned       | 16.66 Â±0.3       | 10.54Ã—         | 1.30Ã— |6.0%    | 1.144409e-0 5   | 0          || NEON (hand, 1-acc) | 24.96 ±0.04 | 7.03x | 0.67x |0.42 | — | — | — | 1.144409e-05 | 0 |

