#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

#define L2_TARGET_BYTES (256 * 1024) // 256 KB footprint
#define ITERS 100000 
#define TRIALS 10
#define WARMUP 2

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

volatile float global_sink = 0.0f;

int main() {
    // 64-byte alignment
    float *data = (float*)aligned_alloc(64, L2_TARGET_BYTES);
    if (!data) return 1;
    
    size_t count = L2_TARGET_BYTES / sizeof(float);
    for (size_t i = 0; i < count; i++) {
        data[i] = 1.0f;
    }

    double times[TRIALS];

    // Warmup + Measurement
    for (int t = -WARMUP; t < TRIALS; t++) {
        // Rung 6 Preview: Four independent accumulators
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        float32x4_t vsum2 = vdupq_n_f32(0.0f);
        float32x4_t vsum3 = vdupq_n_f32(0.0f);

        double start = get_time();
        for (size_t iter = 0; iter < ITERS; iter++) {
            for (size_t i = 0; i < count; i += 16) {
                float32x4_t v0 = vld1q_f32(&data[i]);
                float32x4_t v1 = vld1q_f32(&data[i + 4]);
                float32x4_t v2 = vld1q_f32(&data[i + 8]);
                float32x4_t v3 = vld1q_f32(&data[i + 12]);
                
                vsum0 = vaddq_f32(vsum0, v0);
                vsum1 = vaddq_f32(vsum1, v1);
                vsum2 = vaddq_f32(vsum2, v2);
                vsum3 = vaddq_f32(vsum3, v3);
            }
        }
        double end = get_time();
        
        if (t >= 0) {
            times[t] = end - start;
        }

        // Collapse independent accumulators to prevent dead-code elimination
        float sum_array[4];
        vsum0 = vaddq_f32(vsum0, vsum1);
        vsum2 = vaddq_f32(vsum2, vsum3);
        vsum0 = vaddq_f32(vsum0, vsum2);
        vst1q_f32(sum_array, vsum0);
        global_sink = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    }

    // Statistics
    double mean_time = 0.0;
    for (int i = 0; i < TRIALS; i++) mean_time += times[i];
    mean_time /= TRIALS;

    double sq_diff = 0.0;
    for (int i = 0; i < TRIALS; i++) {
        double diff = times[i] - mean_time;
        sq_diff += diff * diff;
    }
    double stddev = sqrt(sq_diff / TRIALS);
    double ci95 = 1.96 * (stddev / sqrt(TRIALS));

    double total_bytes = (double)L2_TARGET_BYTES * ITERS;
    double bw_gbps = (total_bytes / 1e9) / mean_time;

    printf("L2 Streaming Benchmark (Footprint: 256 KB, 4 Accums)\n");
    printf("Mean Time: %.5f s +/- %.5f s\n", mean_time, ci95);
    printf("Achieved L2 Bandwidth: %.2f GB/s\n", bw_gbps);

    free(data);
    return 0;
}
