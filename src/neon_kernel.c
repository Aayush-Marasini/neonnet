#include <arm_neon.h>
#include <string.h>
#include "matrix.h" 

Matrix neon_mat_mul(const Matrix* a, const Matrix* b) {
    // 1. Allocate the result matrix (applies the 64-byte padding!)
    Matrix c = mat_alloc(a->rows, b->cols);
    memset(c.data, 0, c.rows * c.cols * sizeof(float));
    int M = a->rows;
    int K = a->cols;
    int N = b->cols;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // Broadcast the scalar A value to all 4 lanes
            float32x4_t a_val = vdupq_n_f32(a->data[i * K + k]);

            int j = 0;
            // 2. FIXED: Added <= N to prevent infinite out-of-bounds loop
            for (; j + 4 <= N; j += 4) {
                // load 4 columns of B
                float32x4_t b_vec = vld1q_f32(&b->data[k * N + j]);
                
                // load 4 current accumulation values from C
                float32x4_t c_vec = vld1q_f32(&c.data[i * N + j]);

                // fused multiply add:
                c_vec = vfmaq_f32(c_vec, b_vec, a_val);

                // store updated values back to C
                vst1q_f32(&c.data[i * N + j], c_vec);
            }

            for (; j < N ; j++) {
                // Relies on compiler -ffp-contract=fast for FMA hardware matching
                c.data[i * N + j] += a->data[i * K + k] * b->data[k * N + j];
            }
        }
    } // End k loop
    // 3. FIXED: return c is now outside all loops
    return c;
}
