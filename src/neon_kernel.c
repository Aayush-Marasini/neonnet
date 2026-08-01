#include <arm_neon.h>
#include <string.h>
#include <assert.h>
#include "matrix.h" 
void neon_mat_mul_into(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict c) {

    // 1. Pointer Validity Checks
    assert(a != NULL && b != NULL && c != NULL);
    assert(a->data != NULL && b->data != NULL && c->data != NULL);
    
    // 2. Dimension Positivity Checks
    assert(a->rows > 0 && a->cols > 0 && b->rows > 0 && b->cols > 0);

    assert(a->cols == b->rows);
    assert(c->rows == a->rows && c->cols == b->cols);
    assert(c->data != a->data && c->data != b->data);

    memset(c->data, 0, c->rows * c->cols * sizeof(float));
    
    // Extract internal restricted pointers
    const float *restrict a_data = a->data;
    const float *restrict b_data = b->data;
    float *restrict c_data = c->data;

    int M = a->rows;
    int K = a->cols;
    int N = b->cols;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(a_data[i * K + k]);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                float32x4_t b_vec = vld1q_f32(&b_data[k * N + j]);
                float32x4_t c_vec = vld1q_f32(&c_data[i * N + j]);
                c_vec = vfmaq_f32(c_vec, b_vec, a_val);
                vst1q_f32(&c_data[i * N + j], c_vec);
            }
            // The restrict keyword heavily optimizes this scalar remainder loop
            for (; j < N ; j++) {
                c_data[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
            }
        }
    }
}
