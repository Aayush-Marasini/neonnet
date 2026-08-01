#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include "matrix.h"

Matrix mat_alloc(int rows, int cols){

	Matrix mat = {rows , cols, NULL};
    size_t raw_size = rows * cols * sizeof(float);
    size_t padded_size = (raw_size + 63) & ~(size_t)63;

    mat.data = (float *) aligned_alloc (64, padded_size);
	if (mat.data == NULL){
		fprintf(stderr, "Allocation failure\n");
		return (Matrix){.rows = 0, .cols = 0 , .data = NULL};
	}
    return mat;

}

void mat_free(Matrix *m){
	if (m != NULL){
	free(m->data);
	m->data = NULL;
	m->rows = 0;
	m->cols = 0;
	}

}

void mat_fill_random(Matrix *m){
	
	for ( int i = 0 ; i < m->rows ; i++){
		for (int j = 0 ; j < m->cols ; j++){
			m->data[i* m->cols + j] = (float)rand()/ RAND_MAX;
		}
	}
	
}

void mat_mul_into(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c) {
    assert(a != NULL && b != NULL && c != NULL);
    assert(a->data != NULL && b->data != NULL && c->data != NULL);
    assert(a->rows > 0 && a->cols > 0 && b->rows > 0 && b->cols > 0);
    assert(a->cols == b->rows);
    assert(c->rows == a->rows && c->cols == b->cols);
    assert(c->data != a->data && c->data != b->data);

    size_t raw_res_size = c->rows * c->cols * sizeof(float);
    size_t padded_res_size = (raw_res_size + 63) & ~(size_t)63;
    memset(c->data, 0, padded_res_size);

    const float *restrict a_data = a->data;
    const float *restrict b_data = __builtin_assume_aligned(b->data, 64);
    float *restrict c_data = __builtin_assume_aligned(c->data, 64);

    for (int i = 0 ; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            float a_val = a_data[i * a->cols + k];
            for (int j = 0; j < b->cols; j++) {
                c_data[i * b->cols + j] += a_val * b_data[k * b->cols + j];
            }
        }
    }
}

Matrix mat_mul(const Matrix *a, const Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, " mat_mul: dimension mismatch\n");
        return (Matrix){.rows = 0, .cols = 0, .data = NULL};
    }
    
    Matrix result_mat = mat_alloc(a->rows, b->cols);    
    // Note: The sentinel contract ({0,0,NULL}) survives here in the wrapper,
    // but dies in the zero-allocation _into functions which use asserts instead.
    if (result_mat.data == NULL) {
        fprintf(stderr, "mat_mul: allocation failed\n");
        return (Matrix){.rows = 0, .cols = 0, .data = NULL};
    }
    
    mat_mul_into(a, b, &result_mat);
    return result_mat;
}	

void mat_print (const Matrix *m){
	for (int i = 0; i < m->rows; i++){
		for (int j = 0 ; j < m->cols ; j++){
					printf ("%f\t", m->data[i* m->cols +j]);
				}
		printf("\n");
	}
	
}



