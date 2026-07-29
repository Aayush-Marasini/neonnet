#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matrix.h"
#define BK 64
#define BN 64

Matrix mat_alloc(int rows, int cols){

	Matrix mat = {rows , cols, NULL};
	mat.data = malloc( rows * cols * sizeof(float));
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

Matrix mat_mul(const Matrix *a, const Matrix *b){
	
	if (a->cols != b->rows){
		fprintf(stderr, " mat_mul: dimension mismatch (a->cols=%d, b->rows=%d)\n", a->cols, b->rows);
		return (Matrix){.rows = 0, .cols = 0, .data = NULL};
	}
	Matrix result_mat = mat_alloc (a->rows, b->cols);
	if (result_mat.data == NULL){
	
		fprintf(stderr, "mat_mul: allocation failed for result matrix\n");
		return (Matrix){.rows = 0, .cols =0, .data= NULL};
	}

    memset(result_mat.data, 0, (size_t)result_mat.rows * result_mat.cols * sizeof(float));

for (int i = 0; i < a->rows; i++) {
    
    for (int kk = 0; kk < a->cols; kk += BK) {
        int k_end = (kk + BK < a->cols) ? (kk + BK) : a->cols;
        
        for (int jj = 0; jj < b->cols; jj += BN) {
            int j_end = (jj + BN < b->cols) ? (jj + BN) : b->cols;
            
            for (int k = kk; k < k_end; k++) {
                
                float a_val = a->data[i * a->cols + k];
                
                for (int j = jj; j < j_end; j++) {
                    
                    result_mat.data[i * b->cols + j] += a_val * b->data[k * b->cols + j];
                    
                }
            }
        }
    }
}        

return result_mat;
}




void mat_print (const Matrix *m){
	for (int i = 0; i < m->rows; i++){
		for (int j = 0 ; j < m->cols ; j++){
					printf ("%f\t", m->data[i* m->cols +j]);
				}
	}
	
}



