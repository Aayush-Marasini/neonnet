#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

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
	for (int i = 0; i < a->rows; i++){
	
		for(int j = 0; j < b->cols; j++){
			float sum = 0.0f;
			for(int k = 0; k < a->cols ; k++){
				sum += a->data[i * a->cols +k] * b-> data[ k * b->cols + j];	
			}
		result_mat.data[i * b-> cols + j] = sum;
		}
	}
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



