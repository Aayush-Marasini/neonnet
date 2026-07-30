#ifndef MATRIX_H
#define MATRIX_H
typedef struct{
	int rows;
	int cols;
	float *data;
}Matrix;

Matrix mat_alloc(int rows, int cols);
void mat_free(Matrix *m);
void mat_fill_random (Matrix *m);
Matrix mat_mul (const Matrix *a, const Matrix *b);
void mat_print(const Matrix *m);
Matrix neon_mat_mul(const Matrix* a, const Matrix* b);
#endif 
