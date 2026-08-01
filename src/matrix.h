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

void mat_mul_into(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c);
void neon_mat_mul_into(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c);

#endif 
