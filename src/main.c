#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "main.h"
#include "model.h"
int main(void){

	srand(42);
	
	printf("=== Starting Matrix Library Tests ===\n\n");

	Matrix a = mat_alloc(2,3);
	Matrix b = mat_alloc(3,2);

	if (a.data == NULL || b.data == NULL){
	
		fprintf(stderr, "Main: Matrix allocation failed during test setup.\n");
		mat_free(&a);
		mat_free(&b);
		return 1;
	}
	
	mat_fill_random(&a);
	mat_fill_random(&b);

	printf("Matrix A \n");
	mat_print(&a);
	printf("\n");

	printf("Matrix B \n");
	mat_print(&b);
	printf("\n");

	Matrix c = mat_mul(&a, &b);

	if (c.data!= NULL){
	
		printf("Result Matrix C \n");
		mat_print(&c);
	} else {
	
		fprintf(stderr, "Main: Matrix multiplication failed.\n");;
	}

	mat_free(&a);
	mat_free(&b);
	mat_free(&c);

	printf("\n===Test Complete ===\n:");


	printf("Loading model ...\n");
	Model m = model_load("artifacts/weights.bin");

	if (m.W1.data == NULL){
		fprintf(stderr, "main: model load failed \n");
		return 1;
	}

	printf("W1: %d x %d\n", m.W1.rows, m.W1.cols);	
	printf("b1: %d x %d\n", m.b1.rows, m.b1.cols);	
	printf("W2: %d x %d\n", m.W2.rows, m.W2.cols);	
	printf("b2: %d x %d\n", m.b2.rows, m.b2.cols);

	model_free(&m);

	return 0;
}
