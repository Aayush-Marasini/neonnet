#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "main.h"
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


	return 0;
}
