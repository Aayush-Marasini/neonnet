#include<stdio.h>
#include<math.h>
#include "layers.h"

void relu(Matrix *m){

	if (m == NULL || m->data == NULL){
	
		fprintf(stderr,"layers.c:nothing in the relu matrix\n");
		return;
	}
	
	int size = m->rows * m->cols;
	
	
	for (int i = 0 ; i < size ; i++){
		
		if (m->data[i] < 0.0f){
		
			m->data[i] = 0.0f;
		}			
	}

}

int argmax(const Matrix*m){

	if (m == NULL || m->data == NULL){
	
		fprintf(stderr,"layers.c:nothing in the argmax matrix\n");
		return -1;
	}
	int size = m->rows * m->cols;

	if (size <=0){
	
		fprintf(stderr, "layers.c: argmax on empty matrix\n");
		return -1;
	}
	float largest_value = m->data[0];
	int largest_index = 0;


	for (int i = 1; i < size ; i++){
	
		if (m->data[i] > largest_value){
		
			largest_value = m->data[i];
			largest_index = i;
		}
	}
	
	return largest_index;

}

Matrix linear(const Matrix *input, const Matrix *W, const Matrix *b){
	// CALLER OWNS THIS MATRIX: must mat_free to prevent leaks
	if ( input == NULL || input->data == NULL || W == NULL || W->data == NULL || b == NULL || b->data == NULL){
	
		fprintf(stderr,"layers.c: could not read in the linear function\n");
		Matrix emptymat = {0,0,NULL};
		return emptymat;
	}

	if (input->cols != W->rows || b->cols != W->cols){
	
		fprintf(stderr,"layers.c: matrix dimension mismatch\n");
		Matrix emptymat = {0,0,NULL};
		return emptymat;
	}
	
	Matrix result = mat_mul (input, W);

	if (result.data == NULL){
		fprintf(stderr,"layers.c: multiplied matrix data null\n");
		return result;	
	}

	for ( int i = 0; i < b->cols; i++){
		
		float added = result.data[i] + b->data[i];
		result.data[i] = added;
	}
	return result;
}


void softmax(Matrix *m){

	if (m == NULL || m->data == NULL){
	
		fprintf(stderr, "layers.c: nothing in the softmax matrix\n");
		return;
	}

	int size = m->rows * m->cols;
    	if (size <= 0) {
        	fprintf(stderr, "layers.c: softmax on empty matrix\n");
        	return;
    	}

	float max_val = m->data[0];

    	for (int i = 1; i < size; i++) {

        	if (m->data[i] > max_val) {

			max_val = m->data[i];
        }
    }

    
	float sum = 0.0f;
   	for (int i = 0; i < size; i++) {
        	
		m->data[i] = expf(m->data[i] - max_val);
        	sum += m->data[i];
    	}

    
    
    for (int i = 0; i < size; i++) {
        m->data[i] /= sum;
    }
}


