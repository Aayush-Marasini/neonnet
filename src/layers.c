#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<stdalign.h>
#include "layers.h"
extern int g_use_neon;

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

void linear_into(const Matrix *input, const Matrix *W, const Matrix *b, Matrix *out) {
    // CALLER OWNS THIS MATRIX: Caller must supply a pre-allocated buffer.
    assert(input->cols == W->rows && b->cols == W->cols);

    if (g_use_neon) {
        neon_mat_mul_into(input, W, out);
    } else {
        mat_mul_into(input, W, out);
    }

    for (int i = 0; i < b->cols; i++) {
        out->data[i] += b->data[i];
    }
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


