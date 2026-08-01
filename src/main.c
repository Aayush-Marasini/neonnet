#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include "main.h"
#include "model.h"
int g_use_neon = 0;
int main(int argc, char **argv){
    if (argc > 1 && strcmp(argv[1], "neon") == 0) {
        g_use_neon = 1;
        printf("Main: NEON kernel activated.\n");
    } else {
        printf("Main: Baseline kernel activated.\n");
    }
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


	FILE *file = fopen( "artifacts/test_samples.bin", "rb");
	if (file == NULL){
	
		fprintf(stderr, "main.c:Error opeaning file \n");
		return 1;
	}

	uint32_t magic; 
	
	if(fread(&magic, sizeof(uint32_t), 1, file) != 1){
	
		fprintf(stderr,"main.c: failed to read magic number\n");
		fclose(file);
		return 1;
	}

	if (magic != 0x44415441){
	
		fprintf(stderr, "main.c: magic does not match\n");
		fclose(file);
		return 1;
	}


	uint32_t sample_count;
	if(fread(&sample_count, sizeof(uint32_t), 1, file) != 1){
	
		fprintf(stderr,"main.c: could not retrive total number of samples\n");
		fclose(file);
		return 1;
	}
	

	Matrix image_matrix = mat_alloc(1,784);
	Matrix logits_matrix = mat_alloc(1,10);
	float max_diff = 0.0f;
	int mismatches = 0;
   
    alignas(64) float my_logits_data[16] = {0};
    Matrix my_logits = {1, 10, my_logits_data};
	
    
    for ( uint32_t i = 0; i < sample_count; i++){
			
		uint32_t label;
		if(fread(&label, sizeof(uint32_t), 1, file) !=1){
				
			fprintf(stderr,"main.c:could not read from file\n");
			fclose(file);
			mat_free(&image_matrix);
			mat_free(&logits_matrix);
			return 1;
		}
		if(fread(image_matrix.data, sizeof(float), 784, file)!=784){
		
			fprintf(stderr,"main.c:could not read image data from file\n");
			fclose(file);
			mat_free(&image_matrix);
			mat_free(&logits_matrix);
			return 1;
		}
		if(fread(logits_matrix.data, sizeof(float), 10 , file)!=10){
		
			fprintf(stderr,"main.c:could not read logits data from file\n");
			fclose(file);
			mat_free(&image_matrix);
			mat_free(&logits_matrix);
			return 1;
		}

        model_forward(&m, &image_matrix, &my_logits);

        
        for (int j = 0; j < 10; j++){
                // Read from my_logits instead of out
                float diff = fabsf(my_logits.data[j] - logits_matrix.data[j]);
                if(diff > max_diff){
                  max_diff = diff;
                }
        }

        int my_pred = 0;
        int pt_pred = 0;

        // Read from my_logits instead of out
        float my_max = my_logits.data[0];
        float pt_max = logits_matrix.data[0];

        for (int j = 1; j < 10; j++){
                // Read from my_logits instead of out
                if (my_logits.data[j] > my_max){
                  my_max = my_logits.data[j];
                  my_pred = j;
                }
                if (logits_matrix.data[j] > pt_max){
                  pt_max = logits_matrix.data[j];
                  pt_pred = j;
                }
        }
	
        if (my_pred != pt_pred){
		
		mismatches += 1;
	}
	}

printf("\n=== Test Results===\n");
printf("Total Samples Processed: %u\n", sample_count);
printf("Max Logit Difference:    %e\n", max_diff);
printf("Prediction Mismatches:   %d\n", mismatches);
if (max_diff < 1e-4 && mismatches == 0) {
	printf("\nSTATUS: PASS. C model perfectly matches PyTorch.\n");
	} 
else {
	printf("\nSTATUS: FAIL. Differences are too high.\n");
        }

fclose(file);
mat_free(&image_matrix);
mat_free(&logits_matrix);
model_free(&m);
return 0;
}
