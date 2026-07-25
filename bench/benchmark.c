#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "layers.h"
#include "model.h"

#define WARMUP_ITS  100
#define MEASURE_ITS  10000

uint64_t elapsed_ns(struct timespec start, struct timespec end){

	return (uint64_t)(end.tv_sec - start.tv_sec) * 1000000000ULL + (uint64_t)(end.tv_nsec - start.tv_nsec);
}


int main (void){

	float input_data[784] = {0};
	Matrix input = {1, 784, input_data};
        printf("Loading model ...\n");	
	Model m = model_load ( "artifacts/weights.bin");
	if ( m.W1.data == NULL){
		
		fprintf(stderr,"benchmark.c:failed to load model\n");
		return -1;
	}


	for (int i = 0; i < WARMUP_ITS; i++){
	
		Matrix out = model_forward(&m , &input);

		mat_free(&out);
	}

	uint64_t t [MEASURE_ITS] = {0};
	
	struct timespec start_time;
	struct timespec end_time;

	for (int i = 0; i < MEASURE_ITS; i++){
	
		if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        		fprintf(stderr,"benchmark.c:clock_gettime start failed\n");
        		return 1;
    		}
		Matrix out = model_forward(&m, &input);

		if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0){
		
			fprintf(stderr,"benchmark.c : clock_gettime end failed\n");
			return 1;
		}

		t[i] = elapsed_ns ( start_time, end_time);
		mat_free(&out);
	}

	double running_sum = 0.0;
        	for (int i = 0; i < MEASURE_ITS; i++){
                	running_sum += t[i];
        	}
        
        double mean_ns = running_sum / MEASURE_ITS;
        double variance_sum = 0.0;
        
        for (int i = 0; i < MEASURE_ITS; i++){
                double diff = t[i] - mean_ns;
                variance_sum += (diff * diff); 
        }
        
        double sd_ns = sqrt(variance_sum / MEASURE_ITS);
        
        // Calculate the 95% Confidence Interval
        double sem_ns = sd_ns / sqrt(MEASURE_ITS);
        double ci_95_ns = 1.96 * sem_ns;

        double mean_us = mean_ns / 1000.0;
        double ci_95_us = ci_95_ns / 1000.0;

        printf("======================================\n");
        printf("NeonNet Benchmark (%d iterations)\n", MEASURE_ITS);
        printf("Mean Latency:   %.2f us\n", mean_us);
        printf("95%% CI:         ±%.2f us\n", ci_95_us);
        printf("======================================\n");

        model_free(&m);

        return 0;
}
