#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "model.h"
#include "layers.h"

Model model_load (const char *path){

    FILE *model_file = fopen(path, "rb");

    if (model_file == NULL){
        fprintf(stderr,"model.c: could not open the model binary file\n");
        Model empty_model = {0};
        return empty_model;
    }

    uint32_t check_magic;


    if (fread(&check_magic, sizeof(uint32_t), 1, model_file) != 1) {
        fprintf(stderr, "model.c: failed to read magic number\n");
        fclose(model_file);
        Model empty_model = {0};
        return empty_model;
        }


    if (check_magic != 0x4E454F4E){

        fprintf(stderr, "model.c: Invalid magic number \n");

        fclose (model_file);
        Model empty_model = {0};
        return empty_model;
    }




    uint32_t tensor_count;


    if (fread(&tensor_count, sizeof(uint32_t), 1, model_file) != 1) {
        fprintf(stderr, "model.c: failed to read tensor count\n");
        fclose(model_file);
        Model empty_model = {0};
        return empty_model;
    }

    if (tensor_count != 4){

        fprintf(stderr, "model.c: Invalid tensor count");

        fclose (model_file);
        Model empty_model = {0};
        return empty_model;
    }


    Model m = {0};

    Matrix *tensors[] = {&m.W1, &m.b1, &m.W2, &m.b2};

    uint32_t rows;
    uint32_t cols;

    for (int i = 0; i < 4; i++){

        if (fread(&rows, sizeof (uint32_t), 1, model_file) != 1){
            fprintf(stderr, "model.c: failed to read rows for tensor %d\n", i);
            model_free(&m); 
            fclose(model_file);
            Model empty_model = {0};
            return empty_model;
        }


        if (fread(&cols, sizeof (uint32_t), 1, model_file) != 1){

            fprintf(stderr, "model.c: failed to read cols for tensor %d\n", i);
            model_free(&m); 
            fclose(model_file);
            Model empty_model = {0};
            return empty_model;
        }
        
        *tensors[i] = mat_alloc(rows, cols);

        if (tensors[i]->data == NULL){

            fprintf(stderr, "model.c: could not malloc \n");
            fclose(model_file);
            Model empty_model = {0};
            return empty_model;
        }

        if (fread(tensors[i]->data, sizeof(float), rows * cols, model_file) != rows * cols){

            fprintf(stderr, "model.c: failed to read data for tensor %d\n", i);
            model_free(&m);
            fclose(model_file);
            Model empty_model = {0};
            return empty_model;
        }

        
    }

    fclose (model_file);
    return m;
}

Matrix model_forward(const Model *m, const Matrix *input){

	Matrix out1 = linear( input, &m->W1, &m->b1);

	if(out1.data == NULL){
	
		fprintf(stderr, "model.c: layer 1 (W1) failed\n");
		Matrix out = {0};
		return out;
	}

	relu(&out1);

	Matrix logits = linear (&out1, &m->W2, &m->b2);
	if(logits.data == NULL){	
		fprintf(stderr, "model.c:layer 2 (W2) failed \n");
		Matrix out = {0};
		mat_free(&out1);
		return out;
	}
	
	mat_free(&out1);
	return logits;

}

void model_free(Model *m){

    if (m == NULL){
        return;
    }

    Matrix *tensors[] = {&m->W1, &m->b1, &m->W2, &m->b2};

    for (int i = 0 ; i < 4; i++){
        mat_free(tensors[i]);
    }



}
