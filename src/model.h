#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

typedef struct 
{
    Matrix W1, b1, W2, b2;
} Model;

Model model_load (const char *path);
void model_free(Model *m);

#endif