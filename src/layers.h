#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"

void  linear_into(const Matrix *input, const Matrix *W, const Matrix *b,  Matrix *out);
void relu(Matrix *m);
void softmax(Matrix *m);
int argmax(const Matrix *m);

#endif
