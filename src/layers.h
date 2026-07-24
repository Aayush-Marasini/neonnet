#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"

Matrix linear(const Matrix *input, const Matrix *W, const Matrix *b);
void relu(Matrix *m);
void softmax(Matrix *m);
int argmax(const Matrix *m);

#endif
