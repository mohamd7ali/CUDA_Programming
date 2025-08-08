//Do NOT MODIFY THIS FILE

#ifndef BMM_H
#define BMM_H

#define mem2d(data,q,y,x)   data[((y)<<(q))+(x)]
// data is a 2-dimensional (2^q)*(2^q) matrix implemented as 1-dimensional array
// data[y][x] == data[ y * (2^q) + x ]
// y * (2^q) == y << q

#include "helper.h"
dim3 getDimGrid(const int m, const int n);
dim3 getDimBlock(const int m, const int n);
__global__ void kernelFunc(uint8_t* ad, uint8_t* bd, uint8_t* cd, const int m, const int n);
void gpuKernel(const uint8_t* const a, const uint8_t* const b, uint8_t* c, const int m, const int n);

#endif
//Do NOT MODIFY THIS FILE

