//ONLY MODIFY THIS FILE

/*
	Mohammad Ali Etemadi Naeen - 402200348
	HW1 - Matrix Multiplication 
	Sharif University of Technology
*/

#include "mm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may change the following values
#define TILEX 512
#define TILEY 2

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}

//-----------------------------------------------------------------------------
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m,const int n) {
	// write your GPU kernel function here
	// note that maximum # of threads per block is 1024

	// int i = (bx * TILEX) + tx;
	// int j = (by * TILEY) + ty;

	// float temp_c = 0.0;

	// for (int k=0; k<n; k++)

    // define variables
    // row & column
    int i = by * TILEY + ty;
    int j = bx * TILEX + tx;
    // temp to save the results for output
    float temp_c = 0;

    // for calculating the mm
    for (int k = 0; k < n; k++)
    {
        temp_c += ad[i * n + k] * bd[k * n + j];
    }
    cd[i * n + j] = temp_c;
}

