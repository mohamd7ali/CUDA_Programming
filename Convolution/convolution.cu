/*
	Mohammad Ali Etemadi Naeen - 402200348
	HW4 - Convolution
	Sharif University of Technology
*/

#include "convolution.h"

#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

__global__ void kernelFunc(const float* f, const float* g, float* result, int n)
{
    // Insert your code here...
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output size calculation
    const int resultSize = n + n - 1; 

    if (row < resultSize && col < resultSize) {

        float value = 0.0f;
        
        // External loop for first sigma
        for (int i = 0; i < n; ++i) {

            int idx1 = row - i;

            // Boundary check for first sigma
            if (idx1 < 0){
                break;
            }

            if (idx1 >= n){
                continue;
            }
            // Internal loop for second sigma
            for (int j = 0; j < n; ++j) {

                int idx2 = col - j;

                // Boundary check for second sigma
                if (idx2 >= 0 && idx2 < n){
                    value += f[idx1 * n + idx2] * g[i * n + j];
                }
            }
        }
        
        // result
        result[row * resultSize + col] = value; 
    }
}
