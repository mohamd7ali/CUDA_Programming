//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

/*
	Mohammad Ali Etemadi Naeen - 402200348
	HW2 - Block Matrix Multiplication 
	Sharif University of Technology
*/

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 32
#define TILEY 16

// you may define other parameters here!
#define TILE_S 128
#define T_ROW TILE_S/TILEY
#define T_COL TILE_S/TILEX
#define K  (T_ROW >= T_COL ? T_ROW : T_COL)


dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
	// write your GPU kernel function here
	// use shared_memory
	__shared__ float As[TILEY][TILE_S];
    __shared__ float Bs[TILE_S][TILEX];

    // define variables
    // row & column
	int i = by * TILEY + ty;;
	int j = bx * TILEX + tx;
	// temp to save the results for output
	float temp_c = 0;

	// for calculating the bmm
	for(int k = 0; k < n/TILE_S; k++)
	{
		if (T_ROW >= T_COL)
		{
			for(int r = 0; r < K; r++)
			{
				if (T_COL > r)
				{
					As[ty][tx + TILEX*r] = ad[n*i + TILE_S*k + tx + TILEX*r];
				}
				Bs[ty + TILEY*r][tx] = bd[(TILE_S*k + ty + TILEY*r)*n + j];
			}
		}
		else
		{
			for(int r = 0; r < K; r++)
			{
				if (T_ROW > r)
				{
					Bs[ty + TILEY * r][tx] = bd[(TILE_S * k + ty + TILEY * r) * n + j];
				}
				As[ty][tx + TILEX * r] = ad[n * i + TILE_S * k + tx + TILEX * r];
			}
		}
		__syncthreads();

		for(int TILE_idx = 0; TILE_idx < TILE_S; TILE_idx++){
			temp_c += As[ty][TILE_idx] * Bs[TILE_idx][tx];
		}
		__syncthreads();
	}
	cd[i * n + j] = temp_c;
}
