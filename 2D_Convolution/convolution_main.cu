#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include "gputimer.h"
#include "gpuerrors.h"
#include "convolution.h"

// Define the tile dimensions
const int tilex = 32;
const int tiley = 32;
const int K = 10;
// ===========================> Functions Prototype <===============================
double calc_mse (float* data1, float* data2, int size);
void convolution2DCPU(const float *f, const float *g, float *result, int n, bool flag);
void gpuKernel(const float *f, const float *g, float *result, int n, int tilex, int tiley, double* gpu_kernel_time);
// =================================================================================

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);	
	
    // Set the size of the signals (assumed to be square)
    int m = atoi(argv[1]);
	int n = (1 << m);
	const int result_size = n + n - 1;
	
    // Allocate memory for signals on host
    float *h_f = (float*)malloc(n*n * sizeof(float));
    float *h_g = (float*)malloc(n*n * sizeof(float));
    float *h_result_gpu = (float*)malloc(result_size*result_size * sizeof(float));
    float *h_result_cpu = (float*)malloc(result_size*result_size * sizeof(float));

    // Initialize signals with random values
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < n * n; ++i) {
        h_f[i] = (float) (rand() % 17 - 8);  
    }
    for (int i = 0; i < n * n; ++i) {
        h_g[i] = (float) (rand() % 17 - 8);
    }

	if (n < 256)
		convolution2DCPU(h_f, h_g, h_result_cpu, n, 1); // Flag is 1 when n < 256. Otherwise it is 0.
	else
		convolution2DCPU(h_f, h_g, h_result_cpu, n, 0); 

	// GPU calculations
	double gpu_kernel_time = 0.0;
	clock_t t1 = clock(); 
	gpuKernel (h_f, h_g, h_result_gpu, n, tilex, tiley, &gpu_kernel_time);
    clock_t t2 = clock(); 

	
	// check correctness of GPU calculations against CPU
	double mse = 0.0;
	if(n < 256)
		mse += calc_mse( h_result_cpu, h_result_gpu, result_size * result_size );
	else
		mse += calc_mse( h_result_cpu, h_result_gpu, K * result_size );


	printf("m=%d n=%d GPU=%g ms GPU-Kernel=%g ms mse=%g\n",
	m, n, (t2-t1)/1000.0, gpu_kernel_time, mse);
		
	// free allocated memory for later use
	free(h_f);
	free(h_g);
	free(h_result_gpu);
	free(h_result_cpu);

    return 0;
}

double calc_mse (float* data1, float* data2, int size) {
	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		double e = data1[i]-data2[i];
		e = e * e;
		mse += e;
	}
	return mse;
}


// CPU function for 2D convolution
void convolution2DCPU(const float *f, const float *g, float *result, int n, bool flag) {
    // Calculate the size of the result matrix
    int resultSize = n + n - 1;
	if (flag == 1){
		for (int row = 0; row < resultSize; ++row) {
			for (int col = 0; col < resultSize; ++col) {
				float sum = 0.0f;

				for (int i = 0; i < n; ++i) {
					for (int j = 0; j < n; ++j) {
						int fRow = row - i;
						int fCol = col - j;

						// Check boundaries of the input signal f
						if (fRow >= 0 && fRow < n && fCol >= 0 && fCol < n) {
							sum += f[fRow * n + fCol] * g[i * n + j];
						}
					}
				}

				result[row * resultSize + col] = sum;
			}
		}
	}
	else {
		for (int row = 0; row < K; ++row) {
			for (int col = 0; col < resultSize; ++col) {
				float sum = 0.0f;

				for (int i = 0; i < n; ++i) {
					for (int j = 0; j < n; ++j) {
						int fRow = row - i;
						int fCol = col - j;

						// Check boundaries of the input signal f
						if (fRow >= 0 && fRow < n && fCol >= 0 && fCol < n) {
							sum += f[fRow * n + fCol] * g[i * n + j];
						}
					}
				}

				result[row * resultSize + col] = sum;
			}
		}		
		
	}
}

// GPU kernel function for 2D convolution
// This function is defined in convolution.cu
void gpuKernel(const float *f, const float *g, float *result, int n, int bx_size, int by_size, double* gpu_kernel_time) {

    // Allocate memory for signals on device
	int result_size = n + n - 1;
    float *d_f, *d_g, *d_result_gpu;
    HANDLE_ERROR(cudaMalloc(&d_f, n * n  * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_g, n * n  * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_result_gpu, result_size * result_size * sizeof(float)));

    // Copy signals from host to device
    HANDLE_ERROR(cudaMemcpy(d_f, f, n * n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_g, g, n * n * sizeof(float), cudaMemcpyHostToDevice));	
	
    dim3 blockSize(bx_size, by_size);
    dim3 gridSize((result_size  + blockSize.x - 1) / blockSize.x, (result_size + blockSize.y - 1) / blockSize.y);
	GpuTimer timer;
    timer.Start();
	kernelFunc<<< gridSize, blockSize >>>(d_f, d_g, d_result_gpu, n); //modify this function in convolution.cu
	timer.Stop();
	*gpu_kernel_time = timer.Elapsed();
	
    // Copy the result back from device to host
    HANDLE_ERROR(cudaMemcpy(result, d_result_gpu, result_size * result_size * sizeof(float), cudaMemcpyDeviceToHost));	
	
    HANDLE_ERROR(cudaFree(d_f));
    HANDLE_ERROR(cudaFree(d_g));
    HANDLE_ERROR(cudaFree(d_result_gpu));	
	

}