#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cstdlib"
#include "ctime"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "gputimer.h"
#include "gpuerrors.h"
#include "htt.h"

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
double calc_mse (float* data1, float* data2, int size);
void cpuKernel(const float* const a,float* c, float* temp, const int m, const int n);
void gpuKernels(const float* const a, float* c, const int m, const int n, double* gpu_kernel_time);
// =================================================================================

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	
	// get parameter from command line to build Matrix dimension
	// check for 10<=m<=13, because m>=14 do not fit in the memory of our GPU, i.e., 1GB.
	int m = atoi(argv[1]);
    int n = (1 << m);
	
	// allocate memory in CPU for calculation
	float* a;
	float* c_serial;
	float* c;
	float* temp;

	a        = (float*)malloc(n*n * sizeof(float));
	c_serial = (float*)malloc(n*n * sizeof(float));
	c        = (float*)malloc(n*n * sizeof(float));
	temp     = (float*)malloc(n*n * sizeof(float));

	// fill a, b matrices with random values between 20.0f and 30.0f
	srand(0); 
	fill(a, n*n);

	// CPU calculations
	cpuKernel (a,c_serial,temp, m, n);
		
	// GPU calculations
	double gpu_kernel_time = 0.0;
	clock_t t1 = clock(); 
	gpuKernels (a,c, m, n, &gpu_kernel_time);
    clock_t t2 = clock(); 
		
	// check correctness of GPU calculations against CPU
	double mse = 0.0;
	mse += calc_mse( c_serial, c, n*n );


	printf("m=%d n=%d GPU=%g ms GPU-Kernel=%g ms mse=%g\n",
	m, n, (t2-t1)/1000.0, gpu_kernel_time, mse);
	

	// free allocated memory for later use
	free(a);
	free(c_serial);
	free(c);
	free(temp);

	return 0;
}

// Function to fill the provided data array with random float values in a specific range (20.0 to 30.0)
void fill(float* data, int size) {

    for (int i=0; i<size; ++i){
        int randomInt = rand();
		float randomFloat = 20.0f +(randomInt / (RAND_MAX + 1.0f)) * 10.0f;
		data[i] = randomFloat;
	}
}

// Function to calculate the Mean Squared Error (MSE) between two float arrays of the same size
double calc_mse (float* data1, float* data2, int size) {
	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		double e = data1[i]-data2[i];
		e = e * e;
		mse += e;
	}
	return mse;
}
//-----------------------------------------------------------------------------
void cpuKernel(const float* const a,float* c, float* temp, const int m, const int n) { 
	// Copy the input matrix 'a' (old temperature) to the output matrix 'c' (new temperature)
	for(int i = 0; i < n ; i++){
		for(int j = 0; j < n ; j++){
			c[i*n+j] = a[i*n+j];
		}
	}
	
	// Perform the computation using nested loops for 'count' iterations (10 times)

	for(int count = 0; count <10; count++){
		// Iterate through each cell of the matrix
		for(int i = 0; i < n ; i++){
		
			for(int j = 0; j < n ; j++){

				float newt = c[i*n+j];
				int rt,lt,cr,up,dn;

				// Compute indices of neighboring cells
				rt = i*n+(j + 1);	//right
				lt = i*n+(j - 1);	//left
				cr = i*n+j;		//center
				up = (i - 1)*n+j;	//up
				dn = (i + 1)*n+j;	//down

				 // Check for boundary conditions and adjust neighboring indices if needed	
				if(i==0)	up = cr;
				if(i==n-1)	dn = cr;
				if(j==0)	lt = cr;
				if(j==n-1)	rt = cr;

				// Perform computation based on neighboring cell values
				newt += k_const * ( c[rt] + c[lt] + c[up] + c[dn] - 4 * c[cr] );
				temp[i*n+j] = newt;
			
			}
		}

		// Update the 'c' matrix (new temperature) with the values from the 'temp' matrix (temporary temperature)
		for(int i = 0; i < n ; i++){
			for(int j = 0; j < n ; j++){
				c[i*n+j] = temp[i*n+j];
			}
		}
	}


}

//-----------------------------------------------------------------------------
void gpuKernels(const float* const a, float* c, const int m, const int n, double* gpu_kernel_time) {

	// Declare pointers for GPU memory
	float* ad;	// Pointer for input matrix on GPU
	float* cd;	// Pointer for output matrix on GPU

	// Allocate memory on GPU for input and output matrices
    HANDLE_ERROR(cudaMalloc((void**)&ad, n * n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&cd, n * n * sizeof(float)));

	// Copy input matrix 'a' (old temperature) from CPU to GPU memory
    HANDLE_ERROR(cudaMemcpy(ad, a, n * n * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cd, a, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Create a timer to measure kernel execution time
	GpuTimer timer;
	double timersum=0.0;

	// Perform GPU computations for 'count' iterations (10 times)
	for(int count = 0 ; count <10 ; count++){

		timer.Start();	
		gpuKernel(ad,cd,n,m); 
		timer.Stop();	

		// Copy the output matrix 'cd' from GPU to CPU memory
		cudaMemcpy(c, cd, n * n * sizeof(float), cudaMemcpyDeviceToHost);

		// Copy the updated output matrix 'c' back to GPU memory for the next iteration
		cudaMemcpy(ad, c, n * n * sizeof(float), cudaMemcpyHostToDevice);

		timersum += timer.Elapsed();	// Accumulate the elapsed kernel execution time
    }

	*gpu_kernel_time = timersum;
    
	// Copy the final output matrix 'cd' from GPU to CPU memory
	HANDLE_ERROR(cudaMemcpy(c, cd, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(ad));
    HANDLE_ERROR(cudaFree(cd));
}