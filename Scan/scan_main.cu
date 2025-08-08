#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"
#include "scan.h"
#include "finite_field.h"
#include "helper.h"

// ===========================> Functions Prototype <===============================
void fill(uint8_t* data, int size);
int64_t calc_num_error(uint8_t* data1, uint8_t* data2, int size);
void cpuKernel(uint8_t* a, uint8_t* matrix, uint8_t* c, int m, int n, FiniteField* finiteField);
void gpuKernel(const uint8_t* const a, const uint8_t* const b, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of);
// =================================================================================
using namespace std;
#include <iostream>
int main(int argc, char** argv)
{
	FiniteField finiteField;
	

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	
	// get parameter from command line to build Matrix dimension
	// check for 10<=m<=13, because m>=14 do not fit in the memory of our GPU, i.e., 1GB.
	int m = atoi(argv[1]);
	if (m>26)
	{
		cout << "Input most not be more than 26!" << endl;
		return 0;
	}
    int64_t n = (1 << m);
	
	// allocate memory in CPU for calculation
	uint8_t* a;
	uint8_t* matrix;
	uint8_t* c_serial;
	uint8_t* c;
	a = (uint8_t*)malloc(4 * n * sizeof(uint8_t));
	matrix = (uint8_t*)malloc(4 * 4 * sizeof(uint8_t));
	c_serial = (uint8_t*)malloc(4 * n * sizeof(uint8_t));
	c = (uint8_t*)malloc(4 * n * sizeof(uint8_t));
	
	// fill a, b matrices with random values between -16.0f and 16.0f
	srand(0);
	fill(a, 4 * n);
	fill(matrix, 4*4);

	// time measurement for CPU calculations
	clock_t t0 = clock();
	cpuKernel(a, matrix, c_serial, m, n, &finiteField);
	clock_t t1 = clock();

	// time measurement for GPU calculations
	clock_t t2 = clock(); 
	gpuKernel(a, matrix, c, m, n, finiteField.alpha_to, finiteField.index_of);
    clock_t t3 = clock(); 
		
	// check correctness of GPU calculations against CPU
	int64_t num_error = 0.0;
	num_error += calc_num_error( c_serial, c, n*4 );

	printf("m=%d\t\tn=%ld\t\tCPU=%g ms\t\tGPU=%g ms\t\tnum_error=%ld\n",
	m, n, double(t1-t0)/1000.0, double(t3-t2)/1000.0, num_error);
		
	// free allocated memory for later use
	free(a);
	free(matrix);
	free(c_serial);
	free(c);
   
	return 0;
}
//-----------------------------------------------------------------------------
void fill(uint8_t* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = (uint8_t) (rand() % 256);
}

int64_t calc_num_error(uint8_t* data1, uint8_t* data2, int size){
	int64_t num_error = 0;
	for (int64_t i=0; i<size; i++)
	{
		if (data1[i]!=data2[i])
		{
			num_error += 1;
		}
	}
	return num_error;
}
//-----------------------------------------------------------------------------
// CPU kernel function to perform the matrix-vector multiplication
void matrix_to_vector_multiply(uint8_t* a, uint8_t* v, uint8_t* res, int dim, FiniteField* finiteField)
{
	for (int i=0; i<dim; i++)
	{
		res[i] = 0;
		for (int j=0; j<dim; j++)
		{
			res[i] ^= finiteField->multiply(a[i*dim+j], v[j], finiteField->alpha_to, finiteField->index_of);
		}
	}
}
//-----------------------------------------------------------------------------
// CPU kernel function to perform the matrix multiplication
// It takes two matrices 'a' and 'b', and outputs the result matrix 'c
void matrix_to_matrix_multiply(uint8_t* a, uint8_t* b, uint8_t* c, int dim, FiniteField* finiteField)
{
	for (int i=0; i<dim; i++)
	{
		for (int j=0; j<dim; j++)
		{
			c[i*dim+j] = 0;
			for (int k=0; k<dim; k++)
			{
				c[i*dim+j] ^= finiteField->multiply(a[i*dim+k], b[k*dim+j], finiteField->alpha_to, finiteField->index_of);
			}
		}
	}
}
//-----------------------------------------------------------------------------
// CPU kernel function to perform the scan operation
// It takes the input vector 'a', the transformation matrix 'matrix', and outputs the result vector 'c'
void cpuKernel(uint8_t* a, uint8_t* matrix, uint8_t* c, int m, int n, FiniteField* finiteField)
{


	for (int i=0; i<n; i++)
	{
		if (i==0)
		{
			for (int j=0; j<4; j++)
			{
				c[i*4+j] = a[i*4+j];
			}
		}
		else
		{
			matrix_to_vector_multiply(matrix, &(c[i*4-4]), &(c[i*4]), 4, finiteField);


			for (int j=0; j<4; j++)
			{
				c[i*4+j] ^= a[i*4+j];
			}

		}
	}


}