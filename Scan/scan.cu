//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

/*
	Mohammad Ali Etemadi Naeen - 402200348
	HW3 - Scan 
	Sharif University of Technology
*/

#include "scan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define gdx gridDim.x
#define gdy gridDim.y
#define gdz gridDim.z


// Multiplication in Galois Fields - global func
__device__ uint8_t gf_multiply(uint8_t x, uint8_t y, uint8_t* alpha_to, uint8_t* index_of)
{
    if (x == 0 || y == 0)
    {
        return 0;
    }
    else
    {
        return alpha_to[((index_of[x]) + (index_of[y])) % 255];
    }
}

// Multiplication of values in Galois Fields - host/cpu func  
uint8_t gf_multiply_host(uint8_t x, uint8_t y, uint8_t* alpha_to, uint8_t* index_of)
{
    if (x == 0 || y == 0)
    {
        return 0;
    }
    else
    {
        return alpha_to[(uint32_t(index_of[x]) + uint32_t(index_of[y])) % 255];
    }
}

// Matrix multiplication by a vector in Galois Fields - global func
__device__ void gf_matrix_vector_multiply(uint8_t* mat, uint8_t* vec, uint8_t* result, uint8_t* alpha_to, uint8_t* index_of)
{
    for (int i = 0; i < 4; i++)
    {
        result[i] = 0;
        for (int j = 0; j < 4; j++)
        {
            result[i] ^= gf_multiply(mat[i * 4 + j], vec[j], alpha_to, index_of);
        }
    }
}

// Matrix multiplication by a vector in Galois Fields - host/cpu func 
void gf_matrix_multiply_host(const uint8_t* const mat1, const uint8_t* const mat2, uint8_t* result, int dim, uint8_t* alpha_to, uint8_t* index_of)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            result[i * dim + j] = 0;
            for (int k = 0; k < dim; k++)
            {
                result[i * dim + j] ^= gf_multiply_host(mat1[i * dim + k], mat2[k * dim + j], alpha_to, index_of);
            }
        }
    }
}

// First kernel func
__global__ void kernelFunc1(uint8_t* ad, uint8_t* matrixPowNd, const int m, const int n, const int step, uint8_t* alpha_to, uint8_t* index_of, uint8_t stage)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int section_size = step * 2;
    int start_idx = thread_idx * section_size;
    int change_start_idx = section_size / 2 - 1 + start_idx;
    int change_end_idx = start_idx + (section_size - 1);
    uint8_t temp[4] = { 0 };

    gf_matrix_vector_multiply(&matrixPowNd[16 * stage], &ad[change_start_idx * 4], temp, alpha_to, index_of);

    for (int i = 0; i < 4; i++)
    {
        ad[change_end_idx * 4 + i] = (temp[i] ^ ad[change_end_idx * 4 + i]);
    }
}

// Second kernel func
__global__ void kernelFunc2(uint8_t* ad, uint8_t* matrixPowNd, const int m, const int n, const int step, uint8_t* alpha_to, uint8_t* index_of, uint8_t stage)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int section_size = step * 2;
    int start_idx = thread_idx * section_size;
    int change_start_idx = section_size / 2 - 1 + start_idx;
    int change_end_idx = start_idx + (section_size - 1);
    uint8_t temp1[4] = { 0 };
    uint8_t temp2[4] = { 0 };

    for (int i = 0; i < 4; i++)
    {
        temp2[i] = ad[change_end_idx * 4 + i];
    }

    gf_matrix_vector_multiply(&matrixPowNd[16 * stage], &ad[change_end_idx * 4], temp1, alpha_to, index_of);

    for (int i = 0; i < 4; i++)
    {
        ad[change_end_idx * 4 + i] = (temp1[i] ^ ad[change_start_idx * 4 + i]);
    }

    for (int i = 0; i < 4; i++)
    {
        ad[change_start_idx * 4 + i] = temp2[i];
    }
}

// Kernel func to reset final array values
__global__ void resetFinalArray(uint8_t* ad, const int n)
{
    for (int i = 0; i < 4; i++)
    {
        ad[4 * n - 4 + i] = 0;
    }
}


void gpuKernel(const uint8_t* const a, const uint8_t* const matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
    uint8_t* ad;
    uint8_t* matrixPowNd;
    uint8_t* alpha_tod;
    uint8_t* index_ofd;

    HANDLE_ERROR(cudaMalloc((void**)&ad, 4 * n * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&matrixPowNd, m * 4 * 4 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&alpha_tod, 256 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&index_ofd, 256 * sizeof(uint8_t)));

    uint8_t* matrixPowN = (uint8_t*)malloc(m * 4 * 4 * sizeof(uint8_t));

    for (int i = 0; i < 16; i++)
    {
        matrixPowN[i] = matrix[i];
    }

    gf_matrix_multiply_host(matrix, matrix, &matrixPowN[16], 4, alpha_to, index_of);

    for (int f = 2; f < m; f++)
    {
        gf_matrix_multiply_host(&matrixPowN[(f - 1) * 16], &matrixPowN[(f - 1) * 16], &matrixPowN[f * 16], 4, alpha_to, index_of);
    }

    HANDLE_ERROR(cudaMemcpy(ad, a, 4 * n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(matrixPowNd, matrixPowN, m * 4 * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(alpha_tod, alpha_to, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(index_ofd, index_of, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));

    uint8_t lastElement[4];
    int step = 1;

    for (int i = 1; i <= m; i++)
    {
        int numThreads = n / (step * 2);
        int numBlocks = 1;
        if (numThreads > 1024)
        {
            numBlocks = numThreads / 1024;
            numThreads = 1024;
        }
        dim3 gridDim(numBlocks, 1, 1);
        dim3 blockDim(numThreads, 1, 1);
        kernelFunc1 <<< gridDim, blockDim >>> (ad, matrixPowNd, m, n, step, alpha_tod, index_ofd, i - 1);
        cudaDeviceSynchronize();
        step *= 2;
    }

    cudaMemcpy(lastElement, &ad[4 * n - 4], 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    resetFinalArray <<<1, 1>>> (ad, n);
    cudaDeviceSynchronize();

    step /= 2;
    for (int i = 1, j = m - 1; i <= m; i++, j--)
    {
        int numThreads = n / (step * 2);
        int numBlocks = 1;
        if (numThreads > 1024)
        {
            numBlocks = numThreads / 1024;
            numThreads = 1024;
        }
        dim3 gridDim(numBlocks, 1, 1);
        dim3 blockDim(numThreads, 1, 1);
        kernelFunc2 <<< gridDim, blockDim >>> (ad, matrixPowNd, m, n, step, alpha_tod, index_ofd, j);
        cudaDeviceSynchronize();
        step /= 2;
    }

    cudaMemcpy(c, &ad[4], 4 * n * sizeof(uint8_t) - 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++)
    {
        c[4 * n - 4 + i] = lastElement[i];
    }

    HANDLE_ERROR(cudaFree(ad));
    free(matrixPowN);
    HANDLE_ERROR(cudaFree(matrixPowNd));
    HANDLE_ERROR(cudaFree(index_ofd));
    HANDLE_ERROR(cudaFree(alpha_tod));
}


