// /*
// 	Mohammad Ali Etemadi Naeen - 402200348
// 	HW5 - Heat Transformation with Texture Memory (htt)
// 	Sharif University of Technology
// */

#include "htt.h"

#define tx threadIdx.x
#define bx blockIdx.x
#define ty threadIdx.y
#define by blockIdx.y


// Define a texture memory
texture<float, 2, cudaReadModeElementType> tex;

__global__ void kernelFunc(float* newtemperature, const unsigned int N)
{   
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    // Check if the thread is within bounds
    if (x < N && y < N) {
        // texture memory
        float center = tex2D(tex, x, y);
        float left = tex2D(tex, x - 1, y);
        float right = tex2D(tex, x + 1, y);
        float up = tex2D(tex, x, y - 1);
        float down = tex2D(tex, x, y + 1);

        // new temperature value
        float newt = center + k_const * (left + right + up + down - 4 * center);

        // Store the new temperature value
        newtemperature[y * N + x] = newt;
    }
}

void gpuKernel(float* ad, float* cd, const unsigned int N, const unsigned int M)
{
    // Define block size and grid size
    dim3 blockSize(16, 16); 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Create CUDA array
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuArray, &channelDesc, N, N);

    // Bind the texture to the input data on GPU memory
    cudaMemcpyToArray(cuArray, 0, 0, ad, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(tex, cuArray, channelDesc);

    // Launch the CUDA kernel function
    kernelFunc<<<gridSize, blockSize>>>(cd, N);

    // Unbind the texture after kernel execution
    cudaUnbindTexture(tex);
    cudaFreeArray(cuArray);
}




// // Define a texture memory
// texture<float, 2, cudaReadModeElementType> tex;

// __global__ void kernelFunc(float* newtemperature, const unsigned int N)
// {   
//     int x = bx * blockDim.x + tx;
//     int y = by * blockDim.y + ty;

//     // Check if the thread is within bounds
//     if (x < N && y < N) {
//         // texture memory
//         float center = tex2D(tex, x, y);
//         float left = tex2D(tex, x - 1, y);
//         float right = tex2D(tex, x + 1, y);
//         float up = tex2D(tex, x, y - 1);
//         float down = tex2D(tex, x, y + 1);

//         // new temperature value
//         float newt = center + k_const * (left + right + up + down - 4 * center);

//         // Store the new temperature value
//         newtemperature[y * N + x] = newt;
//     }
// }

// void gpuKernel(float* ad, float* cd, const unsigned int N, const unsigned int M)
// {
//     // Define block size and grid size
//     dim3 blockSize(16, 16); 
//     dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

//     // Create CUDA array
//     cudaArray* cuArray;
//     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
//     cudaMallocArray(&cuArray, &channelDesc, N, N);

//     // Bind the texture to the input data on GPU memory
//     cudaMemcpyToArray(cuArray, 0, 0, ad, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
//     cudaBindTextureToArray(tex, cuArray, channelDesc);

//     for (int i = 0; i < 10; ++i) {
//         kernelFunc<<<gridSize, blockSize>>>(cd, N);
//         cudaDeviceSynchronize();  // Ensure kernel completion
//     }
//     // Unbind the texture after kernel execution
//     cudaUnbindTexture(tex);
//     cudaFreeArray(cuArray);
// }


