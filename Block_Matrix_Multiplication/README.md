# Block Matrix Multiplication (CUDA)

This project implements efficient block matrix multiplication using CUDA for GPU acceleration. It is designed as a homework assignment for the Parallel Programming course at Sharif University of Technology.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Compilation](#compilation)
  - [Running](#running)
  - [Output Example](#output-example)
- [Customization](#customization)
- [Notes](#notes)
- [Author](#author)
## Overview

Matrix multiplication is a fundamental operation in scientific computing and machine learning. This project demonstrates how to leverage CUDA's parallelism and shared memory to accelerate the multiplication of large square matrices.

- **Block Matrix Multiplication:** The algorithm divides matrices into sub-blocks (tiles) and uses shared memory to minimize global memory accesses, improving performance.
- **Configurable Tile Size:** The tile size and block dimensions are configurable for experimentation and optimization.
- **Performance Measurement:** The code measures and reports the GPU kernel execution time and compares the result with a CPU implementation for correctness.

## File Structure

- `bmm.cu` — Main CUDA kernel and helper functions for block matrix multiplication (modifiable).
- `bmm.h` — Function prototypes and macros (do not modify).
- `bmm_main.cu` — Main program: handles memory allocation, timing, correctness checking, and launching kernels (do not modify).
- `gputimer.h` — Utility for measuring GPU kernel execution time.
- `gpuerrors.h` — Error handling macros for CUDA API calls.
- `README.md` — Project documentation.

## How It Works

1. **Matrix Initialization:** Random square matrices `A` and `B` are generated on the host.
2. **CPU Reference Calculation:** For small matrices, the CPU computes the reference result for correctness checking.
3. **GPU Computation:** The matrices are copied to the device, and the CUDA kernel performs block matrix multiplication using shared memory tiling.
4. **Result Validation:** The output is copied back to the host and compared with the CPU result using Mean Squared Error (MSE).
5. **Performance Reporting:** The program prints the matrix size, GPU execution time, kernel time, and MSE.

## Usage

### Prerequisites

- CUDA-capable GPU (with at least 1GB memory)
- NVIDIA CUDA Toolkit
- C++ compiler (e.g., `nvcc`)

### Compilation

Open a terminal in the project directory and run:

```
nvcc -o bmm bmm_main.cu bmm.cu
```

### Running

The program expects a single argument: `m`, where the matrix size is `n = 2^m` (e.g., `m=10` for `1024x1024` matrices).

```
./bmm 10
```

**Note:** For `m >= 14`, the matrices may not fit in 1GB GPU memory.

### Output Example

```
Device Name: NVIDIA GeForce GTX 1050
m=10 n=1024 GPU=123.45 ms GPU-Kernel=120.67 ms mse=0
```

- `GPU` is the total GPU computation time.
- `GPU-Kernel` is the kernel execution time.
- `mse` is the mean squared error between CPU and GPU results.

## Customization

- **Tiling Parameters:** You can adjust `TILEX`, `TILEY`, and `TILE_S` in `bmm.cu` to experiment with different block and tile sizes for performance tuning.
- **Kernel Logic:** The main CUDA kernel is in `bmm.cu` and can be optimized further for your GPU architecture.

## Notes

- Only modify `bmm.cu` for kernel development; other files should remain unchanged.
- The project is designed for square matrices of size 2^m × 2^m.
- For large matrices (m > 13), ensure your GPU has sufficient memory.

## Author

Mohammad Ali Etemadi Naeen  
