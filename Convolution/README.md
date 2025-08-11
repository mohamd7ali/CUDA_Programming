# CUDA 2D Convolution

This project implements efficient 2D convolution of two square matrices using both CPU and GPU (CUDA) approaches. It is designed as a homework assignment for the Parallel Programming course at Sharif University of Technology.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Output Example](#output-example)
- [Customization](#customization)
- [Implementation Details](#implementation-details)
- [Notes](#notes)
- [Author](#author)

## Overview

2D convolution is a fundamental operation in image processing, signal processing, and deep learning. This project demonstrates how to accelerate 2D convolution using CUDA, leveraging GPU parallelism for significant speedup on large matrices.

- **Full 2D Convolution:** Computes the full convolution (not just valid or same) of two square matrices.
- **CPU and GPU Implementations:** Includes both a reference CPU implementation and a CUDA-accelerated GPU implementation for comparison.
- **Performance Measurement:** Measures and reports execution time for both CPU and GPU computations.
- **Correctness Verification:** Compares the GPU result with the CPU result using Mean Squared Error (MSE).

## Features

- **2D Convolution:** Computes the full 2D convolution of two square matrices.
- **CPU and GPU Implementations:** Includes both a reference CPU implementation and a CUDA-accelerated GPU implementation.
- **Performance Measurement:** Measures and reports the execution time for both CPU and GPU computations.
- **Correctness Verification:** Compares the GPU result with the CPU result using Mean Squared Error (MSE).

## File Structure

- `convolution_main.cu` — Main program: handles memory allocation, initialization, timing, correctness checking, and launching kernels.
- `convolution.cu` — Contains the CUDA kernel for 2D convolution (modifiable).
- `convolution.h` — Kernel function declaration (do not modify).
- `gputimer.h` — Utility for timing CUDA kernel execution.
- `gpuerrors.h` — Error handling macros for CUDA API calls.
- `README.md` — Project documentation.

## How It Works

1. **Matrix Initialization:** Random square matrices `f` and `g` are generated on the host.
2. **CPU Reference Calculation:** For small matrices, the CPU computes the reference result for correctness checking.
3. **GPU Computation:** The matrices are copied to the device, and the CUDA kernel performs 2D convolution.
4. **Result Validation:** The output is copied back to the host and compared with the CPU result using Mean Squared Error (MSE).
5. **Performance Reporting:** The program prints the matrix size, GPU execution time, kernel time, and MSE.

## Build Instructions

### Prerequisites

- CUDA-capable GPU (with at least 1GB memory)
- NVIDIA CUDA Toolkit
- C++ compiler (e.g., `nvcc`)

### Compilation

Open a terminal in the project directory and run:

```sh
nvcc -O2 convolution_main.cu convolution.cu -o convolution 
```

## Usage

Run the executable with a single argument specifying the matrix size as a power of two:

```sh
./convolution <m>
```

Where `<m>` is an integer such that the matrix size `n = 2^m`. For example, to run with `n = 256`:

```sh
./convolution 8
```

## Output Example

```
Device Name: NVIDIA GeForce GTX 1050
m=8 n=256 GPU=12.34 ms GPU-Kernel=10.56 ms mse=0
```

- `GPU`: Total GPU computation time (including memory transfers).
- `GPU-Kernel`: Time spent in the CUDA kernel.
- `mse`: Mean squared error between CPU and GPU results.

## Customization

- **Block and Tile Size:** You can adjust `tilex` and `tiley` in `convolution_main.cu` to experiment with different block sizes for performance tuning.
- **Kernel Logic:** The main CUDA kernel is in `convolution.cu` and can be optimized further for your GPU architecture.

## Implementation Details

- **Convolution Algorithm:** For each output element, the kernel computes the sum of products over all valid overlaps of the input matrices.
- **CUDA Kernel:** Each thread computes one output element. The grid and block dimensions are chosen to cover the entire output matrix.
- **Memory Management:** Host and device memory are allocated and freed appropriately. Data is transferred between host and device as needed.
- **Error Handling:** All CUDA API calls are checked for errors.

## Notes

- For large matrices (`n >= 256`), only the first `K` rows are checked for correctness to save computation time.
- Only modify `convolution.cu` for kernel development; other files should remain unchanged.
- The project is designed for square matrices of size 2^m × 2^m.
- For large matrices (m > 13), ensure your GPU has sufficient memory.

## Author

Mohammad Ali Etemadi Naeen  


