# Heat Transformation with Texture Memory (CUDA)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [CUDA Texture Memory](#cuda-texture-memory)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Compilation](#compilation)
  - [Running](#running)
  - [Output Example](#output-example)
- [Customization](#customization)
- [Notes](#notes)
- [Author](#author)

## Overview

This project simulates 2D heat transformation using CUDA, leveraging texture memory for efficient access patterns. It demonstrates how parallel GPU computation and spatially cached memory can accelerate stencil-based scientific simulations. 

- **Stencil Computation:** Each cell's temperature is updated based on its four neighbors, modeling heat diffusion.
- **Texture Memory:** Uses CUDA's 2D texture memory for fast, cached, read-only access to grid data.
- **CPU vs GPU:** Includes both serial (CPU) and parallel (GPU) implementations for performance and correctness comparison.
- **Performance Measurement:** Reports both total GPU time and kernel-only time.

## Features

- Serial CPU and parallel GPU implementations for heat transformation.
- Efficient use of CUDA texture memory for stencil operations.
- Mean Squared Error (MSE) comparison between CPU and GPU results.
- Robust error checking for CUDA API calls.
- Timing utilities for accurate performance measurement.

## File Structure

- `htt_main.cu` — Main program: memory allocation, timing, correctness checking, and kernel launching (**do not modify**).
- `htt.cu` — CUDA kernel and texture memory logic (**modifiable**).
- `htt.h` — Kernel function prototype and constants (**do not modify**).
- `gputimer.h` — Utility for measuring GPU kernel execution time.
- `gpuerrors.h` — Error handling macros for CUDA API calls.
- `README.md` — Project documentation.

## How It Works

1. **Grid Initialization:** The temperature grid is initialized with random values between 20.0 and 30.0 degrees Celsius.
2. **Mathematical Formulation:**  
   The new temperature of each cell depends on the temperatures of its neighboring cells (top, bottom, right, left), incorporating the constant `K` over a specific duration.  
   The update equation for each cell is:
   
   $$
   T_{new} = T_{old} + K \times \left[ (T_{top} - T_{old}) + (T_{bottom} - T_{old}) + (T_{right} - T_{old}) + (T_{left} - T_{old}) \right]
   $$
   
   Or equivalently:
   
   $$
   T_{new} = T_{old} + K \times \left( T_{top} + T_{bottom} + T_{right} + T_{left} - 4 \times T_{old} \right)
   $$
   
   In a two-dimensional scenario, we consider neighbors located at the top, bottom, right, and left of each cell.
3. **CPU Reference Calculation:** The CPU computes the reference result for correctness checking.
4. **GPU Computation:** The grid is copied to the device, bound to a CUDA 2D texture, and the kernel updates temperatures in parallel.
5. **Result Validation:** The output is copied back to the host and compared with the CPU result using Mean Squared Error (MSE).
6. **Performance Reporting:** The program prints the grid size, GPU execution time, kernel time, and MSE.

## CUDA Texture Memory

Texture memory in CUDA is a cached, read-only memory optimized for spatial locality. It is ideal for stencil computations like heat diffusion, where neighboring values are frequently accessed. In this project, the input grid is bound to a 2D texture, and the kernel accesses cell values using `tex2D`, improving memory access efficiency and overall performance.

## Usage

### Prerequisites

- CUDA-capable GPU (with at least 1GB memory)
- NVIDIA CUDA Toolkit
- C++ compiler (e.g., `nvcc`)

### Compilation

Open a terminal in the project directory and run:

```sh
nvcc -O2 htt_main.cu htt.cu -o htt 
```

### Running

The program expects a single argument: `m`, where the grid size is `n = 2^m` (e.g., `m=10` for a `1024x1024` grid).

```sh
./htt 12
```

**Note:** For `m >= 14`, the grid may not fit in 1GB GPU memory.

### Output Example

```
Device Name: NVIDIA GeForce GTX 1050
m=12 n=4096 GPU=123.45 ms GPU-Kernel=98.76 ms mse=0.000123
```

- `GPU` is the total GPU computation time.
- `GPU-Kernel` is the kernel execution time.
- `mse` is the mean squared error between CPU and GPU results.

## Customization

- **Kernel Logic:** You can optimize the CUDA kernel in `htt.cu` for your GPU architecture or experiment with different block sizes.
- **Iteration Count:** The number of heat transformation iterations is set to 10; you can adjust this in the source code for longer simulations.
- **Boundary Handling:** The kernel handles grid boundaries by mirroring edge values; you may modify this for different physical models.

## Notes

- The grid is a matrix of size $N \times N$, where $N = 2^m$, and each array cell represents a cell within the room.
- The simulation applies the heat transformation formula for 10 iterations by default.
- For large grids (`m > 13`), ensure your GPU has sufficient memory.


## Author

Mohammad Ali Etemadi Naeen

