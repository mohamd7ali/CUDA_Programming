# CUDA Programming Projects

This repository contains a collection of CUDA-based projects focused on parallel programming, scientific computing, and GPU acceleration. Each project demonstrates a key concept in high-performance computing, including matrix operations, convolution, heat simulation, and finite field arithmetic.

## Projects Overview

- **Matrix Multiplication**  
  Implements efficient multiplication of large square matrices using both CPU and GPU (CUDA) approaches. Includes performance measurement and correctness validation.  
  *Directory:* `Matrix_Multiplication`

- **Block Matrix Multiplication**  
  Demonstrates tiled (block) matrix multiplication using CUDA shared memory for improved performance. Allows experimentation with tile sizes and block dimensions.  
  *Directory:* `Block_Matrix_Multiplication`

- **CUDA Scan (Parallel Prefix Sum in GF(2^8))**  
  Implements a generalized parallel scan (prefix sum) operation over vectors in a finite field (Galois Field, GF(2^8)), using matrix-vector multiplications. Useful for cryptography and coding theory.  
  *Directory:* `Scan`
  
- **2D Convolution**  
  Performs full 2D convolution of two square matrices, with both CPU and GPU implementations. Measures speedup and validates correctness using Mean Squared Error (MSE).  
  *Directory:* `2D_Convolution`

- **Heat Transformation with Texture Memory**  
  Simulates heat dissipation in a 2D grid using a stencil computation. Utilizes CUDA texture memory for efficient neighbor access and compares CPU vs. GPU performance.  
  *Directory:* `HeatTransformation_TextureMemory`


## Features

- Serial (CPU) and parallel (GPU) implementations for all algorithms
- Performance measurement and reporting (execution time, kernel time)
- Correctness validation (Mean Squared Error, error count)
- Modular code structure for easy experimentation and optimization
- Robust error checking and timing utilities

## Getting Started

Each project contains its own README with detailed instructions on compilation, usage, and customization.  
General prerequisites:
- NVIDIA CUDA-capable GPU (with sufficient memory)
- CUDA Toolkit
- C++ compiler (e.g., `nvcc`)

To build and run a project, navigate to its directory and follow the instructions in its README.

## Author

Mohammad Ali Etemadi Naeen