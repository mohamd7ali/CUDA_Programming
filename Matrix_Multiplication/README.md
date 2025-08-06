# CUDA Matrix Multiplication


This project implements efficient matrix multiplication using both CPU and GPU (CUDA) approaches. It is designed for educational purposes, demonstrating parallel programming concepts and performance comparison between serial and parallel computation.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Build Instructions](#build-instructions)
  - [Output](#output)
- [Customization](#customization)
- [Notes](#notes)
- [Author](#author)

## Overview

- **Language:** C++ with CUDA extensions
- **Purpose:** Multiply two large square matrices and compare CPU vs. GPU performance
- **Features:**
  - Customizable matrix size (power-of-two dimensions)
  - CPU implementation for reference and correctness checking
  - GPU implementation using CUDA kernels
  - Automatic timing and Mean Squared Error (MSE) calculation for result validation

## File Structure

- `mm_main.cu` — Main program, handles memory allocation, timing, and correctness checks. **Do not modify.**
- `mm.cu` — CUDA kernel and grid/block configuration. **Modify this file for kernel optimization.**
- `mm.h` — Header for matrix macros and kernel prototypes. **Do not modify.**
- `gputimer.h` — Utility for measuring GPU kernel execution time.
- `gpuerrors.h` — Error handling macros for CUDA API calls.

## How It Works

1. **Matrix Generation:** Two random matrices are generated with values between -8 and 8.
2. **CPU Multiplication:** Reference result is computed using a serial algorithm.
3. **GPU Multiplication:** CUDA kernel computes the product in parallel.
4. **Validation:** Results are compared using MSE for correctness.
5. **Performance Measurement:** Execution times for both CPU and GPU are reported.

## Usage

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler compatible with CUDA

### Build Instructions

1. Open a terminal in the `Matrix_Multiplication` directory.
2. Compile the project using:
   ```powershell
   nvcc mm_main.cu mm.cu -o matrix_mul.exe
   ```
3. Run the executable with a matrix size parameter (e.g., for 1024x1024 matrices, use m=10):
   ```powershell
   .\matrix_mul.exe 10
   ```
   - Valid values for `m`: 10 to 13 (higher values may exceed GPU memory).

### Output

The program prints:
- Device name
- Matrix size
- CPU and GPU execution times (ms)
- GPU kernel time (ms)
- MSE (Mean Squared Error) between CPU and GPU results

Example:
```
Device Name: NVIDIA GeForce GTX 1050
m=10 n=1024 CPU=1200 ms GPU=15 ms GPU-Kernel=12 ms mse=0
```

## Customization

- **Kernel Optimization:** Edit `mm.cu` to improve kernel performance (e.g., tiling, shared memory).
- **Grid/Block Configuration:** Adjust `getDimGrid` and `getDimBlock` for different matrix sizes and hardware.

## Notes

- Only modify `mm.cu` for kernel development.
- The project is designed for square matrices of size 2^m × 2^m.
- For large matrices (m > 13), ensure your GPU has sufficient memory.


## Author
**Mohammad Ali Etemadi Naeen**