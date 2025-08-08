# CUDA Scan (Parallel Prefix Sum in GF(2^8))

This project implements an efficient parallel scan (prefix sum) operation over vectors of elements in a finite field (Galois Field, GF(2^8)), using CUDA for GPU acceleration. The scan is generalized to perform matrix-vector multiplications in the finite field, making it suitable for cryptographic and coding theory applications.

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

The scan (prefix sum) is a fundamental parallel primitive. In this project, the scan is generalized: instead of a simple sum, each step applies a 4x4 matrix multiplication in GF(2^8) to the running result, then adds the next vector. This is useful in applications such as cryptography, coding theory, and scientific computing.

- **Finite Field Arithmetic:** All operations are performed in GF(2^8), using efficient lookup tables.
- **Matrix-Vector Scan:** Each scan step is a matrix-vector multiplication followed by vector addition in the field.
- **CPU and GPU Implementations:** Both serial (CPU) and parallel (GPU) versions are provided for performance and correctness comparison.
- **Performance Measurement:** Execution time and correctness (error count) are reported.

## File Structure

- `scan_main.cu` — Main program: handles memory allocation, timing, correctness checking, and launching kernels (**do not modify**).
- `scan.cu` — Main CUDA kernel and helper functions for the scan operation (**modifiable**).
- `scan.h` — Kernel prototypes and macros (**do not modify**).
- `finite_field.cpp` / `finite_field.h` — Implements finite field arithmetic and field parameter generation.
- `helper.h` — Type definitions for fixed-width integers.
- `gputimer.h` — Utility for measuring GPU kernel execution time.
- `gpuerrors.h` — Error handling macros for CUDA API calls.
- `README.md` — Project documentation.

## How It Works

1. **Initialization:** Random input vectors and a random 4x4 matrix are generated in GF(2^8).
2. **CPU Reference Calculation:** The CPU computes the scan result serially for correctness checking.
3. **GPU Computation:** The CUDA kernel performs the parallel scan using a balanced tree algorithm, adapted for matrix-vector operations in the finite field.
4. **Result Validation:** The output is copied back to the host and compared with the CPU result (error count).
5. **Performance Reporting:** The program prints the matrix size, CPU and GPU execution times, and the number of errors.

## Usage

### Prerequisites

- CUDA-capable GPU (with at least 1GB memory)
- NVIDIA CUDA Toolkit
- C++ compiler (e.g., `nvcc`)

### Compilation

Open a terminal in the project directory and run:

```
nvcc -O2 scan.cu scan_main.cu finite_field.cpp -o scan.exe 
```

### Running

The program expects a single argument: `m`, where the scan size is `n = 2^m` (e.g., `m=10` for `n=1024` vectors).

```
scan.exe 12
```

**Note:** For `m >= 14`, the data may not fit in 1GB GPU memory.

### Output Example

```
Device Name: NVIDIA GeForce GTX 1080
m=12        n=4096      CPU=12.34 ms      GPU=3.21 ms      num_error=0
```

- `CPU` is the serial CPU computation time.
- `GPU` is the total GPU computation time.
- `num_error` is the number of mismatches between CPU and GPU results.

## Customization

- **Matrix Size:** The scan is performed on vectors of length 4, using a 4x4 matrix. You can modify the code to experiment with different dimensions.
- **Kernel Logic:** The main CUDA kernel is in `scan.cu` and can be optimized or adapted for other scan operations or field sizes.
- **Field Polynomial:** The irreducible polynomial for GF(2^8) can be changed in `finite_field.cpp` for different field definitions.

## Notes

- Only modify `scan.cu` for kernel development; other files should remain unchanged.
- The project is designed for vectors of size 4 and scan lengths of `n = 2^m`.
- For large `m` (e.g., `m > 13`), ensure your GPU has sufficient memory.
- The code checks correctness by comparing GPU results with the CPU reference.

## Author

Mohammad Ali Etemadi Naeen



