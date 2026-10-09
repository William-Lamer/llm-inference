#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n  %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err_), #call);                                              \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// at the next sync
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
