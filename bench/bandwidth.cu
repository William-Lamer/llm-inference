// Measuring DRAM bandwidth with a device-to-device copy that's bigger than the 48 MB L2.
#include <stdio.h>
#include <stdlib.h>

#include "common.cuh"

#define COPY_BYTES ((size_t)512 << 20)
#define WARMUP 3
#define RUNS 20

static int compare_floats(const void *a, const void *b) {
    float x = *(const float *)a;
    float y = *(const float *)b;
    return (x > y) - (x < y);
}

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "%s, copying %zu MiB, median of %d runs\n", prop.name, COPY_BYTES >> 20, RUNS);

    void *src;
    void *dst;
    CUDA_CHECK(cudaMalloc(&src, COPY_BYTES));
    CUDA_CHECK(cudaMalloc(&dst, COPY_BYTES));
    CUDA_CHECK(cudaMemset(src, 1, COPY_BYTES));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ms[RUNS];
    for (int i = 0; i < WARMUP + RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dst, src, COPY_BYTES, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        if (i >= WARMUP) {
            CUDA_CHECK(cudaEventElapsedTime(&ms[i - WARMUP], start, stop));
        }
    }
    qsort(ms, RUNS, sizeof(float), compare_floats);
    float median_ms = ms[RUNS / 2];

    // A copy reads every byte once and writes it once.
    double bytes_moved = 2.0 * COPY_BYTES;
    double gbs = bytes_moved / (median_ms * 1e-3) / 1e9;
    fprintf(stderr, "median %.3f ms (min %.3f, max %.3f)\n", median_ms, ms[0], ms[RUNS - 1]);
    printf("copy_bandwidth_gbs %.1f\n", gbs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    return 0;
}
