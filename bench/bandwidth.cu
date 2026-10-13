// Measuring DRAM bandwidth on buffers bigger than the 48 MB L2
#include <stdio.h>
#include <stdlib.h>

#include "common.cuh"

#define COPY_BYTES ((size_t)512 << 20)
// Needed a lot of warmup steps to actualy get up to speed
#define WARMUP 200
#define RUNS 20

#define READ_WARMUP 3

static int compare_floats(const void *a, const void *b) {
    float x = *(const float *)a;
    float y = *(const float *)b;
    return (x > y) - (x < y);
}

static float median(float *ms, int n) {
    qsort(ms, n, sizeof(float), compare_floats);
    return ms[n / 2];
}

static __global__ void read_kernel(const uint4 *in, size_t n, unsigned int *out) {
    unsigned int acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
#pragma unroll 4
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = in[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) {
        *out = acc;
    }
}

static float time_read(const uint4 *in, unsigned int *out, int blocks, int threads,
                       cudaEvent_t start, cudaEvent_t stop) {
    size_t n = COPY_BYTES / sizeof(uint4);
    float ms[RUNS];
    for (int i = 0; i < READ_WARMUP + RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        read_kernel<<<blocks, threads>>>(in, n, out);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        if (i >= READ_WARMUP) {
            CUDA_CHECK(cudaEventElapsedTime(&ms[i - READ_WARMUP], start, stop));
        }
    }
    return median(ms, RUNS);
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
    float median_ms = median(ms, RUNS);

    // A copy reads every byte once and writes it once.
    double bytes_moved = 2.0 * COPY_BYTES;
    double gbs = bytes_moved / (median_ms * 1e-3) / 1e9;
    fprintf(stderr, "median %.3f ms (min %.3f, max %.3f)\n", median_ms, ms[0], ms[RUNS - 1]);
    printf("copy_bandwidth_gbs %.1f\n", gbs);

    // runs right after the copy so the gpu is already at full clock
    unsigned int *out;
    CUDA_CHECK(cudaMalloc(&out, sizeof(unsigned int)));
    int block_sizes[] = {256, 512, 1024};
    int blocks_per_sm[] = {2, 4, 8, 16};
    float best_ms = 0.0f;
    for (int b = 0; b < 3; b++) {
        for (int k = 0; k < 4; k++) {
            int threads = block_sizes[b];
            int blocks = blocks_per_sm[k] * prop.multiProcessorCount;
            float read_ms = time_read((const uint4 *)src, out, blocks, threads, start, stop);
            double read_gbs = COPY_BYTES / (read_ms * 1e-3) / 1e9;
            fprintf(stderr, "read %4d blocks x %4d threads: %.3f ms, %.1f GB/s\n", blocks, threads,
                    read_ms, read_gbs);
            if (best_ms == 0.0f || read_ms < best_ms) {
                best_ms = read_ms;
            }
        }
    }
    printf("read_bandwidth_gbs %.1f\n", COPY_BYTES / (best_ms * 1e-3) / 1e9);
    CUDA_CHECK(cudaFree(out));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    return 0;
}
