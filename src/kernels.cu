#include "kernels.cuh"

static __global__ void embedding_kernel(bf16 *out, const bf16 *table, int token, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        const bf16 *row = table + (size_t)token * hidden;
        out[i] = row[i];
    }
}

void embedding_forward(bf16 *out, const bf16 *table, int token, int hidden) {
    int block_size = 256;
    int n_blocks = (hidden + block_size - 1) / block_size;
    embedding_kernel<<<n_blocks, block_size>>>(out, table, token, hidden);
    CUDA_CHECK_LAUNCH();
}
