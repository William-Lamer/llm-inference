#include "common.cuh"
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

static __global__ void rmsnorm_kernel(bf16 *out, const bf16 *in, const bf16 *weight, int dim,
                                      float eps) {
    int row = blockIdx.x;
    const bf16 *x = in + (size_t)row * dim;
    bf16 *y = out + (size_t)row * dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        sum += v * v;
    }

    // Each thread now has its partial sum, need to combine

    // sum inside each warp
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }

    __shared__ float warp_sums[32];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    float total = 0.0f;
    int n_warps = blockDim.x / 32;
    for (int w = 0; w < n_warps; w++) total += warp_sums[w];

    float inv_rms = rsqrtf(total / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        bf16 normed = __float2bfloat16(v * inv_rms);
        float w = __bfloat162float(weight[i]);
        y[i] = __float2bfloat16(__bfloat162float(normed) * w);
    }
}

void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps) {
    int block_size = 256;
    rmsnorm_kernel<<<rows, block_size>>>(out, in, weight, dim, eps);
    CUDA_CHECK_LAUNCH();
}
