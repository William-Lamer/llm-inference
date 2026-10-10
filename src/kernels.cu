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

static __global__ void gemv_kernel(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        const bf16 *w_row = w + (size_t)r * cols;
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = __bfloat162float(w_row[c]);
            float b = __bfloat162float(x[c]);
            sum += a * b;
        }
        out[r] = __float2bfloat16(sum);
    }
}

void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols) {
    int block_size = 256;
    int n_blocks = (rows + block_size - 1) / block_size;
    gemv_kernel<<<n_blocks, block_size>>>(out, w, x, rows, cols);
    CUDA_CHECK_LAUNCH();
}


// round to bf16 in float32
static __device__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

static __global__ void rope_kernel(bf16 *x, int n_heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n_heads * half) {
        int head = t / half;
        int j = t % half;
        bf16 *h = x + head * head_dim;

        float inv_freq = 1.0f / powf(theta, (float)(2 * j) / head_dim);
        float angle = pos * inv_freq;

        float c = round_bf16(cosf(angle));
        float s = round_bf16(sinf(angle));

        float a = __bfloat162float(h[j]);
        float b = __bfloat162float(h[j + half]);

        float a_cos = round_bf16(a * c);
        float b_sin = round_bf16(b * s);
        float b_cos = round_bf16(b * c);
        float a_sin = round_bf16(a * s);
        h[j] = __float2bfloat16(a_cos - b_sin);
        h[j + half] = __float2bfloat16(b_cos + a_sin);
    }


}

void rope_forward(bf16 *x, int n_heads, int head_dim, int pos, float theta) {
    int n_pairs = n_heads * (head_dim) / 2;
    int block_size = 256;
    int n_blocks = (n_pairs + block_size - 1) / block_size;
    rope_kernel<<<n_blocks, block_size>>>(x, n_heads, head_dim, pos, theta);
    CUDA_CHECK_LAUNCH();
}
