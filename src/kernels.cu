#include <math.h>

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


// the total over the whole block, returned to every thread.
static __device__ float block_sum(float v) {
    __shared__ float partials[32];
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        partials[warp] = v;
    }
    __syncthreads();
    float total = 0.0f;
    int n_warps = blockDim.x / 32;
    for (int w = 0; w < n_warps; w++) total += partials[w];
    return total;
}

static __device__ float block_max(float v) {
    __shared__ float partials[32];
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        partials[warp] = v;
    }
    __syncthreads();
    float result = -INFINITY;
    int n_warps = blockDim.x / 32;
    for (int w = 0; w < n_warps; w++) result = fmaxf(result, partials[w]);
    return result;
}


// One block per query head
static __global__ void attention_kernel(bf16 *out, const bf16 *q, const bf16 *key_cache,
                                        const bf16 *value_cache, float *att, int n_heads,
                                        int n_kv_heads, int head_dim, int pos, int max_seq,
                                        float scale) {
    int h = blockIdx.x;
    int n_rep = n_heads / n_kv_heads;
    int kv_head = h / n_rep;
    int kv_dim = n_kv_heads * head_dim;

    const bf16 *q_head = q + h * head_dim;
    float *scores = att + (size_t)h * max_seq;

    // step 1: one thread per cache position
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const bf16 *k = key_cache + (size_t)t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __bfloat162float(q_head[d]) * __bfloat162float(k[d]);
        }
        scores[t] = round_bf16(round_bf16(dot) * scale);
    }
    __syncthreads();

    // step 2: softmax from 0 to pos
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) local_max = fmaxf(local_max, scores[t]);
    float max_score = block_max(local_max);

    float local_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) local_sum += expf(scores[t] - max_score);
    float sum_exp = block_sum(local_sum);

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        scores[t] = round_bf16(expf(scores[t] - max_score) / sum_exp);
    }
    __syncthreads();

    // step 3: one thread per output dimension, weighted sum of the cached values.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const bf16 *v = value_cache + (size_t)t * kv_dim + kv_head * head_dim;
            acc += scores[t] * __bfloat162float(v[d]);
        }
        out[h * head_dim + d] = __float2bfloat16(acc);
    }
}

void attention_forward(bf16 *out, const bf16 *q, const bf16 *key_cache, const bf16 *value_cache,
                       float *att, int n_heads, int n_kv_heads, int head_dim, int pos, int max_seq,
                       float scale) {

    int block_size = 128;
    attention_kernel<<<n_heads, block_size>>>(out, q, key_cache, value_cache, att, n_heads,
                                              n_kv_heads, head_dim, pos, max_seq, scale);
    CUDA_CHECK_LAUNCH();
}





