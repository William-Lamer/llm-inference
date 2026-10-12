#include <math.h>
#include <limits.h>

#include "common.cuh"
#include "kernels.cuh"

static __global__ void embedding_kernel(bf16 *out, const bf16 *table, const int *d_token,
                                        int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        int token = *d_token;
        const bf16 *row = table + (size_t)token * hidden;
        out[i] = row[i];
    }
}

void embedding_forward(bf16 *out, const bf16 *table, const int *d_token, int hidden,
                       cudaStream_t stream) {
    int block_size = 256;
    int n_blocks = (hidden + block_size - 1) / block_size;
    embedding_kernel<<<n_blocks, block_size, 0, stream>>>(out, table, d_token, hidden);
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

void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps,
                     cudaStream_t stream) {
    int block_size = 256;
    rmsnorm_kernel<<<rows, block_size, 0, stream>>>(out, in, weight, dim, eps);
    CUDA_CHECK_LAUNCH();
}

static __global__ void gemv_kernel(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r = t / 32;
    int lane = threadIdx.x % 32;

    // r is the warp number. The same for all 32 lanes.
    if (r >= rows) {
        return;
    }

    // a uint4 is 16 bytes = 8 bf16
    const uint4 *w_row = (const uint4 *)(w + (size_t)r * cols);
    const uint4 *x_vec = (const uint4 *)x;
    int n_vec = cols / 8;

    float sum = 0.0f;
    for (int i = lane; i < n_vec; i += 32) {
        uint4 w_packed = w_row[i];
        uint4 x_packed = x_vec[i];
        const __nv_bfloat162 *w_pairs = (const __nv_bfloat162 *)&w_packed;
        const __nv_bfloat162 *x_pairs = (const __nv_bfloat162 *)&x_packed;

        // fully unrolled so w_pairs[k] stays in registers
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 a = __bfloat1622float2(w_pairs[k]);
            float2 b = __bfloat1622float2(x_pairs[k]);
            sum += a.x * b.x;
            sum += a.y * b.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        out[r] = __float2bfloat16(sum);
    }
}

void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols,
                  cudaStream_t stream) {
    // needs to be 16 byte aligned.
    if (cols % 8 != 0) {
        fprintf(stderr, "gemv: cols %d is not a multiple of 8\n", cols);
        exit(1);
    }
    int block_size = 256;
    int rows_per_block = block_size / 32;
    int n_blocks = (rows + rows_per_block - 1) / rows_per_block;
    gemv_kernel<<<n_blocks, block_size, 0, stream>>>(out, w, x, rows, cols);
    CUDA_CHECK_LAUNCH();
}

// round to bf16 in float32
static __device__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

static __global__ void rope_kernel(bf16 *x, int n_heads, int head_dim, const int *d_pos,
                                   float theta) {
    int half = head_dim / 2;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n_heads * half) {
        int head = t / half;
        int j = t % half;
        bf16 *h = x + head * head_dim;

        int pos = *d_pos;
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

void rope_forward(bf16 *x, int n_heads, int head_dim, const int *d_pos, float theta,
                  cudaStream_t stream) {
    int n_pairs = n_heads * (head_dim) / 2;
    int block_size = 256;
    int n_blocks = (n_pairs + block_size - 1) / block_size;
    rope_kernel<<<n_blocks, block_size, 0, stream>>>(x, n_heads, head_dim, d_pos, theta);
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

// Copies k and v into slot *d_pos of one layer's caches. That address is computed on device
// so the launch arguments stay the same at every position.
static __global__ void kv_write_kernel(bf16 *key_cache, bf16 *value_cache, const bf16 *k,
                                       const bf16 *v, const int *d_pos, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        int pos = *d_pos;
        size_t slot = (size_t)pos * kv_dim;
        key_cache[slot + i] = k[i];
        value_cache[slot + i] = v[i];
    }
}

void kv_write_forward(bf16 *key_cache, bf16 *value_cache, const bf16 *k, const bf16 *v,
                      const int *d_pos, int kv_dim, cudaStream_t stream) {
    int block_size = 256;
    int n_blocks = (kv_dim + block_size - 1) / block_size;
    kv_write_kernel<<<n_blocks, block_size, 0, stream>>>(key_cache, value_cache, k, v, d_pos,
                                                        kv_dim);
    CUDA_CHECK_LAUNCH();
}

// One block per query head
static __global__ void attention_kernel(bf16 *out, const bf16 *q, const bf16 *key_cache,
                                        const bf16 *value_cache, float *att, int n_heads,
                                        int n_kv_heads, int head_dim, const int *d_pos,
                                        int max_seq, float scale) {
    int pos = *d_pos;
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
                       float *att, int n_heads, int n_kv_heads, int head_dim, const int *d_pos,
                       int max_seq, float scale, cudaStream_t stream) {

    int block_size = 128;
    attention_kernel<<<n_heads, block_size, 0, stream>>>(out, q, key_cache, value_cache, att,
                                                         n_heads, n_kv_heads, head_dim, d_pos,
                                                         max_seq, scale);
    CUDA_CHECK_LAUNCH();
}

static __global__ void residual_add_kernel(bf16 *x, const bf16 *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = __bfloat162float(x[i]) + __bfloat162float(y[i]);
        x[i] = __float2bfloat16(sum);
    }
}

void residual_add_forward(bf16 *x, const bf16 *y, int n, cudaStream_t stream) {
    int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    residual_add_kernel<<<n_blocks, block_size, 0, stream>>>(x, y, n);
    CUDA_CHECK_LAUNCH();
}

static __global__ void swiglu_kernel(bf16 *gate, const bf16 *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu = round_bf16(g / (1.0f + expf(-g)));
        gate[i] = __float2bfloat16(silu * u);
    }
}

void swiglu_forward(bf16 *gate, const bf16 *up, int n, cudaStream_t stream) {
    int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    swiglu_kernel<<<n_blocks, block_size, 0, stream>>>(gate, up, n);
    CUDA_CHECK_LAUNCH();
}

// keeps the larger value, but on a tie keeps the lowest index.
static __device__ void argmax_pair(float *val, int *idx, float second_val, int second_idx) {
    if (second_val > *val || (second_val == *val && second_idx < *idx)) {
        *val = second_val;
        *idx = second_idx;
    }
}

// its not worth to do multiple passes since its so small
static __global__ void argmax_kernel(int *out, const bf16 *x, int n) {
    float best_val = -INFINITY;
    int best_idx = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        argmax_pair(&best_val, &best_idx, __bfloat162float(x[i]), i);
    }

    for (int off = 16; off > 0; off >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, off);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, off);
        argmax_pair(&best_val, &best_idx, other_val, other_idx);
    }

    __shared__ float warp_vals[32];
    __shared__ int warp_idxs[32];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_vals[warp] = best_val;
        warp_idxs[warp] = best_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n_warps = blockDim.x / 32;
        for (int w = 0; w < n_warps; w++) {
            argmax_pair(&best_val, &best_idx, warp_vals[w], warp_idxs[w]);
        }
        *out = best_idx;
    }
}

void argmax_forward(int *out, const bf16 *x, int n, cudaStream_t stream) {
    int block_size = 1024;
    argmax_kernel<<<1, block_size, 0, stream>>>(out, x, n);
    CUDA_CHECK_LAUNCH();
}

static __device__ float sample_weight(float x, float max_val, float inv_temp) {
    return expf((x - max_val) * inv_temp);
}

// inverse-cdf sampling from softmax(x / temperature).
static __global__ void sample_kernel(int *out, const bf16 *x, int n, float inv_temp, float u) {
    __shared__ float chunk_sums[1024];

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(x[i]));
    }
    float max_val = block_max(local_max);

    int chunk = (n + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * chunk;
    int end = min(start + chunk, n);
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += sample_weight(__bfloat162float(x[i]), max_val, inv_temp);
    }
    chunk_sums[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }
    float total = 0.0f;
    for (int t = 0; t < blockDim.x; t++) total += chunk_sums[t];
    float target = u * total;

    // cdf repeats the additions in the same order,
    float cdf = 0.0f;
    int t = 0;
    while (t < blockDim.x - 1 && cdf + chunk_sums[t] <= target) {
        cdf += chunk_sums[t];
        t++;
    }
    int chosen = -1;
    int chunk_start = t * chunk;
    int chunk_end = min(chunk_start + chunk, n);
    float s = 0.0f;
    for (int i = chunk_start; i < chunk_end; i++) {
        s += sample_weight(__bfloat162float(x[i]), max_val, inv_temp);
        if (cdf + s > target) {
            chosen = i;
            break;
        }
    }
    *out = chosen;
}

// block size must match chunk_sums in sample_kernel
void sample_forward(int *out, const bf16 *x, int n, float inv_temp, float u, cudaStream_t stream) {
    sample_kernel<<<1, 1024, 0, stream>>>(out, x, n, inv_temp, u);
    CUDA_CHECK_LAUNCH();
}
