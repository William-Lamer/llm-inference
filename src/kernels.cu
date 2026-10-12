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

// round to bf16 in float32
static __device__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// One warp computes the dot product of a weight row with x.the result is only valid in lane 0
static __device__ float warp_dot(const bf16 *w_row, const bf16 *x, int cols, int lane) {
    // a uint4 is 16 bytes = 8 bf16
    const uint4 *w_vec = (const uint4 *)w_row;
    const uint4 *x_vec = (const uint4 *)x;
    int n_vec = cols / 8;

    float sum = 0.0f;
    for (int i = lane; i < n_vec; i += 32) {
        uint4 w_packed = w_vec[i];
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
    return sum;
}

// Two rows against the same x in one pass.
static __device__ void warp_dot2(const bf16 *w_a, const bf16 *w_b, const bf16 *x, int cols,
                                 int lane, float *sum_a, float *sum_b) {
    const uint4 *a_vec = (const uint4 *)w_a;
    const uint4 *b_vec = (const uint4 *)w_b;
    const uint4 *x_vec = (const uint4 *)x;
    int n_vec = cols / 8;

    float sa = 0.0f;
    float sb = 0.0f;
    for (int i = lane; i < n_vec; i += 32) {
        uint4 a_packed = a_vec[i];
        uint4 b_packed = b_vec[i];
        uint4 x_packed = x_vec[i];
        const __nv_bfloat162 *a_pairs = (const __nv_bfloat162 *)&a_packed;
        const __nv_bfloat162 *b_pairs = (const __nv_bfloat162 *)&b_packed;
        const __nv_bfloat162 *x_pairs = (const __nv_bfloat162 *)&x_packed;

#pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 a = __bfloat1622float2(a_pairs[k]);
            float2 b = __bfloat1622float2(b_pairs[k]);
            float2 xv = __bfloat1622float2(x_pairs[k]);
            sa += a.x * xv.x;
            sa += a.y * xv.y;
            sb += b.x * xv.x;
            sb += b.y * xv.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sa += __shfl_down_sync(0xffffffff, sa, offset);
        sb += __shfl_down_sync(0xffffffff, sb, offset);
    }
    *sum_a = sa;
    *sum_b = sb;
}

// The uint4 loads need every row to start on a 16 byte boundary
static void check_cols(const char *name, int cols) {
    if (cols % 8 != 0) {
        fprintf(stderr, "%s: cols %d is not a multiple of 8\n", name, cols);
        exit(EXIT_FAILURE);
    }
}

static int gemv_blocks(int rows, int block_size) {
    int rows_per_block = block_size / 32;
    return (rows + rows_per_block - 1) / rows_per_block;
}

static __global__ void gemv_kernel(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r = t / 32;
    int lane = threadIdx.x % 32;

    // r is the warp number. It's same for all 32 lanes so a whole warp returns together
    if (r >= rows) {
        return;
    }

    float sum = warp_dot(w + (size_t)r * cols, x, cols, lane);
    if (lane == 0) {
        out[r] = __float2bfloat16(sum);
    }
}

void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols,
                  cudaStream_t stream) {
    check_cols("gemv", cols);
    int block_size = 256;
    int n_blocks = gemv_blocks(rows, block_size);
    gemv_kernel<<<n_blocks, block_size, 0, stream>>>(out, w, x, rows, cols);
    CUDA_CHECK_LAUNCH();
}


static __global__ void gemv_residual_kernel(bf16 *x, const bf16 *w, const bf16 *in, int rows,
                                            int cols, bf16 *tap) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r = t / 32;
    int lane = threadIdx.x % 32;
    if (r >= rows) {
        return;
    }

    float sum = warp_dot(w + (size_t)r * cols, in, cols, lane);
    if (lane == 0) {
        bf16 y = __float2bfloat16(sum);
        if (tap != NULL) {
            tap[r] = y;
        }
        float total = __bfloat162float(x[r]) + __bfloat162float(y);
        x[r] = __float2bfloat16(total);
    }
}

void gemv_residual_forward(bf16 *x, const bf16 *w, const bf16 *in, int rows, int cols, bf16 *tap,
                           cudaStream_t stream) {
    check_cols("gemv_residual", cols);
    int block_size = 256;
    int n_blocks = gemv_blocks(rows, block_size);
    gemv_residual_kernel<<<n_blocks, block_size, 0, stream>>>(x, w, in, rows, cols, tap);
    CUDA_CHECK_LAUNCH();
}

// q, k, v in one shot. warp r computes row r of the 3 matrices stacked as [wq, wk, wv]
static __global__ void qkv_kernel(bf16 *q, bf16 *k, bf16 *v, const bf16 *wq, const bf16 *wk,
                                  const bf16 *wv, const bf16 *x, int q_rows, int kv_rows,
                                  int cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r = t / 32;
    int lane = threadIdx.x % 32;
    if (r >= q_rows + 2 * kv_rows) {
        return;
    }

    // r is the same for the whole warp so it takes one branch
    const bf16 *w;
    bf16 *out;
    int row;
    if (r < q_rows) {
        w = wq;
        out = q;
        row = r;
    } else if (r < q_rows + kv_rows) {
        w = wk;
        out = k;
        row = r - q_rows;
    } else {
        w = wv;
        out = v;
        row = r - q_rows - kv_rows;
    }

    float sum = warp_dot(w + (size_t)row * cols, x, cols, lane);
    if (lane == 0) {
        out[row] = __float2bfloat16(sum);
    }
}

void qkv_forward(bf16 *q, bf16 *k, bf16 *v, const bf16 *wq, const bf16 *wk, const bf16 *wv,
                 const bf16 *x, int q_rows, int kv_rows, int cols, cudaStream_t stream) {
    check_cols("qkv", cols);
    int block_size = 256;
    int n_blocks = gemv_blocks(q_rows + 2 * kv_rows, block_size);
    qkv_kernel<<<n_blocks, block_size, 0, stream>>>(q, k, v, wq, wk, wv, x, q_rows, kv_rows,
                                                    cols);
    CUDA_CHECK_LAUNCH();
}

// act[r] = silu(gate[r]) * up[r]. gate and up computed with one warp in one pass
static __global__ void gate_up_kernel(bf16 *act, const bf16 *w_gate, const bf16 *w_up,
                                      const bf16 *x, int rows, int cols, bf16 *gate_tap,
                                      bf16 *up_tap) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r = t / 32;
    int lane = threadIdx.x % 32;
    if (r >= rows) {
        return;
    }

    float gate_sum;
    float up_sum;
    const bf16 *gate_row = w_gate + (size_t)r * cols;
    const bf16 *up_row = w_up + (size_t)r * cols;
    warp_dot2(gate_row, up_row, x, cols, lane, &gate_sum, &up_sum);

    if (lane == 0) {
        bf16 gate = __float2bfloat16(gate_sum);
        bf16 up = __float2bfloat16(up_sum);
        if (gate_tap != NULL) {
            gate_tap[r] = gate;
            up_tap[r] = up;
        }
        float g = __bfloat162float(gate);
        float u = __bfloat162float(up);
        float silu = round_bf16(g / (1.0f + expf(-g)));
        act[r] = __float2bfloat16(silu * u);
    }
}

void gate_up_forward(bf16 *act, const bf16 *w_gate, const bf16 *w_up, const bf16 *x, int rows,
                     int cols, bf16 *gate_tap, bf16 *up_tap, cudaStream_t stream) {
    check_cols("gate_up", cols);
    int block_size = 256;
    int n_blocks = gemv_blocks(rows, block_size);
    gate_up_kernel<<<n_blocks, block_size, 0, stream>>>(act, w_gate, w_up, x, rows, cols,
                                                        gate_tap, up_tap);
    CUDA_CHECK_LAUNCH();
}


static __global__ void qk_norm_rope_kernel(bf16 *q, bf16 *k, const bf16 *v,
                                           const bf16 *q_norm_w, const bf16 *k_norm_w,
                                           bf16 *key_cache, bf16 *value_cache, const int *d_pos,
                                           int n_heads, int head_dim, int kv_dim, float eps,
                                           float theta, bf16 *q_tap, bf16 *k_tap) {
    int head = blockIdx.x;
    int is_key = head >= n_heads;
    bf16 *x;
    const bf16 *norm_w;
    bf16 *tap;
    if (is_key) {
        head -= n_heads;
        x = k + head * head_dim;
        norm_w = k_norm_w;
        tap = k_tap;
    } else {
        x = q + head * head_dim;
        norm_w = q_norm_w;
        tap = q_tap;
    }
    int i = threadIdx.x;

    // RMSNorm of the head
    float xi = __bfloat162float(x[i]);
    float sum = xi * xi;
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    __shared__ float warp_sums[32];
    int warp = i / 32;
    int lane = i % 32;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    float total = 0.0f;
    int n_warps = blockDim.x / 32;
    for (int w = 0; w < n_warps; w++) total += warp_sums[w];
    float inv_rms = rsqrtf(total / head_dim + eps);

    bf16 normed = __float2bfloat16(xi * inv_rms);
    float wi = __bfloat162float(norm_w[i]);
    bf16 y = __float2bfloat16(__bfloat162float(normed) * wi);
    if (tap != NULL) {
        tap[head * head_dim + i] = y;
    }

    // RoPE pairs element j with j + half so the normed head goes through shared memory.
    __shared__ bf16 normed_head[1024];
    normed_head[i] = y;
    __syncthreads();

    int pos = *d_pos;
    int half = head_dim / 2;
    size_t slot = (size_t)pos * kv_dim + head * head_dim;
    if (i < half) {
        int j = i;
        float inv_freq = 1.0f / powf(theta, (float)(2 * j) / head_dim);
        float angle = pos * inv_freq;

        float c = round_bf16(cosf(angle));
        float s = round_bf16(sinf(angle));

        float a = __bfloat162float(normed_head[j]);
        float b = __bfloat162float(normed_head[j + half]);

        float a_cos = round_bf16(a * c);
        float b_sin = round_bf16(b * s);
        float b_cos = round_bf16(b * c);
        float a_sin = round_bf16(a * s);
        bf16 out_a = __float2bfloat16(a_cos - b_sin);
        bf16 out_b = __float2bfloat16(b_cos + a_sin);
        x[j] = out_a;
        x[j + half] = out_b;
        if (is_key) {
            key_cache[slot + j] = out_a;
            key_cache[slot + j + half] = out_b;
        }
    }
    if (is_key) {
        value_cache[slot + i] = v[head * head_dim + i];
    }
}

// key_cache and value_cache are one layer's caches.
void qk_norm_rope_forward(bf16 *q, bf16 *k, const bf16 *v, const bf16 *q_norm_w,
                          const bf16 *k_norm_w, bf16 *key_cache, bf16 *value_cache,
                          const int *d_pos, int n_heads, int n_kv_heads, int head_dim, float eps,
                          float theta, bf16 *q_tap, bf16 *k_tap, cudaStream_t stream) {

    if (head_dim % 32 != 0 || head_dim > 1024) {
        fprintf(stderr, "qk_norm_rope: head_dim %d is not a multiple of 32 up to 1024\n",
                head_dim);
        exit(EXIT_FAILURE);
    }
    int n_blocks = n_heads + n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    qk_norm_rope_kernel<<<n_blocks, head_dim, 0, stream>>>(q, k, v, q_norm_w, k_norm_w, key_cache,
                                                           value_cache, d_pos, n_heads, head_dim,
                                                           kv_dim, eps, theta, q_tap, k_tap);
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
