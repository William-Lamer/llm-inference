#pragma once

#include "common.cuh"

// The tap arguments of the fused kernels receive an intermediate tensor that they no
// longer writes to memory.
void embedding_forward(bf16 *out, const bf16 *table, const int *d_token, int hidden,
                       cudaStream_t stream);
void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps,
                     cudaStream_t stream);
void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols, cudaStream_t stream);
void gemv_residual_forward(bf16 *x, const bf16 *w, const bf16 *in, int rows, int cols, bf16 *tap,
                           cudaStream_t stream);
void qkv_forward(bf16 *q, bf16 *k, bf16 *v, const bf16 *wq, const bf16 *wk, const bf16 *wv,
                 const bf16 *x, int q_rows, int kv_rows, int cols, cudaStream_t stream);
void gate_up_forward(bf16 *act, const bf16 *w_gate, const bf16 *w_up, const bf16 *x, int rows,
                     int cols, bf16 *gate_tap, bf16 *up_tap, cudaStream_t stream);
void qk_norm_rope_forward(bf16 *q, bf16 *k, const bf16 *v, const bf16 *q_norm_w,
                          const bf16 *k_norm_w, bf16 *key_cache, bf16 *value_cache,
                          const int *d_pos, int n_heads, int n_kv_heads, int head_dim, float eps,
                          float theta, bf16 *q_tap, bf16 *k_tap, cudaStream_t stream);
void attention_forward(bf16 *out, const bf16 *q, const bf16 *key_cache, const bf16 *value_cache,
                       float *att, int n_heads, int n_kv_heads, int head_dim, const int *d_pos,
                       int max_seq, float scale, cudaStream_t stream);
void argmax_forward(int *out, const bf16 *x, int n, cudaStream_t stream);
void sample_forward(int *out, const bf16 *x, int n, float inv_temp, float u, cudaStream_t stream);
