#pragma once

#include "common.cuh"

void embedding_forward(bf16 *out, const bf16 *table, const int *d_token, int hidden,
                       cudaStream_t stream);
void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps,
                     cudaStream_t stream);
void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols, cudaStream_t stream);
void rope_forward(bf16 *x, int n_heads, int head_dim, const int *d_pos, float theta,
                  cudaStream_t stream);
void kv_write_forward(bf16 *key_cache, bf16 *value_cache, const bf16 *k, const bf16 *v,
                      const int *d_pos, int kv_dim, cudaStream_t stream);
void attention_forward(bf16 *out, const bf16 *q, const bf16 *key_cache, const bf16 *value_cache,
                       float *att, int n_heads, int n_kv_heads, int head_dim, const int *d_pos,
                       int max_seq, float scale, cudaStream_t stream);
void residual_add_forward(bf16 *x, const bf16 *y, int n, cudaStream_t stream);
void swiglu_forward(bf16 *gate, const bf16 *up, int n, cudaStream_t stream);
void argmax_forward(int *out, const bf16 *x, int n, cudaStream_t stream);
void sample_forward(int *out, const bf16 *x, int n, float inv_temp, float u, cudaStream_t stream);
