#pragma once

#include "common.cuh"

void embedding_forward(bf16 *out, const bf16 *table, int token, int hidden);
void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps);
void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols);
void rope_forward(bf16 *x, int n_heads, int head_dim, int pos, float theta);
void attention_forward(bf16 *out, const bf16 *q, const bf16 *key_cache, const bf16 *value_cache,
                       float *att, int n_heads, int n_kv_heads, int head_dim, int pos, int max_seq,
                       float scale);
void residual_add_forward(bf16 *x, const bf16 *y, int n);
void swiglu_forward(bf16 *gate, const bf16 *up, int n);
