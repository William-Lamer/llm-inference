#pragma once

#include "model.cuh"

// allocated once and reused for every position.
typedef struct {
    bf16 *x;  // [hidden] residual stream
    bf16 *xb; // [hidden] normed x, input of the attention and MLP blocks
    bf16 *q;  // [n_heads * head_dim]
    bf16 *k;  // [n_kv_heads * head_dim]
    bf16 *v;  // [n_kv_heads * head_dim]
} RunState;

void alloc_run_state(RunState *s, const Config *cfg);
void free_run_state(RunState *s);
void forward(const Config *cfg, const Weights *w, RunState *s, int token);
