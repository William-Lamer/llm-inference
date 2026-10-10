#pragma once

#include "model.cuh"

// allocated once and reused for every position.
typedef struct {
    bf16 *x;           // [hidden] residual stream
    bf16 *xb;          // [hidden] normed x, input of the attention and MLP blocks
    bf16 *q;           // [n_heads * head_dim]
    bf16 *k;           // [n_kv_heads * head_dim]
    bf16 *v;           // [n_kv_heads * head_dim]
    bf16 *attn_out;    // [n_heads * head_dim] attention output, heads concatenated
    float *att;        // [n_heads, max_seq] scores, then softmax probabilities
    bf16 *key_cache;   // [n_layers, max_seq, n_kv_heads * head_dim]
    bf16 *value_cache; // [n_layers, max_seq, n_kv_heads * head_dim]
    int max_seq;       // positions the KV cache holds
    bf16 *hb;          // [intermediate] gate, then silu(gate) * up
    bf16 *hb2;         // [intermediate] up
    bf16 *logits;      // [vocab]
} RunState;

void alloc_run_state(RunState *s, const Config *cfg);
void free_run_state(RunState *s);
void forward(const Config *cfg, const Weights *w, RunState *s, int token, int pos);
