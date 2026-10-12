#pragma once

#include "model.cuh"

// allocated once and reused for every position.
typedef struct {
    bf16 *x;               // [hidden] residual stream
    bf16 *xb;              // [hidden] normed x, input of the attention and MLP blocks
    bf16 *q;               // [n_heads * head_dim]
    bf16 *k;               // [n_kv_heads * head_dim]
    bf16 *v;               // [n_kv_heads * head_dim]
    bf16 *attn_out;        // [n_heads * head_dim] attention output, heads concatenated
    float *att;            // [n_heads, max_seq] scores, then softmax probabilities
    bf16 *key_cache;       // [n_layers, max_seq, n_kv_heads * head_dim]
    bf16 *value_cache;     // [n_layers, max_seq, n_kv_heads * head_dim]
    int max_seq;           // positions the KV cache holds
    bf16 *hb;              // [intermediate] silu(gate) * up
    bf16 *tap1;            // [max(q_dim, intermediate)] fused-kernel intermediates, for dumps
    bf16 *tap2;            // [max(q_dim, intermediate)] fused-kernel intermediates, for dumps
    bf16 *logits;          // [vocab]
    int *next_token;       // [1] written by sampling, read by the embedding of the next step
    int *pos;              // [1] read by the kernels so their launch arguments never change
    cudaStream_t stream;   // every launch and copy of a step
    cudaGraphExec_t graph; // NULL, or the captured kernels of one forward pass
} RunState;

void alloc_run_state(RunState *s, const Config *cfg);
void free_run_state(RunState *s);
void capture_graph(const Config *cfg, const Weights *w, RunState *s);
void forward(const Config *cfg, const Weights *w, RunState *s, int token, int pos);
int sample_argmax(const Config *cfg, RunState *s);
int sample_temperature(const Config *cfg, RunState *s, float temperature, float u);
size_t weight_bytes_per_step(const Config *cfg);
size_t kv_bytes_per_position(const Config *cfg);
