#pragma once

#include <stddef.h>

#include "common.cuh"

#define MAX_SHARDS 8

typedef struct {
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float rms_norm_eps;
    float rope_theta;
    int tie_word_embeddings;
} Config;

// Matrices are PyTorch linear weights: [out, in] and row-major.
typedef struct {
    const bf16 *attn_norm; // [hidden]
    const bf16 *wq;        // [n_heads * head_dim, hidden]
    const bf16 *wk;        // [n_kv_heads * head_dim, hidden]
    const bf16 *wv;        // [n_kv_heads * head_dim, hidden]
    const bf16 *q_norm;    // [head_dim]
    const bf16 *k_norm;    // [head_dim]
    const bf16 *wo;        // [hidden, n_heads * head_dim]
    const bf16 *mlp_norm;  // [hidden]
    const bf16 *w_gate;    // [intermediate, hidden]
    const bf16 *w_up;      // [intermediate, hidden]
    const bf16 *w_down;    // [hidden, intermediate]
} LayerWeights;

typedef struct {
    char name[96];
    const bf16 *data;
    size_t numel;
} TensorRef;

// The structs are in host memory, the bf16 pointers points to the GPU.
typedef struct {
    const bf16 *embed;      // [vocab, hidden]
    LayerWeights *layers;   // [n_layers]
    const bf16 *final_norm; // [hidden]
    const bf16 *lm_head;    // [vocab, hidden], the same pointer as embed when tied

    TensorRef *tensors; // every tensor loaded, by safetensors name
    int n_tensors;
    void *gpu_buffers[MAX_SHARDS];
    int n_buffers;
    size_t gpu_bytes;
} Weights;

void load_config(const char *model_dir, Config *cfg);
void load_weights(const char *model_dir, const Config *cfg, Weights *w);
void free_weights(Weights *w);
void print_weight_checksums(const Weights *w);
