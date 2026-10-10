#include <math.h>

#include "dump.cuh"
#include "forward.cuh"
#include "kernels.cuh"

// Cap of the KV cache
#define MAX_SEQ 4096

void alloc_run_state(RunState *s, const Config *cfg) {
    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;

    CUDA_CHECK(cudaMalloc(&s->x, cfg->hidden_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->xb, cfg->hidden_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->q, q_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->k, kv_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->v, kv_dim * sizeof(bf16)));

    s->max_seq = cfg->max_seq_len;
    if (s->max_seq > MAX_SEQ) {
        s->max_seq = MAX_SEQ;
    }

    size_t cache_len = (size_t)cfg->n_layers * s->max_seq * kv_dim;
    size_t att_len = (size_t)cfg->n_heads * s->max_seq;

    CUDA_CHECK(cudaMalloc(&s->attn_out, q_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->att, att_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->key_cache, cache_len * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->value_cache, cache_len * sizeof(bf16)));
}

void free_run_state(RunState *s) {
    CUDA_CHECK(cudaFree(s->x));
    CUDA_CHECK(cudaFree(s->xb));
    CUDA_CHECK(cudaFree(s->q));
    CUDA_CHECK(cudaFree(s->k));
    CUDA_CHECK(cudaFree(s->v));
    CUDA_CHECK(cudaFree(s->attn_out));
    CUDA_CHECK(cudaFree(s->att));
    CUDA_CHECK(cudaFree(s->key_cache));
    CUDA_CHECK(cudaFree(s->value_cache));
}

void forward(const Config *cfg, const Weights *w, RunState *s, int token, int pos) {
    // to be safe
    if (token < 0 || token >= cfg->vocab_size) {
        fprintf(stderr, "forward: token %d outside vocab of %d\n", token, cfg->vocab_size);
        exit(EXIT_FAILURE);
    }

    if (pos >= s->max_seq) {
        fprintf(stderr, "forward: position %d past the kv cache size %d\n", pos, s->max_seq);
        exit(EXIT_FAILURE);
    }

    embedding_forward(s->x, w->embed, token, cfg->hidden_size);
    dump_bf16("embed", s->x, cfg->hidden_size);

    int l = 0;
    const LayerWeights *layer = &w->layers[l];

    rmsnorm_forward(s->xb, s->x, layer->attn_norm, 1, cfg->hidden_size, cfg->rms_norm_eps);
    dump_bf16("L00.attn_norm", s->xb, cfg->hidden_size);

    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;

    gemv_forward(s->q, layer->wq, s->xb, q_dim, cfg->hidden_size);
    dump_bf16("L00.q_proj", s->q, q_dim);

    // QK-norm: one rmsnorm per head.
    rmsnorm_forward(s->q, s->q, layer->q_norm, cfg->n_heads, cfg->head_dim, cfg->rms_norm_eps);
    dump_bf16("L00.q_norm", s->q, q_dim);

    gemv_forward(s->k, layer->wk, s->xb, kv_dim, cfg->hidden_size);
    dump_bf16("L00.k_proj", s->k, kv_dim);
    rmsnorm_forward(s->k, s->k, layer->k_norm, cfg->n_kv_heads, cfg->head_dim, cfg->rms_norm_eps);
    dump_bf16("L00.k_norm", s->k, kv_dim);

    gemv_forward(s->v, layer->wv, s->xb, kv_dim, cfg->hidden_size);
    dump_bf16("L00.v_proj", s->v, kv_dim);

    rope_forward(s->q, cfg->n_heads, cfg->head_dim, pos, cfg->rope_theta);
    dump_bf16("L00.q_rope", s->q, q_dim);
    rope_forward(s->k, cfg->n_kv_heads, cfg->head_dim, pos, cfg->rope_theta);
    dump_bf16("L00.k_rope", s->k, kv_dim);

    // the cache of the layer then the pos.
    bf16 *key_cache = s->key_cache + (size_t)l * s->max_seq * kv_dim;
    bf16 *value_cache = s->value_cache + (size_t)l * s->max_seq * kv_dim;
    bf16 *k_slot = key_cache + (size_t)pos * kv_dim;
    bf16 *v_slot = value_cache + (size_t)pos * kv_dim;
    CUDA_CHECK(cudaMemcpyAsync(k_slot, s->k, kv_dim * sizeof(bf16), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(v_slot, s->v, kv_dim * sizeof(bf16), cudaMemcpyDeviceToDevice));

    float scale = (float)(1.0 / sqrt((double)cfg->head_dim));
    attention_forward(s->attn_out, s->q, key_cache, value_cache, s->att, cfg->n_heads,
                      cfg->n_kv_heads, cfg->head_dim, pos, s->max_seq, scale);
    dump_bf16("L00.attn_out", s->attn_out, q_dim);
}
