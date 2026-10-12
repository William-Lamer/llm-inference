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

    CUDA_CHECK(cudaMalloc(&s->hb, cfg->intermediate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->hb2, cfg->intermediate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&s->logits, cfg->vocab_size * sizeof(bf16)));

    CUDA_CHECK(cudaMalloc(&s->next_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->pos, sizeof(int)));

    CUDA_CHECK(cudaStreamCreate(&s->stream));
    s->graph = NULL;
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
    CUDA_CHECK(cudaFree(s->hb));
    CUDA_CHECK(cudaFree(s->hb2));
    CUDA_CHECK(cudaFree(s->logits));
    CUDA_CHECK(cudaFree(s->next_token));
    CUDA_CHECK(cudaFree(s->pos));
    if (s->graph != NULL) {
        CUDA_CHECK(cudaGraphExecDestroy(s->graph));
    }
    CUDA_CHECK(cudaStreamDestroy(s->stream));
}

static void dump_layer(int l, const char *name, const bf16 *d_data, int n) {
    char full_name[64];
    snprintf(full_name, sizeof(full_name), "L%02d.%s", l, name);
    dump_bf16(full_name, d_data, n);
}


// The number of weight bytes that are read for one forward() call 
size_t weight_bytes_per_step(const Config *cfg) {
    size_t hidden = cfg->hidden_size;
    size_t head_dim = cfg->head_dim;
    size_t q_dim = cfg->n_heads * head_dim;
    size_t kv_dim = cfg->n_kv_heads * head_dim;
    size_t inter = cfg->intermediate_size;

    size_t layer = 0;
    layer += hidden;          // attn_norm
    layer += q_dim * hidden;  // wq
    layer += kv_dim * hidden; // wk
    layer += kv_dim * hidden; // wv
    layer += head_dim;        // q_norm
    layer += head_dim;        // k_norm
    layer += hidden * q_dim;  // wo
    layer += hidden;          // mlp_norm
    layer += inter * hidden;  // w_gate
    layer += inter * hidden;  // w_up
    layer += hidden * inter;  // w_down

    size_t elements = layer * cfg->n_layers;
    elements += hidden;                   // final_norm
    elements += cfg->vocab_size * hidden; // lm_head
    elements += hidden;                   // one embedding row
    return elements * sizeof(bf16);
}

// K and V for one cached position for all layers
size_t kv_bytes_per_position(const Config *cfg) {
    size_t kv_dim = cfg->n_kv_heads * (size_t)cfg->head_dim;
    return 2 * cfg->n_layers * kv_dim * sizeof(bf16);
}



static void forward_kernels(const Config *cfg, const Weights *w, RunState *s) {
    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int inter = cfg->intermediate_size;
    float scale = (float)(1.0 / sqrt((double)cfg->head_dim));

    embedding_forward(s->x, w->embed, s->next_token, cfg->hidden_size, s->stream);
    dump_bf16("embed", s->x, cfg->hidden_size);

    for (int l = 0; l < cfg->n_layers; l++) {
        const LayerWeights *layer = &w->layers[l];

        rmsnorm_forward(s->xb, s->x, layer->attn_norm, 1, cfg->hidden_size, cfg->rms_norm_eps,
                        s->stream);
        dump_layer(l, "attn_norm", s->xb, cfg->hidden_size);

        gemv_forward(s->q, layer->wq, s->xb, q_dim, cfg->hidden_size, s->stream);
        dump_layer(l, "q_proj", s->q, q_dim);

        // QK-norm: one rmsnorm per head.
        rmsnorm_forward(s->q, s->q, layer->q_norm, cfg->n_heads, cfg->head_dim, cfg->rms_norm_eps,
                        s->stream);
        dump_layer(l, "q_norm", s->q, q_dim);

        gemv_forward(s->k, layer->wk, s->xb, kv_dim, cfg->hidden_size, s->stream);
        dump_layer(l, "k_proj", s->k, kv_dim);
        rmsnorm_forward(s->k, s->k, layer->k_norm, cfg->n_kv_heads, cfg->head_dim,
                        cfg->rms_norm_eps, s->stream);
        dump_layer(l, "k_norm", s->k, kv_dim);

        gemv_forward(s->v, layer->wv, s->xb, kv_dim, cfg->hidden_size, s->stream);
        dump_layer(l, "v_proj", s->v, kv_dim);

        rope_forward(s->q, cfg->n_heads, cfg->head_dim, s->pos, cfg->rope_theta, s->stream);
        dump_layer(l, "q_rope", s->q, q_dim);
        rope_forward(s->k, cfg->n_kv_heads, cfg->head_dim, s->pos, cfg->rope_theta, s->stream);
        dump_layer(l, "k_rope", s->k, kv_dim);

        // this layer's caches
        bf16 *key_cache = s->key_cache + (size_t)l * s->max_seq * kv_dim;
        bf16 *value_cache = s->value_cache + (size_t)l * s->max_seq * kv_dim;
        kv_write_forward(key_cache, value_cache, s->k, s->v, s->pos, kv_dim, s->stream);

        attention_forward(s->attn_out, s->q, key_cache, value_cache, s->att, cfg->n_heads,
                          cfg->n_kv_heads, cfg->head_dim, s->pos, s->max_seq, scale, s->stream);
        dump_layer(l, "attn_out", s->attn_out, q_dim);

        gemv_forward(s->xb, layer->wo, s->attn_out, cfg->hidden_size, q_dim, s->stream);
        dump_layer(l, "o_proj", s->xb, cfg->hidden_size);

        residual_add_forward(s->x, s->xb, cfg->hidden_size, s->stream);
        dump_layer(l, "attn_resid", s->x, cfg->hidden_size);

        rmsnorm_forward(s->xb, s->x, layer->mlp_norm, 1, cfg->hidden_size, cfg->rms_norm_eps,
                        s->stream);
        dump_layer(l, "mlp_norm", s->xb, cfg->hidden_size);

        gemv_forward(s->hb, layer->w_gate, s->xb, inter, cfg->hidden_size, s->stream);
        dump_layer(l, "gate", s->hb, inter);
        gemv_forward(s->hb2, layer->w_up, s->xb, inter, cfg->hidden_size, s->stream);
        dump_layer(l, "up", s->hb2, inter);

        swiglu_forward(s->hb, s->hb2, inter, s->stream);
        dump_layer(l, "mlp_act", s->hb, inter);

        gemv_forward(s->xb, layer->w_down, s->hb, cfg->hidden_size, inter, s->stream);
        dump_layer(l, "down", s->xb, cfg->hidden_size);

        residual_add_forward(s->x, s->xb, cfg->hidden_size, s->stream);
        dump_layer(l, "out", s->x, cfg->hidden_size);
    }

    rmsnorm_forward(s->xb, s->x, w->final_norm, 1, cfg->hidden_size, cfg->rms_norm_eps, s->stream);
    dump_bf16("final_norm", s->xb, cfg->hidden_size);

    gemv_forward(s->logits, w->lm_head, s->xb, cfg->vocab_size, cfg->hidden_size, s->stream);
    dump_bf16("logits", s->logits, cfg->vocab_size);
}

// Records the launches of forward_kernels without running them.
void capture_graph(const Config *cfg, const Weights *w, RunState *s) {
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(s->stream, cudaStreamCaptureModeGlobal));
    forward_kernels(cfg, w, s);
    CUDA_CHECK(cudaStreamEndCapture(s->stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&s->graph, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
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

    CUDA_CHECK(
        cudaMemcpyAsync(s->next_token, &token, sizeof(int), cudaMemcpyHostToDevice, s->stream));
    CUDA_CHECK(cudaMemcpyAsync(s->pos, &pos, sizeof(int), cudaMemcpyHostToDevice, s->stream));

    if (s->graph != NULL) {
        CUDA_CHECK(cudaGraphLaunch(s->graph, s->stream));
    } else {
        forward_kernels(cfg, w, s);
    }
}

int sample_argmax(const Config *cfg, RunState *s) {
    argmax_forward(s->next_token, s->logits, cfg->vocab_size, s->stream);
    int token;
    CUDA_CHECK(
        cudaMemcpyAsync(&token, s->next_token, sizeof(int), cudaMemcpyDeviceToHost, s->stream));
    CUDA_CHECK(cudaStreamSynchronize(s->stream));
    return token;
}

int sample_temperature(const Config *cfg, RunState *s, float temperature, float coin) {
    float inv_temp = 1.0f / temperature;
    sample_forward(s->next_token, s->logits, cfg->vocab_size, inv_temp, coin, s->stream);
    int token;
    CUDA_CHECK(
        cudaMemcpyAsync(&token, s->next_token, sizeof(int), cudaMemcpyDeviceToHost, s->stream));
    CUDA_CHECK(cudaStreamSynchronize(s->stream));
    if (token < 0) {
        fprintf(stderr, "sampling found no token (coin %g)\n", coin);
        exit(EXIT_FAILURE);
    }
    return token;
}
