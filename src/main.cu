#include <stdio.h>
#include <string.h>
#include <time.h>

#include "model.cuh"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [--checksums]\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    int checksums = argc > 2 && strcmp(argv[2], "--checksums") == 0;

    Config cfg;
    load_config(model_dir, &cfg);

    CUDA_CHECK(cudaFree(0));

    double start = now_seconds();
    Weights w;
    load_weights(model_dir, &cfg, &w);
    double load_seconds = now_seconds() - start;

    if (checksums) {
        print_weight_checksums(&w);
    } else {
        printf("layers %d, hidden %d, heads %d/%d, head_dim %d, mlp %d, vocab %d\n", cfg.n_layers,
               cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate_size,
               cfg.vocab_size);
        printf("rms_norm_eps %g, rope_theta %g, tied embeddings %d\n", cfg.rms_norm_eps,
               cfg.rope_theta, cfg.tie_word_embeddings);
        printf("loaded %d tensors from %d file(s): %.3f GB on the GPU in %.2f s\n", w.n_tensors,
               w.n_buffers, w.gpu_bytes / 1e9, load_seconds);
    }

    free_weights(&w);
    return 0;
}
