#include <stdio.h>
#include <string.h>
#include <time.h>

#include "dump.cuh"
#include "model.cuh"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s <model_dir> [--checksums] [--tokens file] [--dump dir]\n", prog);
    exit(EXIT_FAILURE);
}

// reads token IDs, like the dump tokens.txt.
static int *read_tokens(const char *path, int *n_tokens) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(EXIT_FAILURE);
    }
    int capacity = 64;
    int n = 0;
    int *tokens = (int *)malloc(capacity * sizeof(int));
    int token;
    while (fscanf(f, "%d", &token) == 1) {
        if (n == capacity) {
            capacity *= 2;
            tokens = (int *)realloc(tokens, capacity * sizeof(int));
        }
        if (tokens == NULL) {
            fprintf(stderr, "out of memory reading %s\n", path);
            exit(EXIT_FAILURE);
        }
        tokens[n] = token;
        n++;
    }
    fclose(f);
    *n_tokens = n;
    return tokens;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
    }
    const char *model_dir = argv[1];
    int checksums = 0;
    const char *tokens_path = NULL;
    const char *dump_dir = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--checksums") == 0) {
            checksums = 1;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            i++;
            tokens_path = argv[i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            i++;
            dump_dir = argv[i];
        } else {
            usage(argv[0]);
        }
    }

    Config cfg;
    load_config(model_dir, &cfg);

    CUDA_CHECK(cudaFree(0));

    double start = now_seconds();
    Weights w;
    load_weights(model_dir, &cfg, &w);
    double load_seconds = now_seconds() - start;

    if (checksums) {
        print_weight_checksums(&w);
        free_weights(&w);
        return 0;
    }

    printf("layers %d, hidden %d, heads %d/%d, head_dim %d, mlp %d, vocab %d\n", cfg.n_layers,
           cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.intermediate_size,
           cfg.vocab_size);
    printf("rms_norm_eps %g, rope_theta %g, tied embeddings %d\n", cfg.rms_norm_eps, cfg.rope_theta,
           cfg.tie_word_embeddings);
    printf("loaded %d tensors from %d file(s): %.3f GB on the GPU in %.2f s\n", w.n_tensors,
           w.n_buffers, w.gpu_bytes / 1e9, load_seconds);

    if (dump_dir != NULL) {
        dump_open(dump_dir);
    }

    if (tokens_path != NULL) {
        int n_tokens;
        int *tokens = read_tokens(tokens_path, &n_tokens);
        for (int pos = 0; pos < n_tokens; pos++) {
            // TODO(M2): forward pass for tokens[pos] at position pos
        }
        printf("ran %d positions\n", n_tokens);
        free(tokens);
    }

    free_weights(&w);
    return 0;
}
