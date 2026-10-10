#include <stdio.h>
#include <string.h>
#include <time.h>

#include "dump.cuh"
#include "forward.cuh"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <model_dir> [--checksums] [--tokens file] [--dump dir] [--max-new n]\n",
            prog);
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

static int is_eos(const Config *cfg, int token) {
    for (int i = 0; i < cfg->n_eos; i++) {
        if (cfg->eos_ids[i] == token) {
            return 1;
        }
    }
    return 0;
}

static void generate(const Config *cfg, const Weights *w, RunState *s, int n_prompt, int max_new) {
    double start = now_seconds();
    int pos = n_prompt;
    int n_generated = 0;
    int n_steps = 0;
    int token = sample_argmax(cfg, s);
    while (1) {
        printf("%d ", token);
        fflush(stdout);
        n_generated++;
        if (is_eos(cfg, token)) break;
        if (n_generated == max_new) break;
        if (pos == s->max_seq) break;
        forward(cfg, w, s, token, pos);
        pos++;
        n_steps++;
        token = sample_argmax(cfg, s);
    }
    printf("\n");
    fflush(stdout);
    double seconds = now_seconds() - start;
    fprintf(stderr, "generated %d tokens, %d decode steps in %.3f s (%.1f steps/s)\n", n_generated,
            n_steps, seconds, n_steps / seconds);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
    }
    const char *model_dir = argv[1];
    int checksums = 0;
    const char *tokens_path = NULL;
    const char *dump_dir = NULL;
    int max_new = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--checksums") == 0) {
            checksums = 1;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            i++;
            tokens_path = argv[i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            i++;
            dump_dir = argv[i];
        } else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            i++;
            max_new = atoi(argv[i]);
            if (max_new < 0) {
                usage(argv[0]);
            }
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

    // stdout carries only generated token IDs, so everything else goes to stderr.
    fprintf(stderr, "layers %d, hidden %d, heads %d/%d, head_dim %d, mlp %d, vocab %d\n",
            cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
            cfg.intermediate_size, cfg.vocab_size);
    fprintf(stderr, "rms_norm_eps %g, rope_theta %g, tied embeddings %d\n", cfg.rms_norm_eps,
            cfg.rope_theta, cfg.tie_word_embeddings);
    fprintf(stderr, "loaded %d tensors from %d file(s): %.3f GB on the GPU in %.2f s\n",
            w.n_tensors, w.n_buffers, w.gpu_bytes / 1e9, load_seconds);

    if (dump_dir != NULL) {
        dump_open(dump_dir);
    }

    if (tokens_path != NULL) {
        int n_tokens;
        int *tokens = read_tokens(tokens_path, &n_tokens);
        if (n_tokens == 0) {
            fprintf(stderr, "%s has no tokens\n", tokens_path);
            exit(EXIT_FAILURE);
        }
        RunState s;
        alloc_run_state(&s, &cfg);
        for (int pos = 0; pos < n_tokens; pos++) {
            forward(&cfg, &w, &s, tokens[pos], pos);
        }
        fprintf(stderr, "ran %d positions\n", n_tokens);

        if (max_new > 0) {
            // the reference dumps only cover the prompt
            dump_close();
            generate(&cfg, &w, &s, n_tokens, max_new);
        }
        free_run_state(&s);
        free(tokens);
    }

    free_weights(&w);
    return 0;
}
