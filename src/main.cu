#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cuda_profiler_api.h>

#include "dump.cuh"
#include "forward.cuh"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <model_dir> [--checksums] [--tokens file] [--dump dir] [--max-new n]\n"
            "       [--temperature t] [--seed n] [--bench steps] [--runs n] [--no-graph]\n",
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

// xorshift64*
static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// 23 random bits: at most 1 - 2^-23, so u * total rounds to less than total.
static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 9) / 8388608.0f;
}

static int sample(const Config *cfg, RunState *s, float temperature, unsigned long long *rng) {
    if (temperature == 0.0f) {
        return sample_argmax(cfg, s);
    }
    float u = random_f32(rng);
    return sample_temperature(cfg, s, temperature, u);
}


static void generate(const Config *cfg, const Weights *w, RunState *s, int n_prompt, int max_new, float temperature, unsigned long long seed) {
    double start = now_seconds();
    unsigned long long rng = seed;
    int pos = n_prompt;
    int n_generated = 0;
    int n_steps = 0;
    int token = sample(cfg, s, temperature, &rng);
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
        token = sample(cfg, s, temperature, &rng);
    }
    printf("\n");
    fflush(stdout);
    double seconds = now_seconds() - start;
    fprintf(stderr, "generated %d tokens, %d decode steps in %.3f s (%.1f steps/s)\n", n_generated,
            n_steps, seconds, n_steps / seconds);
}

// greedy decode steps after the prompt ignoring EOS so every run has the same length. Run 0 is warmup
static void bench(const Config *cfg, const Weights *w, RunState *s, const int *tokens,
                  int n_tokens, int n_steps, int n_runs) {
    if (n_tokens + n_steps > s->max_seq) {
        fprintf(stderr, "bench: %d prompt tokens + %d steps is past the kv cache size %d\n",
                n_tokens, n_steps, s->max_seq);
        exit(EXIT_FAILURE);
    }

    printf("weight_bytes_per_step %zu\n", weight_bytes_per_step(cfg));
    printf("kv_bytes_per_position %zu\n", kv_bytes_per_position(cfg));

    for (int run = 0; run <= n_runs; run++) {
        for (int pos = 0; pos < n_tokens; pos++) {
            forward(cfg, w, s, tokens[pos], pos);
        }
        int token = sample_argmax(cfg, s);

        if (run > 0) {
            CUDA_CHECK(cudaProfilerStart());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_seconds();
        for (int i = 0; i < n_steps; i++) {
            forward(cfg, w, s, token, n_tokens + i);
            token = sample_argmax(cfg, s);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double seconds = now_seconds() - start;
        if (run > 0) {
            CUDA_CHECK(cudaProfilerStop());
        }

        fprintf(stderr, "%s %d: %d steps in %.3f s (%.2f ms/step)\n", run == 0 ? "warmup" : "run",
                run, n_steps, seconds, seconds * 1e3 / n_steps);
        if (run > 0) {
            printf("run %d %d %.6f\n", run, n_steps, seconds);
        }
    }
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
    float temperature = 0.0f;
    unsigned long long seed = (unsigned long long)time(NULL);
    int bench_steps = 0;
    int bench_runs = 5;
    int use_graph = 1;
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
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            i++;
            temperature = atof(argv[i]);
            if (temperature < 0.0f) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            i++;
            seed = strtoull(argv[i], NULL, 10);
            if (seed == 0) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            i++;
            bench_steps = atoi(argv[i]);
            if (bench_steps <= 0) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            i++;
            bench_runs = atoi(argv[i]);
            if (bench_runs <= 0) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[i], "--no-graph") == 0) {
            use_graph = 0;
        } else {
            usage(argv[0]);
        }
    }
    if (bench_steps > 0 && tokens_path == NULL) {
        fprintf(stderr, "--bench needs --tokens\n");
        exit(EXIT_FAILURE);
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

        if (use_graph && dump_dir == NULL) {
            capture_graph(&cfg, &w, &s);
        }
        fprintf(stderr, "cuda graph %s\n", s.graph != NULL ? "on" : "off");
        if (bench_steps > 0) {
            bench(&cfg, &w, &s, tokens, n_tokens, bench_steps, bench_runs);
            free_run_state(&s);
            free(tokens);
            free_weights(&w);
            return 0;
        }
        for (int pos = 0; pos < n_tokens; pos++) {
            forward(&cfg, &w, &s, tokens[pos], pos);
        }
        fprintf(stderr, "ran %d positions\n", n_tokens);

        if (max_new > 0) {
            // the reference dumps only cover the prompt
            dump_close();
            if (temperature > 0.0f) {
                fprintf(stderr, "temperature %g, seed %llu\n", temperature, seed);
            }
            generate(&cfg, &w, &s, n_tokens, max_new, temperature, seed);
        }
        free_run_state(&s);
        free(tokens);
    }

    free_weights(&w);
    return 0;
}
