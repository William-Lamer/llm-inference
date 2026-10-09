#include "dump.cuh"
#include "forward.cuh"
#include "kernels.cuh"

void alloc_run_state(RunState *s, const Config *cfg) {
    CUDA_CHECK(cudaMalloc(&s->x, cfg->hidden_size * sizeof(bf16)));
}

void free_run_state(RunState *s) {
    CUDA_CHECK(cudaFree(s->x));
}

void forward(const Config *cfg, const Weights *w, RunState *s, int token) {
    // to be safe
    if (token < 0 || token >= cfg->vocab_size) {
        fprintf(stderr, "forward: token %d outside vocab of %d\n", token, cfg->vocab_size);
        exit(EXIT_FAILURE);
    }

    embedding_forward(s->x, w->embed, token, cfg->hidden_size);
    dump_bf16("embed", s->x, cfg->hidden_size);
}
