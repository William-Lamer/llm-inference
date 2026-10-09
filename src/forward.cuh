#pragma once

#include "model.cuh"

// allocated once and reused for every position.
typedef struct {
    bf16 *x; // [hidden] residual stream
} RunState;

void alloc_run_state(RunState *s, const Config *cfg);
void free_run_state(RunState *s);
void forward(const Config *cfg, const Weights *w, RunState *s, int token);
