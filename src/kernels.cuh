#pragma once

#include "common.cuh"

void embedding_forward(bf16 *out, const bf16 *table, int token, int hidden);
void rmsnorm_forward(bf16 *out, const bf16 *in, const bf16 *weight, int rows, int dim, float eps);
void gemv_forward(bf16 *out, const bf16 *w, const bf16 *x, int rows, int cols);
