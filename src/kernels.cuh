#pragma once

#include "common.cuh"

// Every pointer is a GPU pointer.
void embedding_forward(bf16 *out, const bf16 *table, int token, int hidden);
