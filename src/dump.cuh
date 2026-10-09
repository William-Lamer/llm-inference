#pragma once

#include "common.cuh"

void dump_open(const char *dir);
void dump_bf16(const char *name, const bf16 *d_data, int n);
void dump_f32(const char *name, const float *d_data, int n);
