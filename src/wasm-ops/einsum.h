// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void einsum_f32(void *);
void einsum_f32_imp(const float *A, const float *B, float *Y,
                    const int32_t *dims, const int32_t rank,
                    const int32_t *outputIndices, int32_t outputRank,
                    const int32_t *input1Indices, int32_t input1Rank,
                    const int32_t *input2Indices, int32_t input2Rank);
void einsum_single_f32(void *);
void einsum_single_f32_imp(const float *A, float *Y, const int32_t *dims,
                           const int32_t rank, const int32_t *outputIndices,
                           int32_t outputRank, const int32_t *inputIndices,
                           int32_t inputRank);
}
