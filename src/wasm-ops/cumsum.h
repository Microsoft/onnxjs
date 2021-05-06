// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void cumsum_f32(void *);
void cumsum_f32_imp(const float *X, const int32_t *dims, const int32_t rank,
                    int32_t axis, const bool exclusive, const bool reverse,
                    float *Y);
}
