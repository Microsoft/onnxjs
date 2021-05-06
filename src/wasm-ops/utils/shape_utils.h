// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

namespace ShapeUtils {
size_t size_from_dims(const std::vector<int32_t> &dims);
std::vector<int32_t> compute_strides(const std::vector<int32_t> &dims);
// Fills in values in the strides vector. Assumes it is of the required size.
void compute_strides(const std::vector<int32_t> &dims,
                     std::vector<int32_t> &strides);
size_t indices_to_offset(const std::vector<int32_t> &strides,
                         const std::vector<int32_t> &indices);
std::vector<int32_t> offset_to_indices(const std::vector<int32_t> &strides,
                                       size_t offset);
// Fills in values in the indices vector. Assumes it is of the required size.
void offset_to_indices(const std::vector<int32_t> &strides, size_t offset,
                       std::vector<int32_t> &indices);
// Gives the index at a specific axis from a given offset
size_t offset_to_index(const std::vector<int32_t> &strides, size_t offset,
                       int32_t axis);
}; // namespace ShapeUtils
