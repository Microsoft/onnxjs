// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "shape_utils.h"
#include <math.h>

size_t ShapeUtils::size_from_dims(const std::vector<int32_t> &dims) {
  auto rank = dims.size();
  if (rank == 0) {
    return 1;
  }
  if (rank == 1) {
    return dims[0];
  }
  size_t size = 1;
  for (auto &e : dims) {
    size *= e;
  }
  return size;
}

std::vector<int32_t>
ShapeUtils::compute_strides(const std::vector<int32_t> &dims) {
  auto rank = dims.size();
  if (rank == 0 || rank == 1) {
    std::vector<int32_t> strides(1, 1);
    return strides;
  }
  std::vector<int32_t> strides(rank);
  ShapeUtils::compute_strides(dims, strides);
  return strides;
}

void ShapeUtils::compute_strides(const std::vector<int32_t> &dims,
                                 std::vector<int32_t> &strides) {
  auto rank = dims.size();
  if (rank == 0 || rank == 1) {
    strides[0] = 1;
    return;
  }
  strides[rank - 1] = 1;
  strides[rank - 2] = dims[rank - 1];
  for (int32_t i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
}

size_t ShapeUtils::indices_to_offset(const std::vector<int32_t> &strides,
                                     const std::vector<int32_t> &indices) {
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += strides[i] * indices[i];
  }
  return offset;
}

std::vector<int32_t>
ShapeUtils::offset_to_indices(const std::vector<int32_t> &strides,
                              size_t offset) {
  auto rank = strides.size();
  if (rank == 0) {
    return std::vector<int32_t>();
  }
  if (rank == 1) {
    return std::vector<int32_t>(1, offset * strides[0]);
  }
  std::vector<int32_t> indices(rank);
  ShapeUtils::offset_to_indices(strides, offset, indices);
  return indices;
}

void ShapeUtils::offset_to_indices(const std::vector<int32_t> &strides,
                                   size_t offset,
                                   std::vector<int32_t> &indices) {
  auto rank = strides.size();
  if (rank == 0) {
    return;
  }
  if (rank == 1) {
    indices[0] = offset * strides[0];
    return;
  }
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    indices[i] = floor(offset / strides[i]);
    offset -= indices[i] * strides[i];
  }
  indices[indices.size() - 1] = offset;
}

void ShapeUtils::increment_index(std::vector<int32_t> &index,
                                 const std::vector<int32_t> &dims,
                                 size_t axisToIncrementOn) {
  for (size_t i = axisToIncrementOn - 1; i >= 0; --i) {
    index[i]++;
    if (index[i] < dims[i]) {
      break;
    }
    index[i] = 0;
  }
}
