// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "cumsum.h"
#include "common.h"
#include "utils/shape_utils.h"

// Wasm interop method
void cumsum_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  const float *x = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const int32_t *dims = PARAM_INT32_PTR(data, dataIndex[2]);
  const int32_t rank = PARAM_INT32(data, dataIndex[3]);
  const int32_t axis = PARAM_INT32(data, dataIndex[4]);
  const bool exclusive = PARAM_BOOL(data, dataIndex[5]);
  const bool reverse = PARAM_BOOL(data, dataIndex[6]);

  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  cumsum_f32_imp(x, dims, rank, axis, exclusive, reverse, output);
}

// Core operator implementation
void cumsum_f32_imp(const float *X, const int32_t *dims, const int32_t rank,
                    int32_t axis, const bool exclusive, const bool reverse,
                    float *Y) {
  if (axis < 0) {
    axis = rank + axis;
  }

  // const index : number[] = new Array(y.dims.length).fill(0);
  size_t i = 0;
  std::vector<int32_t> dimsVector(dims, dims + rank);
  std::vector<int32_t> strides = ShapeUtils::compute_strides(dimsVector);
  size_t size = ShapeUtils::size_from_dims(dimsVector);

  if (reverse) {
    i = size - 1;
  }

  while (i < size && i >= 0) {

    size_t indexAtAxis = ShapeUtils::offset_to_index(strides, i, axis);

    size_t prevIndex = i + (reverse ? strides.at(axis) : -strides.at(axis));

    bool start = (indexAtAxis == 0 && !reverse) ||
                 (indexAtAxis == dimsVector.at(axis) && reverse);

    if (start && !exclusive) {
      Y[i] = X[i];
    } else if (start && exclusive) {
      Y[i] = 0;
    } else if (!start && !exclusive) {
      Y[i] = Y[prevIndex] + X[i];
    } else {
      Y[i] = Y[prevIndex] + X[prevIndex];
    }

    if (reverse) {
      i--;
    } else {
      i++;
    }
  }
}
