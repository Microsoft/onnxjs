// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "einsum.h"
#include "common.h"
#include "utils/shape_utils.h"

// Wasm interop method
void einsum_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];

  const float *a = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *b = PARAM_FLOAT_PTR(data, dataIndex[2]);
  float *y = PARAM_FLOAT_PTR(data, dataIndex[3]);
  const int32_t *dims = PARAM_INT32_PTR(data, dataIndex[4]);
  const int32_t rank = PARAM_INT32(data, dataIndex[5]);
  const int32_t *outputIndices = PARAM_INT32_PTR(data, dataIndex[6]);
  const int32_t outputRank = PARAM_INT32(data, dataIndex[7]);
  const int32_t *input1Indices = PARAM_INT32_PTR(data, dataIndex[8]);
  const int32_t input1Rank = PARAM_INT32(data, dataIndex[9]);
  const int32_t *input2Indices = PARAM_INT32_PTR(data, dataIndex[10]);
  const int32_t input2Rank = PARAM_INT32(data, dataIndex[11]);

  einsum_f32_imp(a, b, y, dims, rank, outputIndices, outputRank, input1Indices,
                 input1Rank, input2Indices, input2Rank);
}

void einsum_f32_imp(const float *A, const float *B, float *Y,
                    const int32_t *dims, const int32_t rank,
                    const int32_t *outputIndices, int32_t outputRank,
                    const int32_t *input1Indices, int32_t input1Rank,
                    const int32_t *input2Indices, int32_t input2Rank) {
  std::vector<int32_t> dimsVector(dims, dims + rank);
  // std::vector<int32_t> strides = ShapeUtils::compute_strides(dimsVector);
  size_t totalSize = ShapeUtils::size_from_dims(dimsVector);
  size_t i = 0;
  std::vector<int32_t> index(rank, 0);

  std::vector<int32_t> outputStrides(outputRank, 1);
  for (size_t j = outputRank - 2; j >= 0; j--) {
    outputStrides[j] = outputStrides[j + 1] * dimsVector[outputIndices[j]];
  }

  std::vector<int32_t> input1Strides(input1Rank, 1);
  for (size_t j = input1Rank - 2; j >= 0; j--) {
    input1Strides[j] = input1Strides[j + 1] * dimsVector[input1Indices[j]];
  }

  std::vector<int32_t> input2Strides(input2Rank, 1);
  for (size_t j = input2Rank - 2; j >= 0; j--) {
    input2Strides[j] = input2Strides[j + 1] * dimsVector[input2Indices[j]];
  }

  while (i < totalSize) {
    size_t outputOffset = 0;
    for (size_t j = 0; j < outputRank; j++) {
      outputOffset += index[outputIndices[j]] * outputStrides[j];
    }

    size_t input1Offset = 0;
    for (size_t j = 0; j < input1Rank; j++) {
      input1Offset += index[input1Indices[j]] * input1Strides[j];
    }

    size_t input2Offset = 0;
    for (size_t j = 0; j < input2Rank; j++) {
      input2Offset += index[input2Indices[j]] * input2Strides[j];
    }

    Y[outputOffset] += A[input1Offset] * B[input2Offset];

    i++;
    ShapeUtils::increment_index(index, dimsVector, dimsVector.size());
  }
}

void einsum_single_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];

  const float *a = PARAM_FLOAT_PTR(data, dataIndex[1]);
  float *y = PARAM_FLOAT_PTR(data, dataIndex[2]);
  const int32_t *dims = PARAM_INT32_PTR(data, dataIndex[3]);
  const int32_t rank = PARAM_INT32(data, dataIndex[4]);
  const int32_t *outputIndices = PARAM_INT32_PTR(data, dataIndex[5]);
  const int32_t outputRank = PARAM_INT32(data, dataIndex[6]);
  const int32_t *inputIndices = PARAM_INT32_PTR(data, dataIndex[7]);
  const int32_t inputRank = PARAM_INT32(data, dataIndex[8]);

  einsum_single_f32_imp(a, y, dims, rank, outputIndices, outputRank,
                        inputIndices, inputRank);
}

// Core operator implementation
void einsum_single_f32_imp(const float *A, float *Y, const int32_t *dims,
                           const int32_t rank, const int32_t *outputIndices,
                           int32_t outputRank, const int32_t *inputIndices,
                           int32_t inputRank) {
  std::vector<int32_t> dimsVector(dims, dims + rank);
  // std::vector<int32_t> strides = ShapeUtils::compute_strides(dimsVector);
  size_t totalSize = ShapeUtils::size_from_dims(dimsVector);
  size_t i = 0;
  std::vector<int32_t> index(rank, 0);

  std::vector<int32_t> outputStrides(outputRank, 1);
  for (size_t j = outputRank - 2; j >= 0; j--) {
    outputStrides[j] = outputStrides[j + 1] * dimsVector[outputIndices[j]];
  }

  std::vector<int32_t> inputStrides(inputRank, 1);
  for (size_t j = inputRank - 2; j >= 0; j--) {
    inputStrides[j] = inputStrides[j + 1] * dimsVector[inputIndices[j]];
  }

  while (i < totalSize) {
    size_t outputOffset = 0;
    for (size_t j = 0; j < outputRank; j++) {
      outputOffset += index[outputIndices[j]] * outputStrides[j];
    }

    size_t input1Offset = 0;
    for (size_t j = 0; j < inputRank; j++) {
      input1Offset += index[inputIndices[j]] * inputStrides[j];
    }

    Y[outputOffset] += A[input1Offset];

    i++;
    ShapeUtils::increment_index(index, dimsVector, dimsVector.size());
  }
}
