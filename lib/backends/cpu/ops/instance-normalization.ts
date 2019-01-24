// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InstanceNormalization} from '../../../ops/instance-normalization';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuInstanceNormalization extends InstanceNormalization {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = instanceNormalization(inputs[0], inputs[1], inputs[2], this.epsilon);
    return [output];
  }
}

export function instanceNormalization(x: Tensor, scale: Tensor, b: Tensor, epsilon: number) {
  const inputDimensions = x.dims;
  const N = inputDimensions[0];
  const C = inputDimensions[1];

  // calculate channel size (i.e.) data points per channel
  let channelSize = 1;
  for (let i = 2; i < inputDimensions.length; i++) {
    channelSize *= inputDimensions[i];
  }
  const sampleSize = C * channelSize;

  const output = new Tensor(x.dims, x.type);

  const X = x.floatData;
  const Y = output.floatData;
  const scaleData = scale.numberData;
  const bData = b.numberData;

  let temp: number;
  let mean: number;
  let variance: number;
  let sampleOffset: number;
  let physicalOffset: number;
  let iterEnd: number;

  for (let n = 0; n < N; ++n) {
    sampleOffset = n * sampleSize;
    for (let c = 0; c < C; ++c) {
      physicalOffset = sampleOffset + c * channelSize;
      iterEnd = physicalOffset + channelSize;

      // compute mean for this channel
      temp = 0;
      for (let i = physicalOffset; i < iterEnd; ++i) {
        temp += X[i];
      }
      mean = temp / channelSize;

      // compute variance for this channel
      temp = 0;
      for (let i = physicalOffset; i < iterEnd; ++i) {
        temp += Math.pow(X[i] - mean, 2);
      }
      variance = temp / channelSize;

      // compute normalized value for data in this channel
      for (let i = physicalOffset; i < iterEnd; ++i) {
        Y[i] = scaleData[c] * ((X[i] - mean) / Math.sqrt(variance + epsilon)) + bData[c];
      }
    }
  }

  return output;
}
