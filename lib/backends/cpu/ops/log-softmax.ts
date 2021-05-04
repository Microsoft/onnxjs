// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {LogSoftmax} from '../../../ops/log-softmax';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';
import {softmax} from './softmax';

export class CpuLogSoftmax extends LogSoftmax {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = logSoftmax(inputs[0], this.axis);
    return [output];
  }
}

export function logSoftmax(x: Tensor, axis: number): Tensor {
  const y = softmax(x, axis);
  const yData = y.numberData;

  const output = new Tensor(x.dims, x.type);
  const data = output.numberData;

  for (let i = 0; i < yData.length; ++i) {
    data[i] = Math.log(yData[i]);
  }

  return output;
}
