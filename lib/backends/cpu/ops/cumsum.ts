// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {CumSum} from '../../../ops/cumsum';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuCumSum extends CumSum {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const ax = inputs[1].integerData[0];
    const output = cumsum(inputs[0], ax, this.exclusive, this.reverse);
    return [output];
  }
}

export function cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean) {
  const y = new Tensor(x.dims, x.type);

  if (axis < 0) {
    axis = y.dims.length + axis;
  }

  const index: number[] = new Array(y.dims.length).fill(0);
  let i = 0;

  if (reverse) {
    i = y.data.length - 1;
    for (let j = 0; j < y.dims.length; j++) {
      index[j] = y.dims[j] - 1;
    }
  }

  while (i < y.data.length && i >= 0) {
    const prevIndex = updateIndex(index, axis, index[axis] + (reverse ? 1 : -1));

    const start = (index[axis] === 0 && !reverse) || (index[axis] === (y.dims[axis] - 1) && reverse);

    if (start && !exclusive) {
      y.set(index, x.get(index));
    } else if (start && exclusive) {
      y.set(index, 0);
    } else if (!start && !exclusive) {
      const prevValue = y.get(prevIndex) as number;
      y.set(index, prevValue + (x.get(index) as number));
    } else {
      const prevValue = y.get(prevIndex) as number;
      y.set(index, prevValue + (x.get(prevIndex) as number));
    }

    if (reverse) {
      ShapeUtil.decrementIndex(index, x.dims);
      i--;
    } else {
      ShapeUtil.incrementIndex(index, x.dims);
      i++;
    }
  }

  return y;
}

function updateIndex(index: number[], axis: number, value: number) {
  const result = index.slice();
  result[axis] = value;
  return result;
}
