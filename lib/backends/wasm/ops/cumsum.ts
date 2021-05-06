// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {CumSum} from '../../../ops/cumsum';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmCumSum extends CumSum {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const ax = inputs[1].integerData[0];

    const outputSize = ShapeUtil.size(inputs[0].dims);
    const resultData = new Float32Array(outputSize);
    WasmBinding.getInstance().ccall(
        '_cumsum_f32', [inputs[0].floatData, 'float32ptr'], [inputs[0].dims, 'int32ptr'],
        [inputs[0].dims.length, 'int32'], [ax, 'int32'], [this.exclusive, 'bool'], [this.reverse, 'bool'],
        [resultData, 'float32ptr', 'out']);

    const result = new Tensor(inputs[0].dims, inputs[0].type);
    result.floatData.set(resultData);
    return [result];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32') {
      return false;
    }

    return true;
  }
}
