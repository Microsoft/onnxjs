// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {Einsum} from '../../../ops/einsum';
import {Tensor} from '../../../tensor';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmEinsum extends Einsum {
  initialize(attributes: Attribute): void {
    super.initialize(attributes);
    if (this.inputs.length > 2) {
      throw new Error('Wasm implementation of Einsum currently supports at most 2 inputs');
    }
  }

  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const {outputShape, sizes, outputIndices, inputIndices} = this.prepareRun(inputs);

    const y = new Tensor(outputShape, inputs[0].type);

    if (inputs.length === 2) {
      WasmBinding.getInstance().ccall(
          '_einsum_f32',
          [inputs[0].floatData, 'float32ptr'],
          [inputs[1].floatData, 'float32ptr'],
          [y.floatData, 'float32ptr', 'inout'],
          [sizes, 'int32ptr'],
          [sizes.length, 'int32'],
          [outputIndices, 'int32ptr'],
          [outputIndices.length, 'int32'],
          [inputIndices[0], 'int32ptr'],
          [inputIndices[0].length, 'int32'],
          [inputIndices[1], 'int32ptr'],
          [inputIndices[2].length, 'int32'],
      );
    } else {
      WasmBinding.getInstance().ccall(
          '_einsum_single_f32',
          [inputs[0].floatData, 'float32ptr'],
          [y.floatData, 'float32ptr', 'inout'],
          [sizes, 'int32ptr'],
          [sizes.length, 'int32'],
          [outputIndices, 'int32ptr'],
          [outputIndices.length, 'int32'],
          [inputIndices[0], 'int32ptr'],
          [inputIndices[1].length, 'int32'],
      );
    }

    return [y];
  }

  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || (inputs.length > 1 && inputs[1].type !== 'float32')) {
      return false;
    }

    return super.checkInputTypes(inputs);
  }
}
