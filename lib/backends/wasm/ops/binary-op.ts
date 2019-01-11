// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmBinaryOp extends BinaryOp {
  constructor(typeConstraint: ReadonlyArray<Tensor.DataType>, opType: string, resultType?: Tensor.DataType) {
    super(typeConstraint, opType, resultType);
  }

  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputShape = BroadcastUtil.calcShape(inputs[0].dims, inputs[1].dims, false);
    if (!outputShape) {
      throw new Error('not broadcastable');
    }
    let fun = '';
    let binaryOpType = '';
    switch (this.opType) {
      case 'Add':
        if (inputs[0].type === 'float32') {
          fun = '_add_f32';
          binaryOpType = 'float32InFloat32Out';
        }
        break;
      case 'Sub':
        if (inputs[0].type === 'float32') {
          fun = '_sub_f32';
          binaryOpType = 'float32InFloat32Out';
        }
        break;
      case 'Mul':
        if (inputs[0].type === 'float32') {
          fun = '_mul_f32';
          binaryOpType = 'float32InFloat32Out';
        }
        break;
      case 'Div':
        if (inputs[0].type === 'float32') {
          fun = '_div_f32';
          binaryOpType = 'float32InFloat32Out';
        }
        break;
      case 'PRelu':
        if (inputs[0].type === 'float32') {
          fun = '_prelu_f32';
          binaryOpType = 'float32InFloat32Out';
        }
        break;
      case 'Xor':
        fun = '_xor_';
        binaryOpType = 'boolInBoolOut';
        break;
      case 'Or':
        fun = '_or_';
        binaryOpType = 'boolInBoolOut';
        break;
      case 'And':
        fun = '_and_';
        binaryOpType = 'boolInBoolOut';
        break;
      default:
        throw Error(`unsupported binary op by the Wasm backend`);
    }
    let result: Tensor;
    if (binaryOpType === 'float32InFloat32Out') {
      const outputType = this.resultType ? this.resultType : 'float32';
      result = new Tensor(outputShape, outputType);
      WasmBinding.getInstance().ccall(
          fun, [inputs[0].floatData, 'float32ptr'], [inputs[0].dims.length, 'int32'], [inputs[0].dims, 'int32ptr'],
          [inputs[1].floatData, 'float32ptr'], [inputs[1].dims.length, 'int32'], [inputs[1].dims, 'int32ptr'],
          [result.floatData, 'float32ptr', 'out'], [result.floatData.length, 'int32'], [outputShape.length, 'int32'],
          [outputShape, 'int32ptr']);
    } else if (binaryOpType === 'boolInBoolOut') {
      const outputType = this.resultType ? this.resultType : 'bool';
      result = new Tensor(outputShape, outputType);
      WasmBinding.getInstance().ccall(
          fun, [inputs[0].integerData as Uint8Array, 'boolptr'], [inputs[0].dims.length, 'int32'],
          [inputs[0].dims, 'int32ptr'], [inputs[1].integerData as Uint8Array, 'boolptr'],
          [inputs[1].dims.length, 'int32'], [inputs[1].dims, 'int32ptr'],
          [result.integerData as Uint8Array, 'boolptr', 'out'], [result.integerData.length, 'int32'],
          [outputShape.length, 'int32'], [outputShape, 'int32ptr']);
    } else {
      throw new Error(`Unsupported binary op format. Probably unsupported data types.`);
    }
    return [result];
  }
}
