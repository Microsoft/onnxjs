// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {OpSet} from '../../opset';

import {WasmBatchNormalization} from './ops/batch-normalization';
import {WasmBinaryOp} from './ops/binary-op';
import {WasmClip} from './ops/clip';
import {WasmConv} from './ops/conv';
import {WasmCumSum} from './ops/cumsum';
import {WasmGemm} from './ops/gemm';
import {WasmInstanceNormalization} from './ops/instance-normalization';
import {WasmMatMul} from './ops/matmul';
import {WasmAveragePool, WasmGlobalAveragePool, WasmGlobalMaxPool, WasmMaxPool} from './ops/pool';
import {WasmSoftmax} from './ops/softmax';
import {WasmSum} from './ops/sum';

export const WASM_OP_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule> = [
  ['Add', '', '7+', () => new WasmBinaryOp(['float32', 'int32'], 'Add')],
  ['And', '', '7+', () => new WasmBinaryOp(['bool'], 'And')],
  ['AveragePool', '', '7-10', () => new WasmAveragePool()],  // TODO: support new attributes for AveragePool-10
  ['BatchNormalization', '', '7+', () => new WasmBatchNormalization()],
  ['Clip', '', '6-10', () => new WasmClip()],
  ['Conv', '', '1+', () => new WasmConv()],
  ['Div', '', '7+', () => new WasmBinaryOp(['float32'], 'Div')],
  ['Gemm', '', '7-10', () => new WasmGemm(false)],
  ['Gemm', '', '11+', () => new WasmGemm(true)],
  ['GlobalAveragePool', '', '1+', () => new WasmGlobalAveragePool()],
  ['GlobalMaxPool', '', '1+', () => new WasmGlobalMaxPool()],
  ['InstanceNormalization', '', '6+', () => new WasmInstanceNormalization()],
  ['MatMul', '', '1+', () => new WasmMatMul()],
  ['MaxPool', '', '1-9', () => new WasmMaxPool()],  // TODO: support new attributes for MaxPool-8 and MaxPool-10
  ['Mul', '', '7+', () => new WasmBinaryOp(['float32', 'int32'], 'Mul')],
  ['Or', '', '7+', () => new WasmBinaryOp(['bool'], 'Or')],
  ['PRelu', '', '7+', () => new WasmBinaryOp(['float32'], 'PRelu')],
  ['Softmax', '', '1+', () => new WasmSoftmax()],
  ['Sub', '', '7+', () => new WasmBinaryOp(['float32', 'int32'], 'Sub')],
  ['Sum', '', '6+', () => new WasmSum()],  // TODO: support multidirectional broadcast for Sum-8
  ['Xor', '', '7+', () => new WasmBinaryOp(['bool'], 'Xor')],
  ['CumSum', '', '11+', () => new WasmCumSum()],
];
