// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend, InferenceHandler, SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {ProtoUtil} from '../../util';
import {getInstance} from '../../wasm-binding-core';

import {WasmInferenceHandler} from './inference-handler';

export class WasmSessionHandler implements SessionHandler {
  constructor(readonly backend: Backend, readonly context: Session.Context, fallbackToCpuOps: boolean) {}
  resolve(node: Graph.Node, opsets: readonly OpSet[], graph: Graph): Operator {
    throw new Error('Method not implemented.');
  }

  createInferenceHandler(): InferenceHandler {
    return new WasmInferenceHandler(this, this.context.profiler);
  }

  // vNEXT latest:
  ortInit: boolean;
  sessionHandle: number;

  inputNames: string[];
  inputNamesUTF8Encoded: number[];
  outputNames: string[];
  outputNamesUTF8Encoded: number[];

  loadModel(model: Uint8Array) {
    const wasm = getInstance();
    if (!this.ortInit) {
      wasm._OrtInit(4, 0);
      this.ortInit = true;
    }

    const modelDataOffset = wasm._malloc(model.byteLength);
    try {
      wasm.HEAPU8.set(model, modelDataOffset);
      this.sessionHandle = wasm._OrtCreateSession(modelDataOffset, model.byteLength);
    } finally {
      wasm._free(modelDataOffset);
    }

    const inputCount = wasm._OrtGetInputCount(this.sessionHandle);
    const outputCount = wasm._OrtGetOutputCount(this.sessionHandle);

    this.inputNames = [];
    this.inputNamesUTF8Encoded = [];
    this.outputNames = [];
    this.outputNamesUTF8Encoded = [];
    for (let i = 0; i < inputCount; i++) {
      const name = wasm._OrtGetInputName(this.sessionHandle, i);
      this.inputNamesUTF8Encoded.push(name);
      this.inputNames.push(wasm.UTF8ToString(name));
    }
    for (let i = 0; i < outputCount; i++) {
      const name = wasm._OrtGetOutputName(this.sessionHandle, i);
      this.outputNamesUTF8Encoded.push(name);
      this.outputNames.push(wasm.UTF8ToString(name));
    }
  }

  run(inputs: Map<string, Tensor>|Tensor[]): Map<string, Tensor> {
    const wasm = getInstance();

    let inputIndices: number[] = [];
    if (!Array.isArray(inputs)) {
      const inputArray: Tensor[] = [];
      inputs.forEach((tensor, name) => {
        const index = this.inputNames.indexOf(name);
        if (index === -1) {
          throw new Error(`invalid input '${name}'`);
        }
        inputArray.push(tensor);
        inputIndices.push(index);
      });
      inputs = inputArray;
    } else {
      inputIndices = inputs.map((t, i) => i);
    }

    const inputCount = inputs.length;
    const outputCount = this.outputNames.length;

    const inputValues: number[] = [];
    const inputDataOffsets: number[] = [];
    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const data = inputs[i].numberData;
      const dataOffset = wasm._malloc(data.byteLength);
      inputDataOffsets.push(dataOffset);
      wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), dataOffset);

      const dims = inputs[i].dims;

      const stack = wasm.stackSave();
      const dimsOffset = wasm.stackAlloc(4 * dims.length);
      try {
        let dimIndex = dimsOffset / 4;
        dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
        const tensor = wasm._OrtCreateTensor(
            ProtoUtil.tensorDataTypeStringToEnum(inputs[i].type), dataOffset, data.byteLength, dimsOffset, dims.length);
        inputValues.push(tensor);
      } finally {
        wasm.stackRestore(stack);
      }
    }

    const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
    const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
    const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
    const outputNamesOffset = wasm.stackAlloc(outputCount * 4);
    try {
      let inputValuesIndex = inputValuesOffset / 4;
      let inputNamesIndex = inputNamesOffset / 4;
      let outputValuesIndex = outputValuesOffset / 4;
      let outputNamesIndex = outputNamesOffset / 4;
      for (let i = 0; i < inputCount; i++) {
        wasm.HEAP32[inputValuesIndex++] = inputValues[i];
        wasm.HEAP32[inputNamesIndex++] = this.inputNamesUTF8Encoded[i];
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAP32[outputValuesIndex++] = 0;
        wasm.HEAP32[outputNamesIndex++] = this.outputNamesUTF8Encoded[i];
      }

      wasm._OrtRun(
          this.sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset);

      const output = new Map<string, Tensor>();

      for (let i = 0; i < outputCount; i++) {
        const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];

        const beforeGetTensorDataStack = wasm.stackSave();
        // stack allocate 4 pointer value
        const tensorDataOffset = wasm.stackAlloc(4 * 4);
        try {
          wasm._OrtGetTensorData(
              tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
          let tensorDataIndex = tensorDataOffset / 4;
          const dataType = wasm.HEAPU32[tensorDataIndex++];
          const dataOffset = wasm.HEAPU32[tensorDataIndex++];
          const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
          const dimsLength = wasm.HEAPU32[tensorDataIndex++];
          const dims = [];
          for (let i = 0; i < dimsLength; i++) {
            dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
          }
          wasm._OrtFree(dimsOffset);

          const t = new Tensor(dims, ProtoUtil.tensorDataTypeFromProto(dataType));
          new Uint8Array(t.numberData.buffer, t.numberData.byteOffset, t.numberData.byteLength)
              .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + t.numberData.byteLength));
          output.set(this.outputNames[i], t);
        } finally {
          wasm.stackRestore(beforeGetTensorDataStack);
        }

        wasm._OrtReleaseTensor(tensor);
      }

      inputValues.forEach(t => wasm._OrtReleaseTensor(t));
      inputDataOffsets.forEach(i => wasm._free(i));

      return output;
    } finally {
      wasm.stackRestore(beforeRunStack);
    }
  }
  dispose() {
    const wasm = getInstance();
    if (this.inputNamesUTF8Encoded) {
      this.inputNamesUTF8Encoded.forEach(str => wasm._OrtFree(str));
      this.inputNamesUTF8Encoded = [];
    }
    if (this.outputNamesUTF8Encoded) {
      this.outputNamesUTF8Encoded.forEach(str => wasm._OrtFree(str));
      this.outputNamesUTF8Encoded = [];
    }
    if (this.sessionHandle) {
      wasm._OrtReleaseSession(this.sessionHandle);
      this.sessionHandle = 0;
    }
  }
}
