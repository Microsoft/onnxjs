// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {CumSum} from '../../../ops/cumsum';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLCumSum extends CumSum implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const ax = inputs[1].integerData[0];
    const rank = inputs[0].dims.length;
    const dims = inputs[0].dims;

    const startIx = this.reverse ? (dims[ax] - 1) : 0;
    const comp = this.exclusive ? '' : '=';
    const condition = this.reverse ? `k >${comp} endIx` : `k <${comp} endIx`;
    const update = this.reverse ? 'k--' : 'k++';

    const shaderSource = `
      float process(int indices[${rank}]) {
        float value = 0.0;
        int endIx = indices[${ax}];
        for (int k=${startIx}; ${condition}; ${update}) {
          indices[${ax}] = k;
          value += _A(indices);
        }
        return value;
      }`;
    const inputLayouts = [inferenceHandler.getOrCreateTextureLayout(inputs[0])];
    return {
      inputLayouts,
      outputLayout: inferenceHandler.createTextureLayoutFromShape(inputs[0].dims),
      samplers: ['A'],
      shaderSource,
    };
  }

  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [inferenceHandler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
