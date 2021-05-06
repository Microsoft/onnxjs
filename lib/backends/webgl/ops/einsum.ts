// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Einsum} from '../../../ops/einsum';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

import {ShapeUtil} from './../../../util';

const samplerNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

export class WebGLEinsum extends Einsum implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const {outputShape, sizes, outputIndices, inputIndices} = this.prepareRun(inputs);

    const sumDims = [];
    const sumDimSizes = [];
    for (let i = 0; i < sizes.length; i++) {
      if (outputIndices.indexOf(i) === -1) {
        sumDims.push(i);
        sumDimSizes.push(sizes[i]);
      }
    }
    const sumSize = ShapeUtil.size(sumDimSizes);

    let rank = outputShape.length;
    // Webgl doesnt like 0 length arrays
    if (rank === 0) {
      rank = 1;
    }

    const initIndex1 = outputIndices.map((x, i) => `index[${x}] = indices[${i}];`).join('\n');
    const initIndex2 = sumDims.map(x => `index[${x}] = 0;`).join('\n');

    const findInputValues = inputs.map((_, i) => this.buildFindInputValueScript(i, inputIndices[i])).join('\n');

    const incrementIndex = this.buildIncrementIndexScript(sumDims, sumDimSizes);

    const shaderSource = `
      float process(int indices[${rank}]) {
        float value = 0.0;

        int index[${sizes.length}];
        ${initIndex1}
        ${initIndex2}

        int i = 0;
        while(i < ${sumSize}) {
          float add = 1.0;

          ${findInputValues}

          value += add;

          ${incrementIndex}
          i++;
        }

        return value;
      }`;
    const inputLayouts = inputs.map(t => inferenceHandler.getOrCreateTextureLayout(t));
    return {
      inputLayouts,
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: samplerNames.slice(0, inputs.length),
      shaderSource,
    };
  }

  buildFindInputValueScript(i: number, indices: number[]): string {
    const initInputIndex = indices.map((ix, indiceNum) => `input${i}Index[${indiceNum}] = index[${ix}];`).join('\n');

    const script = `int input${i}Index[${indices.length}];
      ${initInputIndex}
      add *= _${samplerNames[i]}(input${i}Index);`;

    return script;
  }

  buildIncrementIndexScript(sumDims: number[], sumDimSizes: number[]): string {
    let script = '';
    for (let i = 0; i < sumDims.length; i++) {
      script += `
        index[${sumDims[i]}] += 1;
        if (index[${sumDims[i]}] >= ${sumDimSizes[i]}) {
          index[${sumDims[i]}] = 0;
      `;
    }
    for (let i = 0; i < sumDims.length; i++) {
      script += '}\n';
    }

    return script;
  }

  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((v, i) => inferenceHandler.getOrCreateTextureData(v, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
