// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {expect} from 'chai';

import {Backend, InferenceHandler, SessionHandler} from '../../../../lib/backend';
import {WebGLInferenceHandler} from '../../../../lib/backends/webgl/inference-handler';
import {WebGLPack} from '../../../../lib/backends/webgl/ops/pack';
import {WebGLUnpack} from '../../../../lib/backends/webgl/ops/unpack';
import {WebGLContext} from '../../../../lib/backends/webgl/webgl-context';
import {Profiler} from '../../../../lib/instrument';
import {Tensor} from '../../../../lib/tensor';
import {ShapeUtil} from '../../../../lib/util';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

function createAscendingArray(size: number): Float32Array {
  return new Float32Array(Array.from({length: size}, (v, i) => (i + 1)));
}

function createTextureFromArray(
    glContext: WebGLContext, dataArray: Float32Array, type: GLenum, width: number, height: number) {
  const gl = glContext.gl;

  // create the texture
  const texture = gl.createTexture();
  // bind the texture so the following methods effect this texture.
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (glContext.version === 2) {
    const webgl2Gl = gl as WebGL2RenderingContext;
    gl.texImage2D(webgl2Gl.TEXTURE_2D, 0, webgl2Gl.RGBA32F, width, height, 0, webgl2Gl.RGBA, webgl2Gl.FLOAT, dataArray);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, dataArray);
  }

  glContext.checkError();
  return texture;
}

function createArrayFromTexture(
    gl: WebGLRenderingContext, texture: WebGLTexture, width: number, height: number): Float32Array {
  const resultDataBuffer = new Float32Array(width * height * 4);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
      0);  // 0, we aren't using MIPMAPs
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, resultDataBuffer);
  return resultDataBuffer;
}

function getExpectedElementCount(inputShape: number[]): number {
  const rank = inputShape.length;

  // TODO: add rank === 0

  // 1D tensor
  if (rank === 1) {
    if (inputShape[0] % 2) {
      return (inputShape[0] + 1) * 2;
    } else {
      return inputShape[0] * 2;
    }
  }

  // process width
  let inputWidth = 1;
  if (rank >= 2) {
    for (let i = 0; i <= rank - 2; ++i) {
      if (inputShape[i] % 2) {
        inputWidth *= (inputShape[i] + 1);
      } else {
        inputWidth *= inputShape[i];
      }
    }
  }

  // process height
  let inputHeight = inputShape[rank - 1];
  if (inputHeight % 2) {
    inputHeight++;
  }
  return inputWidth * inputHeight;
}

function generateExpected(inputArray: Float32Array, inputShape: number[]): Float32Array {
  const rank = inputShape.length;

  let inputHeight = rank === 1 ? 1 : inputShape[rank - 2];
  const inputWidth = rank === 1 ? inputShape[rank - 1] : inputShape[rank - 1];
  if (rank > 2) {
    // TODO: add check for squeezing other dims
    for (let i = 0; i < rank - 2; ++i) {
      inputHeight *= inputShape[i];
    }
  }
  const width = inputWidth % 2 ? inputWidth + 1 : inputWidth;
  const height = inputHeight % 2 ? inputHeight + 1 : inputHeight;

  const result = new Float32Array(width * height);
  let ii = 0;
  for (let j = 0; j < height; j += 2) {
    for (let i = 0; i < width; i += 2) {
      const index = j * inputWidth + i;
      result[ii++] = inputArray[index];

      if (i + 1 < inputWidth) {
        result[ii++] = inputArray[index + 1];
      } else {
        result[ii++] = 0;
      }

      if ((j + 1) < inputHeight) {
        result[ii++] = inputArray[(j + 1) * inputWidth + i];
      } else {
        result[ii++] = 0;
      }

      if (i + 1 < inputWidth && j + 1 < inputHeight) {
        result[ii++] = inputArray[(j + 1) * inputWidth + i + 1];
      } else {
        result[ii++] = 0;
      }
    }
  }
  return result;
}

describe('#UnitTest# - pack - Tensor pack', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await Backend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });
  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test pack ${JSON.stringify(testData)}`, () => {});
    it(`Test pack kernal ${JSON.stringify(testData)}`, () => {
      const op = new WebGLPack();

      const elementCount = testData.elementCount;
      const inputData = createAscendingArray(elementCount);
      const inputTensorShape = testData.inputShape;
      const outputTextureShape = testData.outputTextureShape;
      // const elementCount = 16;
      // const inputTensorShape = [4, 4];
      // const outputTextureShape = [2, 2];

      // create input data and tensor. The input data will be used to verify if the output tensor
      // contains the same value but possibly different order depending on our packing algorithm.
      // const inputData = createAscendingArray(elementCount);
      const inputTensor = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // compile shader code
      const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;
      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      // const result = runData.outputTextureData;
      const resultTexture = runData.outputTextureData.texture;
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      const resultDataBuffer = createArrayFromTexture(gl, resultTexture, outputTextureShape[1], outputTextureShape[0]);

      // // test only: download input texture to cpu
      // // const inputTextData = webglInferenceHandler.getTextureData(inputTensor.dataId);
      // // const aa = createArrayFromTexture(gl, inputTextData!.texture, 4, 4);
      // // console.log(aa);

      // verify result.
      // TODO: add more verifications including output value and order
      // TODO: also different input dimensions for different pack code path
      expect(resultDataBuffer).to.not.equal(null);

      const outputElementCount = getExpectedElementCount(testData.inputShape);

      expect(resultDataBuffer).to.have.lengthOf(outputElementCount);
      // TODO: enable verifications after code integration.
      for (let i = 0; i < resultDataBuffer.length; ++i) {
        console.log(resultDataBuffer[i]);
      }

      const expectedOutput = generateExpected(inputData, testData.inputShape);
      // expect(resultDataBuffer).to.deep.equal(new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10,
      // 13, 14, 11, 12, 15, 16]));
      // console.log(expectedOutput);
      for (let i = 0; i < expectedOutput.length; ++i) {
        console.log('expected: ', expectedOutput[i]);
      }
      expect(resultDataBuffer).to.deep.equal(expectedOutput);
    });
  }

  it(`Test unpack kernal `, () => {
    const op = new WebGLUnpack();

    const elementCount = 16;
    const inputTensorShape = [4, 4];
    const inputTextureShape = [2, 2];
    const outputTensorShape = [4, 4];

    // create input data and tensor. The input data will be used to verify if the output tensor contains the
    // same value but possibly different order depending on our packing algorithm.
    const inputData = createAscendingArray(elementCount);
    const inputTensor = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

    const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

    // manually creat packed texture from inputTensor, and insert in cache
    const gl = webglInferenceHandler.session.textureManager.glContext.gl;
    webglInferenceHandler.session.textureManager.glContext.checkError();
    const webglTexture = createTextureFromArray(
        webglInferenceHandler.session.textureManager.glContext, inputData, gl.RGBA, inputTextureShape[0],
        inputTextureShape[1]);
    webglInferenceHandler.session.textureManager.glContext.checkError();
    const packedShape = inputTextureShape;
    const textureData = {
      width: inputTextureShape[0],
      height: inputTextureShape[1],
      channels: 4 as const,
      isPacked: true,
      shape: packedShape,
      strides: ShapeUtil.computeStrides(packedShape),
      unpackedShape: outputTensorShape,
      tensor: inputTensor,
      texture: webglTexture!
    };

    webglInferenceHandler.setTextureData(inputTensor.dataId, textureData);

    // compile shader code
    const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);

    const artifact = webglInferenceHandler.session.programManager.build(programInfo);
    webglInferenceHandler.session.programManager.setArtifact(op, artifact);

    // run kernal and get output
    const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
    webglInferenceHandler.session.programManager.run(artifact, runData);
    const result = runData.outputTextureData.tensor.data;

    const resultDataBuffer = createArrayFromTexture(gl, webglTexture!, inputTextureShape[0], inputTextureShape[1]);

    webglInferenceHandler.session.textureManager.glContext.checkError();
    // verify result.
    expect(result).to.not.equal(null);
    expect(result).to.have.lengthOf(elementCount);
    // TODO: enable verifications after code integration.
    expect(resultDataBuffer).to.deep.equal(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));

    console.log(result);

    expect(result).to.deep.equal(new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]));
  });
});
interface TestData {
  elementCount: number;
  inputShape: number[];
  outputTextureShape: number[];
}
function getTestData(): TestData[] {
  return [
    // test 2D tensor
    {elementCount: 16, inputShape: [4, 4], outputTextureShape: [2, 2]},
    {elementCount: 16, inputShape: [2, 8], outputTextureShape: [1, 4]},
    {elementCount: 16, inputShape: [8, 2], outputTextureShape: [4, 1]},
    {elementCount: 15, inputShape: [3, 5], outputTextureShape: [2, 3]},
    {elementCount: 18, inputShape: [3, 6], outputTextureShape: [2, 3]},
    {elementCount: 10, inputShape: [2, 5], outputTextureShape: [1, 3]},
    {elementCount: 6, inputShape: [1, 6], outputTextureShape: [1, 3]},
    {elementCount: 6, inputShape: [6, 1], outputTextureShape: [3, 1]},
    {elementCount: 5, inputShape: [5, 1], outputTextureShape: [3, 1]},
    {elementCount: 5, inputShape: [1, 5], outputTextureShape: [1, 3]},

    // test 1D tensor
    {elementCount: 16, inputShape: [16], outputTextureShape: [1, 8]},
    {elementCount: 9, inputShape: [9], outputTextureShape: [1, 5]},

    // // TODO: not working yet
    // {elementCount: 16, inputShape: [2, 2, 4], outputTextureShape: [2, 2]},
    // {elementCount: 24, inputShape: [2, 3, 4], outputTextureShape: [4, 2]},
    // ADD empty tensor
  ];
}
