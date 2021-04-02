// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {expect} from 'chai';

// import {Attribute} from '../../../../lib/attribute';
import {Backend, InferenceHandler, SessionHandler} from '../../../../lib/backend';
import {WebGLBackend} from '../../../../lib/backends/backend-webgl';
import {WebGLInferenceHandler} from '../../../../lib/backends/webgl/inference-handler';
import {WebGLReshapePacked} from '../../../../lib/backends/webgl/ops/reshape-packed';
import {Profiler} from '../../../../lib/instrument';
import {Tensor} from '../../../../lib/tensor';
// import {ShapeUtil} from '../../../../lib/util';

import {createAscendingArray} from './test_utils';
// import {createTextureFromArray} from './test_utils';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - reshape - packed', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await Backend('webgl');
    // Explicitly set to true to trigger packed version
    (backend as WebGLBackend).pack = true;
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });

  // Set it back to false, apparently this state is sticky throughout all the tests running in same browser session..
  after('Resetting Context', () => {
    (backend as WebGLBackend).pack = false;
  });

  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test reshape ${JSON.stringify(testData)}`, () => {});
    it(`Test packed reshape kernel ${JSON.stringify(testData.outputShape)}`, () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // TODO support WebGl 1.0
      if (webglInferenceHandler.session.textureManager.glContext.version === 1) {
        console.log('Running packed concat with webgl1 is not supported. Skipping.');
        return;
      }

      const op = new WebGLReshapePacked();

      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;
      // const inputTextureShape = testData.inputTextureShape;
      const outputTensorShape = testData.outputShape;

      // create input data and tensor.
      const inputData = createAscendingArray(elementCount);
      const inputTensorA = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // create shape data tensor
      const inputTensorB =
          new Tensor([outputTensorShape.length], 'int32', undefined, undefined, new Int32Array(outputTensorShape));

      // manually creat packed texture from inputTensor, and insert in cache
      // // const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      // // webglInferenceHandler.session.textureManager.glContext.checkError();
      // // const webglTextureA = createTextureFromArray(
      // //     webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput :
      // inputData,
      // //     gl.RGBA, inputTextureShape[0], inputTextureShape[1]);
      // // // const webglTextureB = createTextureFromArray(
      // // //     webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput :
      // inputData,
      // // //     gl.RGBA, inputTextureShape[0], inputTextureShape[1]);

      // // webglInferenceHandler.session.textureManager.glContext.checkError();
      // // const packedShape = inputTextureShape;
      // // const textureDataA = {
      // //   width: inputTextureShape[0],
      // //   height: inputTextureShape[1],
      // //   channels: 4 as const,
      // //   isPacked: true,
      // //   shape: packedShape,
      // //   strides: ShapeUtil.computeStrides(packedShape),
      // //   unpackedShape: outputTensorShape,
      // //   tensor: inputTensorA,
      // //   texture: webglTextureA!
      // // };
      // // // const textureDataB = {
      // // //   width: inputTextureShape[0],
      // // //   height: inputTextureShape[1],
      // // //   channels: 4 as const,
      // // //   isPacked: true,
      // // //   shape: packedShape,
      // // //   strides: ShapeUtil.computeStrides(packedShape),
      // // //   unpackedShape: outputTensorShape,
      // // //   tensor: inputTensorB,
      // // //   texture: webglTextureB!
      // // // };

      // // webglInferenceHandler.setTextureData(inputTensorA.dataId, textureDataA);
      // webglInferenceHandler.setTextureData(inputTensorB.dataId, textureDataB);

      // compile shader code
      const programInfo =
          op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensorA, inputTensorB]);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      // const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensorA, inputTensorB]);
      const resultTensor = webglInferenceHandler.run(op, [inputTensorA, inputTensorB]);
      const result = resultTensor[0].data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      // const expectedOutput = testData.expectedOutput;
      expect(result).to.not.equal(null);

      expect(result).to.have.lengthOf(elementCount);

      expect(result).to.deep.equal(inputData);
    });
  }
});
interface TestData {
  elementCount: number;
  axis: number;
  inputShape: number[];
  outputShape: number[];
  inputTextureShape: number[];
  outputTextureShape: number[];
  expectedOutput: Float32Array;
  // If empty, the test will use auto-generated data.
  rawInput?: Float32Array;
}
function getTestData(): TestData[] {
  return [
    // test 2D tensor
    {
      elementCount: 16,
      axis: 0,
      inputShape: [4, 4],
      outputShape: [2, 8],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16
      ]),
    },
    {
      elementCount: 16,
      axis: 1,
      inputShape: [4, 4],
      outputShape: [1, 16],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 2],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 1, 2, 5, 6, 3, 4, 7, 8, 3, 4, 7, 8, 9, 10, 13, 14, 9, 10, 13, 14, 11, 12, 15, 16, 11, 12, 15, 16
      ]),
    },
    {
      elementCount: 8,
      axis: 0,
      inputShape: [2, 4],
      outputShape: [4, 2],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([
        1,
        2,
        5,
        6,
        3,
        4,
        7,
        8,
        1,
        2,
        5,
        6,
        3,
        4,
        7,
        8,
      ]),
    },
    {
      elementCount: 8,
      axis: 1,
      inputShape: [2, 4],
      outputShape: [1, 8],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1,
        2,
        5,
        6,
        1,
        2,
        5,
        6,
        3,
        4,
        7,
        8,
        3,
        4,
        7,
        8,
      ]),
    },
    {
      elementCount: 6,
      axis: 0,
      inputShape: [2, 3],
      outputShape: [1, 6],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
      rawInput: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0])
    },
    {
      elementCount: 6,
      axis: 1,
      inputShape: [2, 3],
      outputShape: [3, 2],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]),
      rawInput: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0])
    },

    // test 3d tensor
    {
      elementCount: 16,
      axis: 0,
      inputShape: [2, 2, 4],
      outputShape: [4, 2, 2],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16
      ])
    },
    {
      elementCount: 16,
      axis: 1,
      inputShape: [2, 2, 4],
      outputShape: [2, 4, 2],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 2],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 9, 10, 13, 14, 11, 12, 15, 16
      ])
    },
    {
      elementCount: 16,
      axis: 2,
      inputShape: [2, 2, 4],
      outputShape: [1, 1, 2, 8],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 1, 2, 5, 6, 3, 4, 7, 8, 3, 4, 7, 8, 9, 10, 13, 14, 9, 10, 13, 14, 11, 12, 15, 16, 11, 12, 15, 16
      ])
    },

    // test 4d tensor
    {
      elementCount: 32,
      axis: 0,
      inputShape: [2, 2, 2, 4],
      outputShape: [4, 2, 2, 2],
      inputTextureShape: [2, 4],
      outputTextureShape: [2, 8],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 17, 18, 21, 22, 19, 20,
        23, 24, 25, 26, 29, 30, 27, 28, 31, 32, 1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14,
        11, 12, 15, 16, 17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },
    {
      elementCount: 32,
      axis: 1,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 4, 2, 2],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 1,  2,  5,  6,  3,  4,
        7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 25, 26, 29, 30, 27, 28, 31, 32, 25, 26, 29, 30,
        27, 28, 31, 32, 25, 26, 29, 30, 27, 28, 31, 32, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },

    {
      elementCount: 32,
      axis: 2,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 2, 4, 2],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  1,  2,  5,  6,  3,  4,  7,  8,  17, 18, 21, 22, 19, 20,
        23, 24, 17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32, 25, 26, 29, 30,
        27, 28, 31, 32, 25, 26, 29, 30, 27, 28, 31, 32, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },
    {
      elementCount: 32,
      axis: 3,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 1, 4, 4],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  1,  2,  5,  6,  3,  4,  7,  8,  3,  4,  7,  8,  17, 18, 21, 22, 17, 18,
        21, 22, 19, 20, 23, 24, 19, 20, 23, 24, 25, 26, 29, 30, 25, 26, 29, 30, 27, 28, 31, 32,
        27, 28, 31, 32, 25, 26, 29, 30, 25, 26, 29, 30, 27, 28, 31, 32, 27, 28, 31, 32
      ])
    },
  ];
}