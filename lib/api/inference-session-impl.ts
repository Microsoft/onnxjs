// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger} from '../instrument';
import {Session} from '../session';
import {Tensor as InternalTensor} from '../tensor';

import {InferenceSession as InferenceSessionInterface} from './inference-session';
import * as TensorInterface from './tensor';
import {Tensor as ApiTensor} from './tensor-impl';
import * as tensorUtils from './tensor-impl-utils';

type InputType = InferenceSessionInterface.InputType;
type RunOptions = InferenceSessionInterface.RunOptions;
type OutputType = InferenceSessionInterface.OutputType;

export class InferenceSession implements InferenceSessionInterface {
  session: Session;
  constructor(config?: InferenceSessionInterface.Config) {
    const sessionConfig: Session.Config = {};
    sessionConfig.backendHint = config?.backendHint;
    sessionConfig.profiler = config?.profiler;
    if (config?.logger) {
      const loggerConf = config?.logger;
      sessionConfig.logger = {};
      sessionConfig.logger.minimalSeverity = loggerConf.logLevel ? loggerConf.logLevel as Logger.Severity : 'info';
    }
    this.session = new Session(sessionConfig);
  }
  loadModel(uri: string): Promise<void>;
  loadModel(blob: Blob): Promise<void>;
  loadModel(buffer: ArrayBuffer, byteOffset?: number, length?: number): Promise<void>;
  loadModel(buffer: Uint8Array): Promise<void>;
  loadModel(arg0: string|Blob|ArrayBuffer|Uint8Array, byteOffset?: number, length?: number): Promise<void> {
    if (typeof arg0 === 'string') {
      return this.session.loadModel(arg0);
    } else if (typeof Blob !== 'undefined' && (arg0 instanceof Blob)) {
      // create a url from Blob
      const url = URL.createObjectURL(arg0);
      return this.session.loadModel(url);
    } else if (arg0 instanceof ArrayBuffer) {
      // load model from array buffer
      return this.session.loadModel(arg0, byteOffset, length);
    } else if (ArrayBuffer.isView(arg0)) {
      // load model from Uint8array
      return this.session.loadModel(arg0);
    } else {
      throw new Error('Model type is not supported.');
    }
  }

  async run(inputFeed: InputType, options?: RunOptions): Promise<OutputType> {
    let output = new Map<string, InternalTensor>();
    if (inputFeed instanceof Map) {
      const modelInputFeed = new Map<string, InternalTensor>();
      inputFeed.forEach((value: ApiTensor, key: string) => {
        modelInputFeed.set(key, value.internalTensor);
      });
      output = await this.session.run(modelInputFeed);
    } else if (Array.isArray(inputFeed)) {
      const modelInputFeed: InternalTensor[] = [];
      inputFeed.forEach((value) => {
        modelInputFeed.push(value.internalTensor);
      });
      output = await this.session.run(modelInputFeed);
    } else {
      const modelInputFeed = new Map<string, InternalTensor>();
      for (const name in inputFeed) {
        modelInputFeed.set(name, (inputFeed as {readonly [name: string]: ApiTensor})[name].internalTensor);
      }
    }
    const convertedOutput: Map<string, TensorInterface.Tensor> = new Map<string, TensorInterface.Tensor>();
    output.forEach((value, key) => {
      convertedOutput.set(key, tensorUtils.fromInternalTensor(value));
    });
    return convertedOutput;
  }
  startProfiling(): void {
    this.session.startProfiling();
  }
  endProfiling(): void {
    this.session.endProfiling();
  }
}
