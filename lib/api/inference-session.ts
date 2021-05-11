// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from './tensor';

/**
 * represent a runtime instance of an ONNX model
 */
export interface InferenceSession {
  //#region loadModel

  /**
   * load an ONNX model asynchronously
   * @param uri the URI of the model to load
   */
  loadModel(uri: string): Promise<void>;
  /**
   * load an ONNX model
   * @param blob a Blob object representation of an ONNX model
   */
  loadModel(blob: Blob): Promise<void>;
  /**
   * load an ONNX model
   * @param buffer an ArrayBuffer representation of an ONNX model
   */
  loadModel(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Promise<void>;
  /**
   * load an ONNX model
   * @param buffer an Uint8Array representation of an ONNX model
   */
  loadModel(buffer: Uint8Array): Promise<void>;

  //#endregion loadModel

  /**
   * execute the model asynchronously with the given inputs, output names and options
   * @param inputs representation of the model input. It can be a string-to-tensor Map object or a plain object,
   *     with all required inputs present with their names as keys; it can be a tensor array as well, with input
   *     tensors inside in the order defined by the model.
   * @param options Optional. A set of options that controls the behavior of model inference
   * @returns a Promise object representing the result of the execution. Resolve to a string-to-tensor Map object
   *     for the model output, or reject to any runtime error.
   */
  run(inputs: InferenceSession.InputType, options?: InferenceSession.RunOptions): Promise<InferenceSession.OutputType>;

  /**
   * start profiling for the session
   */
  startProfiling(): void;
  /**
   * end profiling for the session and flush data
   */
  endProfiling(): void;
}

export declare namespace InferenceSession {
  type TensorsMapType = ReadonlyMap<string, Tensor>;
  type TensorsIndexType = {readonly [name: string]: Tensor};
  type TensorsArrayType = ReadonlyArray<Tensor>;
  type InputType = TensorsMapType|TensorsIndexType|TensorsArrayType;
  type OutputType = TensorsMapType;

  export namespace Config {
    /**
     * represent the configuration of the profiler that used in an inference session
     */
    export interface Profiler {
      /**
       * the max number of events to be recorded
       */
      maxNumberEvents?: number;
      /**
       * the maximum size of a batch to flush
       */
      flushBatchSize?: number;
      /**
       * the maximum interval in milliseconds to flush
       */
      flushIntervalInMilliseconds?: number;
    }

    /**
     * represent the configuration of the logger that is used in an inference session
     */
    export interface Logger {
      /**
       * the log level
       */
      logLevel?: string;
    }
  }

  /**
   * configuration for creating a new inference session
   */
  export interface Config {
    /**
     * specify a hint of the preferred backend. If not set, the backend will be determined by the platform and
     * environment.
     */
    backendHint?: string;

    /**
     * specify the configuration of the profiler that used in an inference session
     */
    profiler?: Config.Profiler;

    /**
     * specify the configuration of the logger that used in an inference session
     */
    logger?: Config.Logger;
  }

  /**
   * options for running inference
   */
  export interface RunOptions {
    /**
     * represent a list of output names as an array of string. This must be a subset of the output list defined by the
     * model. If not specified, use the model's output list.
     */
    outputNames?: ReadonlyArray<string>;
  }
}

export interface InferenceSessionConstructor {
  /**
   * construct a new inference session
   * @param config specify configuration for creating a new inference session
   */
  new(config?: InferenceSession.Config): InferenceSession;
}

import * as InferenceSessionImpl from './inference-session-impl';
export const InferenceSession: InferenceSessionConstructor = InferenceSessionImpl.InferenceSession;
