// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {CpuBackend} from '../backends/backend-cpu';
import {WasmBackend} from '../backends/backend-wasm';
import {WebGLBackend} from '../backends/backend-webgl';

import {Environment} from './env';
import {envImpl} from './env-impl';
import {Backend} from './onnx';
import {MixedBackend} from '../backends/backend-mixed';

export * from './env';
export * from './onnx';
export * from './tensor';
export * from './inference-session';

export const backend: Backend = {
  cpu: new CpuBackend(),
  wasm: new WasmBackend(),
  webgl: new WebGLBackend(),
  mixed: new MixedBackend()
};

export const ENV: Environment = envImpl;
