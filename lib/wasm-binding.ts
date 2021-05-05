// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger} from './instrument';
import * as bindingCore from './wasm-binding-core';
import {WasmCallArgument} from './wasm-binding-core';

export {WasmCallArgument} from './wasm-binding-core';

interface PerformanceData extends bindingCore.PerformanceData {
  startTimeWorker?: number;
  endTimeWorker?: number;
}

let workers: Worker[];
let WORKER_NUMBER: number;

// complete callback after
type CompleteCallbackType = (buffer: ArrayBuffer, perfData: PerformanceData) => void;
let completeCallbacks: CompleteCallbackType[][];

let initialized = false;
let initializing = false;

/**
 * initialize the WASM instance.
 *
 * this function should be called before any other calls to methods in WasmBinding.
 */
export function init(numWorkers: number, initTimeout: number): Promise<void> {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error(`multiple calls to 'init()' detected.`);
  }

  initializing = true;
  return new Promise<void>((resolve, reject) => {
    // the timeout ID that used as a guard for rejecting binding init.
    // we set the type of this variable to unknown because the return type of function 'setTimeout' is different
    // in node.js (type Timeout) and browser (number)
    let waitForBindingInitTimeoutId: unknown;
    const clearWaitForBindingInit = () => {
      if (waitForBindingInitTimeoutId !== undefined) {
        // tslint:disable-next-line:no-any
        clearTimeout(waitForBindingInitTimeoutId as any);
        waitForBindingInitTimeoutId = undefined;
      }
    };

    const onFulfilled = () => {
      clearWaitForBindingInit();
      resolve();
      initializing = false;
      initialized = true;
    };
    const onRejected = (err: unknown) => {
      clearWaitForBindingInit();
      reject(err);
      initializing = false;
    };

    const bindingInitTask = bindingCore.init(numWorkers);
    // a promise that gets rejected after 5s to work around the fact that
    // there is an unrejected promise in the wasm glue logic file when
    // it has some problem instantiating the wasm file
    const rejectAfterTimeOutPromise = new Promise((resolve, reject) => {
      waitForBindingInitTimeoutId = setTimeout(() => {
        reject('Wasm init promise failed to be resolved within set timeout');
      }, initTimeout);
    });

    // user requests positive number of workers
    if (numWorkers > 0) {
      Logger.verbose('WebAssembly-Workers', `User has requested ${numWorkers} Workers.`);
      // check if environment supports usage of workers
      if (areWebWorkersSupported()) {
        Logger.verbose(
            'WebAssembly-Workers', `Environment supports usage of Workers. Will spawn ${numWorkers} Workers`);
        // TODO: This code will be replaced to control the number of WebAssembly threads later.
        WORKER_NUMBER = 0;
      } else {
        Logger.error('WebAssembly-Workers', 'Environment does not support usage of Workers. Will not spawn workers.');
        WORKER_NUMBER = 0;
      }
    }

    // user explicitly disables workers
    else {
      Logger.verbose('WebAssembly-Workers', 'User has disabled usage of Workers. Will not spawn workers.');
      WORKER_NUMBER = 0;
    }

    const workerInitTasks = new Array<Promise<void>>(WORKER_NUMBER);
    workers = new Array(WORKER_NUMBER);
    completeCallbacks = new Array(WORKER_NUMBER);

    for (let workerId = 0; workerId < WORKER_NUMBER; workerId++) {
      const workerInitTask = new Promise<void>((resolveWorkerInit, rejectWorkerInit) => {
        // tslint:disable-next-line
        const worker = require('worker-loader?filename=onnx-worker.js!./worker/worker-main').default() as Worker;
        workers[workerId] = worker;
        completeCallbacks[workerId] = [];
        worker.onerror = e => {
          Logger.error('WebAssembly-Workers', `worker-${workerId} ERR: ${e}`);
          if (initialized) {
            // TODO: we need error-handling logic
          } else {
            rejectWorkerInit();
          }
        };
        worker.onmessage = e => {
          if (e && e.data && e.data.type) {
            if (e.data.type === 'init-success') {
              resolveWorkerInit();
            } else if (e.data.type === 'ccall') {
              const perfData = e.data.perfData as PerformanceData;
              completeCallbacks[workerId].shift()!(e.data.buffer as ArrayBuffer, perfData);
            } else {
              throw new Error(`unknown message type from worker: ${e.data.type}`);
            }
          } else {
            throw new Error(`missing message type from worker`);
          }
        };
      });
      workerInitTasks[workerId] = workerInitTask;
    }

    // TODO: Fix this hack to work-around the fact that the Wasm binding instantiate promise
    // is unrejected incase there is a fatal exception (missing wasm file for example)
    // we impose a healthy timeout (should not affect core framework performance)
    Promise.race([bindingInitTask, rejectAfterTimeOutPromise])
        .then(
            () => {
              // Wasm init promise resolved
              Promise.all(workerInitTasks)
                  .then(
                      // Wasm AND Web-worker init promises resolved. SUCCESS!!
                      onFulfilled,
                      // Wasm init promise resolved. Some (or all) web-worker init promises failed to be resolved.
                      // PARTIAL SUCCESS. Use Wasm backend with no web-workers (best-effort).
                      (e) => {
                        Logger.warning(
                            'WebAssembly-Workers',
                            `Unable to get all requested workers initialized. Will use Wasm backend with 0 workers. ERR: ${
                                e}`);
                        // TODO: need house-keeping logic to cull exisitng successfully initialized workers
                        WORKER_NUMBER = 0;
                        onFulfilled();
                      });
            },
            // Wasm init promise failed to be resolved. COMPLETE FAILURE. Reject this init promise.
            onRejected);
  });
}

// Extending the WasmBinding class to deal with web-worker specific logic here
export class WasmBinding extends bindingCore.WasmBinding {
  protected static instance?: WasmBinding;
  static getInstance(): WasmBinding {
    if (!WasmBinding.instance) {
      WasmBinding.instance = new WasmBinding();
    }
    return WasmBinding.instance;
  }
  static get workerNumber() {
    return WORKER_NUMBER;
  }
  ccallRemote(workerId: number, functionName: string, ...params: WasmCallArgument[]): Promise<PerformanceData> {
    if (!initialized) {
      throw new Error(`wasm not initialized. please ensure 'init()' is called.`);
    }

    if (workerId < 0 || workerId >= WORKER_NUMBER) {
      throw new Error(`invalid worker ID ${workerId}. should be in range [0, ${WORKER_NUMBER})`);
    }

    const offset: number[] = [];
    const size = WasmBinding.calculateOffsets(offset, params);
    const buffer = new ArrayBuffer(size);
    WasmBinding.ccallSerialize(new Uint8Array(buffer), offset, params);

    const startTime = bindingCore.now();
    workers[workerId].postMessage({type: 'ccall', func: functionName, buffer}, [buffer]);

    return new Promise<PerformanceData>((resolve, reject) => {
      completeCallbacks[workerId].push((buffer, perf) => {
        perf.startTimeWorker = perf.startTime;
        perf.endTimeWorker = perf.endTime;
        perf.startTime = startTime;
        perf.endTime = bindingCore.now();

        WasmBinding.ccallDeserialize(new Uint8Array(buffer), offset, params);
        resolve(perf);
      });
    });
  }
}

function areWebWorkersSupported(): boolean {
  // very simplistic check to make sure the environment supports usage of workers
  // tslint:disable-next-line:no-any
  if (typeof window !== 'undefined' && typeof (window as any).Worker !== 'undefined') {
    return true;
  }
  return false;
}
