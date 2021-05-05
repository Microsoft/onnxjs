## Prerequisites

- Node.js (10.13.0+): https://nodejs.org/en/ - (Optional) Use nvm ([Windows](https://github.com/coreybutler/nvm-windows) / [Mac/Linux](https://github.com/creationix/nvm)) to install Node.js

- Python (2.7 or 3.6+): https://www.python.org/downloads/

  - python should be added to the PATH environment variable

- Visual Studio Code: https://code.visualstudio.com/

  - **required** extension: [TSLint](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-typescript-tslint-plugin)
  - **required** extension: [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
  - **required** extension: [Debugger for Chrome](https://marketplace.visualstudio.com/items?itemName=msjsdiag.debugger-for-chrome)

- Chrome Browser

- (Optional) Netron: https://lutzroeder.github.io/Netron/

## Development

Please follow the following steps to running tests:

1. run `npm ci` in the root folder of the repo.
2. (Optional) build WebAssembly backend:
   1. build ONNX Runtime WebAssembly and copy files "ort-wasm.\*" to /dist/.
   2. if building ONNX Runtime WebAssembly with multi-threads support, copy files "ort-wasm-threaded.\*" to /dist/.
   3. run `npm run build` in the root folder of the repo to enable WebAssebmly features.
3. run `npm test` to run suite0 test cases and check the console output.
   - if (2) is not run, please run `npm test -- -b=cpu,webgl` to skip WebAssebmly tests

### Debugging on Desktop/Chrome

To debug the code from test-runner on Chrome:

- make sure `npm ci` executed at least once.
- use **vscode** to open the root folder.
- run `npm test -- <your-args> --debug` to run one or more test cases.
- in the open Chrome browser, click the `DEBUG` button on the top-right of the page.
- in VSCode, click [menu]->Debug->Start Debugging or press F5, select `Karma DEBUG RUNNER` to attach.
- put breakpoints in source code, and Refresh the page to reload.

_Note: run `npm test -- --help` for a full CLI instruction._

### Debugging on iOS/Safari

To debug on an Apple iOS device, please refer to the following steps:

- install [
  RemoteDebug iOS WebKit Adapter](https://github.com/RemoteDebug/remotedebug-ios-webkit-adapter) by following its instructions.
- launch the adapter in commandline: `remotedebug_ios_webkit_adapter --port=9000`.
- in VSCode, select debug configuration `Remote Browser via Webkit Adaptor`.
- follow the steps above to debug.

### Debugging on Android/Chrome

To debug on an Android device, please refer to the following steps:

- Install [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools) and make sure `adb` is ready to use.
- Follow instructions in [Remote Debugging on Android](https://developer.chrome.com/devtools/docs/remote-debugging-legacy) to launch `adb`. Make sure to use port 9000 so that the existing debug configuration works.
- in VSCode, select debug configuration `Remote Browser via Webkit Adaptor`.
- follow the steps above to debug.

### Performance profiling

To profile the kernel level execution time, please run the following command:

npm test -- model <model_name> --profile

It generates raw perf data for each kernel, you may want use tools/parse-profiler.ts to parse it.
