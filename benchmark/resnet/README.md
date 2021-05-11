# Benchmarks
This sub project is to benchmark and compare ONNX.js peformance vs other leading in-browser AI Inference frameworks.

## Frameworks
- TensorFlow.js
- Keras.js
- WebDNN
- ONNX.js

## Backends
(not all backends supported by all platforms)
- WebGL
- WebAssembly
- CPU

## Browsers
(not all framework/backend combinations are supported by all browsers)
- Chrome (WebGL 2)
- Edge (WebGL 1)

## Instructions
Please download all the sub-folders (containing the model files and corresponding test data) under
https://github.com/Microsoft/onnxjs-demo/tree/data/data/benchmark and place them in ./benchmark/data prior to running the benchmark tests

1. Ensure that the ONNX.js project (the parent) is already installed and built:
```bash
npm ci
npm run build
```
2. Change to `benchmark` subfolder and run npm ci and build in the benchmark folder
```bash
cd benchmark
npm install
npm run build
```
3. Run tests (Chrome)
```bash
npm run test
```
4. Run tests (Edge)

Note that the Edge tests are likely to crash the broswer. A recommended way would be to comment out
all Frameworks and backends except one and repeat this for all others. Look in the definition for
`BenchmarkImageNetData` in src/index.js
```bash
npm run test-edge
```
