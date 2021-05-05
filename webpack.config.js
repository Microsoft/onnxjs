const util = require('util');
const path = require('path');
const webpack = require('webpack');
const HardSourceWebpackPlugin = require('hard-source-webpack-plugin');

module.exports = (env, argv) => {
  const bundleMode = argv['bundle-mode'] || 'prod';  // 'prod'|'dev'|'perf'|undefined;

  const config = {
    resolve: {extensions: ['.ts', '.js']},
    plugins: [new webpack.WatchIgnorePlugin([/\.js$/, /\.d\.ts$/])],
    module: {rules: [{test: /\.tsx?$/, loader: 'ts-loader'}]},
    node: {fs: 'empty'},
    externals: {
      '../dist/ort-wasm':'onnxWasmBindingJs',
      '../dist/ort-wasm-threaded':'onnxWasmBindingJs',
      'perf_hooks':'perf_hooks',
      'worker_threads':'worker_threads'
    }
  };

  if (bundleMode === 'perf' || bundleMode === 'dev') {
    config.entry = path.resolve(__dirname, 'test/test-main.ts');
  } else {
    config.entry = path.resolve(__dirname, 'lib/api/index.ts');
  }

  if (bundleMode === 'perf') {
    config.output = {path: path.resolve(__dirname, 'test'), filename: 'onnx.perf.js', libraryTarget: 'umd'};
  } else if (bundleMode === 'dev') {
    config.output = {path: path.resolve(__dirname, 'test'), filename: 'onnx.dev.js', libraryTarget: 'umd'};
  } else {
    config.output = {path: path.resolve(__dirname, 'dist'), filename: 'onnx.min.js', libraryTarget: 'umd'};
  }

  if (bundleMode === 'prod') {
    config.mode = 'production';
    config.devtool = 'source-map';
  } else if (bundleMode === 'perf') {
    config.mode = 'production';
    config.devtool = '';
  } else {
    config.mode = 'development';
    config.devtool = 'inline-source-map';
    config.plugins.push(new HardSourceWebpackPlugin());
  }

  return config;
};
