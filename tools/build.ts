// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import npmlog from 'npmlog';
import * as path from 'path';

npmlog.info('Build', 'Initializing...');

// Flags
// To trigger fetching some WASM dependencies and some post-processing
const buildWasm = process.argv.indexOf('--build-wasm') !== -1;
// To trigger a clean install
const cleanInstall = process.argv.indexOf('--clean-install') !== -1;
// To call webpack to generate the bundle .js file
const buildBundle = process.argv.indexOf('--build-bundle') !== -1;

// tslint:disable: non-literal-fs-path

// Path variables
const ROOT = path.join(__dirname, '..');
const DEPS = path.join(ROOT, 'deps');
const DEPS_ONNX = path.join(DEPS, 'onnx');
const TEST = path.join(ROOT, 'test');
const TEST_DATA = path.join(TEST, 'data');
const TEST_DATA_NODE = path.join(TEST_DATA, 'node');
const OUT = path.join(ROOT, 'dist');
const OUT_WASM_JS = path.join(OUT, 'ort-wasm.js');
const OUT_WASM = path.join(OUT, 'ort-wasm.wasm');

npmlog.info('Build', 'Initialization completed. Start to build...');

npmlog.info('Build', `Ensure output folder: ${OUT}`);
fs.ensureDirSync(OUT);

npmlog.info('Build', 'Updating submodules...');
// Step 1: Clean folders if needed
npmlog.info('Build.SubModules', '(1/2) Cleaning dependencies folder...');
if (cleanInstall) {
  fs.removeSync(DEPS);
}
npmlog.info('Build.SubModules', `(1/2) Cleaning dependencies folder... ${cleanInstall ? 'DONE' : 'SKIPPED'}`);

// Step 2: Get dependencies (if needed)
npmlog.info('Build.SubModules', '(2/2) Fetching submodules...');
const update = spawnSync('git submodule update --init --recursive', {shell: true, stdio: 'inherit', cwd: ROOT});
if (update.status !== 0) {
  if (update.error) {
    console.error(update.error);
  }
  process.exit(update.status === null ? undefined : update.status);
}
npmlog.info('Build.SubModules', '(2/2) Fetching submodules... DONE');

npmlog.info('Build', 'Updating submodules... DONE');

npmlog.info('Build', 'Preparing test data...');
const prepareTestData = cleanInstall || !fs.existsSync(TEST_DATA_NODE);
if (prepareTestData) {
  // Step 1: Clean folders if needed
  npmlog.info('Build.TestData', '(1/3) Cleaning folder...');
  if (cleanInstall) {
    fs.emptyDirSync(TEST_DATA_NODE);
  }
  npmlog.info('Build.TestData', `(1/3) Cleaning folder... ${cleanInstall ? 'DONE' : 'SKIPPED'}`);
  // Step 2: copy node tests for different version
  npmlog.info('Build.TestData', '(2/3) Copy tests...');
  [['v7', 'rel-1.2.3'],
   ['v8', 'rel-1.3.0'],
   ['v9', 'rel-1.4.1'],
   ['v10', 'rel-1.5.0'],
   ['v11', 'rel-1.6.1'],
   ['v12', 'rel-1.7.0'],
  ].forEach(v => {
    const version = v[0];
    const commit = v[1];
    npmlog.info('Build.TestData', `Checking out deps/onnx ${commit}...`);
    const checkout = spawnSync(`git checkout -q -f ${commit}`, {shell: true, stdio: 'inherit', cwd: DEPS_ONNX});
    if (checkout.status !== 0) {
      if (checkout.error) {
        console.error(checkout.error);
      }
      process.exit(checkout.status === null ? undefined : checkout.status);
    }
    const from = path.join(DEPS_ONNX, 'onnx/backend/test/data/node');
    const to = path.join(TEST_DATA_NODE, version);
    npmlog.info('Build.TestData', `Copying folders from "${from}" to "${to}"...`);
    fs.copySync(from, to);
  });
  npmlog.info('Build.TestData', '(2/3) Copy tests... DONE');
  // Step 3: revert git index
  npmlog.info('Build.TestData', '(3/3) Revert git index...');
  const update = spawnSync(`git submodule update ${DEPS_ONNX}`, {shell: true, stdio: 'inherit', cwd: ROOT});
  if (update.status !== 0) {
    if (update.error) {
      console.error(update.error);
    }
    process.exit(update.status === null ? undefined : update.status);
  }
  npmlog.info('Build.TestData', '(3/3) Revert git index... DONE');
}
npmlog.info('Build', `Preparing test data... ${prepareTestData ? 'DONE' : 'SKIPPED'}`);

npmlog.info('Build', 'Building WebAssembly sources...');
if (!buildWasm) {
  // if not building Wasm AND the file onnx-wasm.js is not present, create a place holder file
  if (!fs.existsSync(OUT_WASM_JS)) {
    npmlog.info('Build.Wasm', `Writing fallback target file: ${OUT_WASM_JS}`);
    fs.writeFileSync(OUT_WASM_JS, `;throw new Error("please build WebAssembly before use wasm backend.");`);
  }
} else {
  if (!fs.existsSync(OUT_WASM)) {
    npmlog.error('Build.Wasm', 'Please make sure ort-wasm.wasm is built and exists in /dist/');
    process.exit(1);
  }
}
npmlog.info('Build', `Building WebAssembly sources... ${buildWasm ? 'DONE' : 'SKIPPED'}`);

// generate bundle
//
npmlog.info('Build', `Building bundle...`);
if (buildBundle) {
  // only generate bundle when WASM is built
  //
  if (!fs.existsSync(OUT_WASM)) {
    npmlog.error('Build.Bundle', `Cannot find wasm file: ${OUT_WASM}. Please build WebAssembly sources first.`);
    process.exit(2);
  } else {
    npmlog.info('Build.Bundle', '(1/2) Retrieving npm bin folder...');
    const npmBin = execSync('npm bin', {encoding: 'utf8'}).trimRight();
    npmlog.info('Build.Bundle', `(1/2) Retrieving npm bin folder... DONE, folder: ${npmBin}`);

    npmlog.info('Build.Bundle', '(2/2) Running webpack to generate onnx.min.js...');
    const webpackCommand = path.join(npmBin, 'webpack');
    const webpackArgs = ['--bundle-mode', 'prod'];
    npmlog.info('Build.Bundle', `CMD: ${webpackCommand} ${webpackArgs.join(' ')}`);
    const webpack = spawnSync(webpackCommand, webpackArgs, {shell: true, stdio: 'inherit'});
    if (webpack.status !== 0) {
      console.error(webpack.error);
      process.exit(webpack.status === null ? undefined : webpack.status);
    }
    npmlog.info('Build.Bundle', '(2/2) Running webpack to generate onnx.min.js... DONE');
  }
}
npmlog.info('Build', `Building bundle... ${buildBundle ? 'DONE' : 'SKIPPED'}`);
