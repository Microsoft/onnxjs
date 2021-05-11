// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// tslint:disable:no-require-imports

if (typeof window !== 'undefined' && !onnx.backend.webgl.disabled) {
  require('./backends/webgl/test_glsl_function_inliner');
  require('./backends/webgl/test_conv_new');
  require('./backends/webgl/test_pack_unpack');
  require('./backends/webgl/test_concat_packed');
  require('./backends/webgl/test_depth_to_space');
  require('./backends/webgl/test_reshape_packed');
  require('./backends/webgl/test_matmul_packed');
}

// require('./api/onnx');
// require('./api/inference-session');
// require('./api/tensor');
// require('./api/types');

// require('./opset');
