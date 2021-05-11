// Karma configuration
// Generated on Thu Nov 01 2018 14:03:26 GMT-0700 (Pacific Daylight Time)
const path = require('path')

module.exports = function(config) {
  config.set({
    basePath: './',
    frameworks: ['mocha'],
    files: [
      { pattern: 'dist/main.js' },
      { pattern: 'dist/onnx-wasm.wasm', included: false},
	    { pattern: 'dist/onnx-worker.js', included: false},
      { pattern: 'data/**/*', watched: false, included: false, served: true, nocache: true }
    ],
    proxies: {
      '/onnx-wasm.wasm': '/base/dist/onnx-wasm.wasm',
      '/onnx-worker.js': '/base/dist/onnx-worker.js',
	 },
    exclude: [
    ],
    // available preprocessors: https://npmjs.org/browse/keyword/karma-preprocessor
    preprocessors: {
    },
    reporters: ['mocha'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    logLevel: config.LOG_INFO,
    autoWatch: false,
    customLaunchers: {
      ChromeTest: {base: 'Chrome', flags: ['--window-size=1,1']},
      ChromeDebug: {debug: true, base: 'Chrome', flags: ['--remote-debugging-port=9333']}
    },
    client: {
      captureConsole: true,
      mocha: {expose: ['body'], timeout: 3000000},
      browser: config.browsers,
      printMatches: config.printMatches ? true : false
    },
    browsers: ['ChromeTest', 'ChromeDebug', 'Edge', 'Safari'],
    browserConsoleLogOptions: {level: "debug", format: "%b %T: %m", terminal: true},
  })
}
