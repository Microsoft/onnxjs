import * as tf from '@tensorflow/tfjs';
import loadImage from 'blueimp-load-image';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import KerasJS from 'keras-js';
import * as WebDNN from 'webdnn';
import * as onnx from 'onnxjs';
import {imagenetClassesTopK} from './imagenet';

const IMAGE_URLS = [
    { name: 'cat', url: 'https://i.imgur.com/CzXTtJV.jpg' },
    { name: 'dog', url: 'https://i.imgur.com/OB0y6MR.jpg' },
    { name: 'fox', url: 'https://farm4.staticflickr.com/3852/14447103450_2d0ff8802b_z_d.jpg' },
    { name: 'cheetah', url: 'https://farm2.staticflickr.com/1533/26541536141_41abe98db3_z_d.jpg' },
    // { name: 'bird', url: 'https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z_d.jpg' },
    // { name: 'goldfish', url: 'https://farm2.staticflickr.com/1301/1349366952_982df2276f_z_d.jpg' },
    // { name: 'whale', url: 'https://farm9.staticflickr.com/8505/8441256181_4e98d8bff5_z_d.jpg' },
    // { name: 'bridge', url: 'https://i.imgur.com/OnwEDW3.jpg' },
    // { name: 'lighthouse', url: 'https://farm3.staticflickr.com/2220/1572613671_7311098b76_z_d.jpg' },
    // { name: 'airplane', url: 'https://farm6.staticflickr.com/5590/14821526429_5c6ea60405_z_d.jpg' },
    // { name: 'sailboat', url: 'https://farm7.staticflickr.com/6089/6115759179_86316c08ff_z_d.jpg' },
    // { name: 'cello', url: 'https://farm2.staticflickr.com/1090/4595137268_0e3f2b9aa7_z_d.jpg' },
    // { name: 'piano', url: 'https://farm4.staticflickr.com/3224/3081748027_0ee3d59fea_z_d.jpg' },
    // { name: 'apple', url: 'https://farm8.staticflickr.com/7377/9359257263_81b080a039_z_d.jpg' },
    // { name: 'orange', url: 'https://farm6.staticflickr.com/5251/5522940446_0d5724d43a_z_d.jpg' },
    // { name: 'flower', url: 'https://farm9.staticflickr.com/8295/8007075227_dc958c1fe6_z_d.jpg' },
    // { name: 'mushroom', url: 'https://farm2.staticflickr.com/1449/24800673529_64272a66ec_z_d.jpg' },
    // { name: 'coffee', url: 'https://farm4.staticflickr.com/3752/9684880330_9b4698f7cb_z_d.jpg' },
    // { name: 'wine', url: 'https://farm4.staticflickr.com/3827/11349066413_99c32dee4a_z_d.jpg' }
  ];
const SERVER_BASE_PATH = '/base';

const BackendMapping = {
  'ONNX.js' : {
    'webgl': 'GPU-webgl',
    'wasm': 'CPU-webassembly+webworker',
    'cpu': 'CPU-javascript'
  },
  'TensorFlow.js': {
    'webgl': 'GPU-webgl',
    'cpu': 'CPU-javascript'
  },
  'Keras.js': {
    'webgl': 'GPU-webgl',
    'cpu': 'CPU-javascript'
  },
  'WebDNN': {
    'webgl': 'GPU-webgl',
    'cpu': 'CPU-javascript',
    'webassembly': 'CPU-webassembly'
  }
}

const BenchmarkImageNetData = [
    {
        model: 'resnet50',
        imageSize: 224,
        testCases: [
            {
                impl: 'TensorFlow.js',
                modelPath: `${SERVER_BASE_PATH}/data/model-tfjs/model.json`,
                backends: [ 'webgl', 'cpu' ],
                inputs: IMAGE_URLS,
                webglLevels: [1, 2]
            },
            {
                impl: 'Keras.js',
                modelPath: `${SERVER_BASE_PATH}/data/model-keras/resnet50.bin`,
                backends: [ 'webgl', 'cpu' ],
                inputs: IMAGE_URLS,
                webglLevels: [2]
            },
            {
                impl: 'WebDNN',
                modelPath: `${SERVER_BASE_PATH}/data/model-webdnn/resnet50`,
                backends: [ 'webgl' , 'webassembly' ],
                inputs: IMAGE_URLS,
                webglLevels: [2]
            },
            {
                impl: 'ONNX.js',
                modelPath: `${SERVER_BASE_PATH}/data/model-onnx/resnet50_8.onnx`,
                backends: [  'webgl', 'wasm' ],
                inputs: IMAGE_URLS,
                webglLevels: [1, 2]
            },
        ]
    },
];
class ImageLoader {
    constructor(imageWidth, imageHeight) {
        this.canvas = document.createElement('canvas');
        this.canvas.width = imageWidth;
        this.canvas.height = imageHeight;
        this.ctx = this.canvas.getContext('2d');
    }
    async getImageData(url) {
        //console.log(url);
        //console.log(`${this.canvas.width}, ${this.canvas.height}`);
        await this.loadImageAsync(url);
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return imageData;
    }
    loadImageAsync(url) {
        return new Promise((resolve, reject)=>{
            this.loadImageCb(url, ()=>{
                resolve();
            });
        });
    }
    loadImageCb(url, cb) {
        loadImage(
            url,
            img => {
                if (img.type === 'error') {
                    throw `Could not load image: ${url}`;
                } else {
                    // load image data onto input canvas
                    this.ctx.drawImage(img, 0, 0)
                    //console.log(`image was loaded`);
                    window.setTimeout(() => {  cb();  }, 0);
                }
            },
            {
                maxWidth: this.canvas.width,
                maxHeight: this.canvas.height,
                cover: true,
                crop: true,
                canvas: true,
                crossOrigin: 'Anonymous'
            }
        );
    }
}
function createBenchmark(name) {
    switch (name) {
        case 'TensorFlow.js': return new TensorFlowResnetBenchmark();
        case 'Keras.js': return new KerasResnetBenchmark();
        case 'WebDNN': return new WebDnnResnetBenchmark();
        case 'ONNX.js': return new OnnxJsResnetBenchmark();
    }
}
async function runBenchmark(benchmarkData, backend, imageSize) {
    console.log(`runBenchmark is being called with ${benchmarkData.impl}, ${backend}, ${imageSize}`)
    const impl = createBenchmark(benchmarkData.impl);
    console.log(`impl: ${benchmarkData.impl}, modelPath: ${benchmarkData.modelPath}`)
    await impl.init(backend, benchmarkData.modelPath, imageSize);
    const imageLoader = new ImageLoader(imageSize, imageSize);
    const durations = [];
    for(const input of benchmarkData.inputs) {
        console.log(`Running ${input.name}`)
        const imageData = await imageLoader.getImageData(input.url);
        const outputData = await impl.runModel(imageData.data);
        durations.push(impl.duration);
        if(shouldPrintMatches) {
            printMatches(outputData);
        }
    }
    durations.shift();
    const sum = durations.reduce((a,b)=>a+b);
    const avg = sum / durations.length;
    console.log(`avg duration: ${avg}`);
    return {
        framework: benchmarkData.impl,
        backend: BackendMapping[benchmarkData.impl][backend],
        duration: avg
    };
}
function printMatches(data) {
    let outputClasses = [];
    if(!data || data.length === 0) {
        const empty = [];
        for (let i = 0; i < 5; i++) {
          empty.push({ name: '-', probability: 0, index: 0 });
        }
        outputClasses = empty;
    } else {
        outputClasses = imagenetClassesTopK(data, 5);
    }
    for(let i of [0, 1, 2, 3, 4]) {
        console.log(`Prediction: ${outputClasses[i].name}, probability: ${Math.round(100 * outputClasses[i].probability)}%`);
    }
}
class TensorFlowResnetBenchmark {
    async init(backend, modelPath, imageSize) {
        this.imageSize = imageSize;
        tf.disposeVariables();
        if(backend) {
            console.log(`Setting the backend to ${backend}`);
            tf.setBackend(backend);
        }
        this.model = await tf.loadModel(modelPath);
        console.log('Model loaded');
    }
    async runModel(data) {
        const inputTensor = this.preprocess(data, this.imageSize, this.imageSize);
        const start = performance.now();
        const output = this.model.predict(inputTensor);
        const outputData = output.dataSync();
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
    preprocess(data, width, height) {
        // data processing
        const dataTensor = ndarray(new Float32Array(data), [width, height, 4])
        const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, width, height, 3])

        ops.subseq(dataTensor.pick(null, null, 2), 103.939)
        ops.subseq(dataTensor.pick(null, null, 1), 116.779)
        ops.subseq(dataTensor.pick(null, null, 0), 123.68)
        ops.assign(dataProcessedTensor.pick(0, null, null, 0), dataTensor.pick(null, null, 2))
        ops.assign(dataProcessedTensor.pick(0, null, null, 1), dataTensor.pick(null, null, 1))
        ops.assign(dataProcessedTensor.pick(0, null, null, 2), dataTensor.pick(null, null, 0))

        return tf.tensor(dataProcessedTensor.data, dataProcessedTensor.shape);
    }
}
class KerasResnetBenchmark {
    async init(backend, modelPath, imageSize) {
        this.imageSize = imageSize;
        this.model = new KerasJS.Model({
            filepath: modelPath,
            gpu: backend === 'webgl'});
        await this.model.ready();
        //console.log('Model loaded');
    }
    async runModel(data) {
        const preprocessedData = this.preprocess(data, this.imageSize, this.imageSize);
        const inputName = this.model.inputLayerNames[0];
        const outputName = this.model.outputLayerNames[0];
        const inputData = { [inputName]: preprocessedData }
        const start = performance.now();
        const output = await this.model.predict(inputData);
        const outputData = output[outputName];
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
    preprocess(data, width, height) {
      // data processing
      // see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
      const dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])

      ops.subseq(dataTensor.pick(null, null, 2), 103.939)
      ops.subseq(dataTensor.pick(null, null, 1), 116.779)
      ops.subseq(dataTensor.pick(null, null, 0), 123.68)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))

      const preprocessedData = dataProcessedTensor.data
      return preprocessedData
    }
}
class WebDnnResnetBenchmark {
    async init(backend, modelPath, imageSize) {
      if(backend === 'cpu') {
        backend = 'fallback';
      }
      this.imageSize = imageSize;
      this.model = await WebDNN.load(modelPath, {backendOrder: backend});
    }
    async runModel(data) {
        const options = {
            type: Float32Array,
            color: WebDNN.Image.Color.BGR,
            order: WebDNN.Image.Order.HWC,
            bias: [123.68, 116.779, 103.939],
            scale: [1, 1, 1]
        };
        const preprocessedData = WebDNN.Image.getImageArrayFromImageData(
            {data:data, width:this.imageSize, height:this.imageSize}, options);
        console.log(`input size: ${preprocessedData.length}`);
        this.model.inputs[0].set(preprocessedData);
        const start = performance.now();
        await this.model.run();
        const outputData = this.model.outputs[0];
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
}
class OnnxJsResnetBenchmark {
    async init(backend, modelPath, imageSize) {
        this.imageSize = imageSize;
        const hint = {backendHint: backend };
        this.model = new onnx.InferenceSession(hint);
        await this.model.loadModel(modelPath);
    }
    async runModel(data) {
        const preprocessedData = this.preprocess(data, this.imageSize, this.imageSize);
        const start = performance.now();
        const outputMap = await this.model.run([preprocessedData]);
        const outputData = outputMap.values().next().value.data;
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
    preprocess(data, width, height) {
      // data processing
      const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
      const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

      ops.divseq(dataTensor, 128.0);
      ops.subseq(dataTensor, 1.0);

      ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
      ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
      ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));

      const tensor = new onnx.Tensor(dataProcessedTensor.data, 'float32', [1, 3, width, height]);
      return tensor;
    }
}
const results = [];
const browser = __karma__.config.browser[0];
const shouldPrintMatches = __karma__.config.printMatches;
console.log(`browser: ${browser}, shouldPrintMatches: ${shouldPrintMatches}`)

describe('ImageNet Tests', ()=> {
    for(const modelTestcase of BenchmarkImageNetData) {
        describe(`model: ${modelTestcase.model}`, ()=> {
            for(const testCase of modelTestcase.testCases) {
                for(const backend of testCase.backends) {
                    it(`testCase:${testCase.impl} ${backend}`,
                        async function() {
                            // rule 1: if only supports WebGL 2 then skip Edge
                            if(browser.startsWith('Edge') && backend === 'webgl' && !testCase.webglLevels.includes(1)) {
                                this.skip();
                                return;
                            }
                            // rule 2: For TensorFlow.js skip Edge for CPU since it crashes
                            // if(browser.startsWith('Edge') && testCase.impl === 'TensorFlow.js' && backend === 'cpu') {
                            //     this.skip();
                            //     return;
                            // }
                            results.push(await runBenchmark(testCase, backend, modelTestcase.imageSize));
                        }
                    );
                }
            }
        });
    }
    after('printing results', ()=> {
        console.log(JSON.stringify(results));
    });
});
