console.log('data import started');



// var arr = [];

// var str = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,37,37,37,7,0,0,0,0,0,0,0,0,78,0,0,0,0,0,0,0,0,0,0,14,84,182,188,193,254,254,254,124,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,22,91,130,193,254,254,204,125,201,254,254,246,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,214,254,255,254,255,173,22,0,98,254,255,152,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,245,198,75,31,2,0,117,245,254,221,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,36,0,0,0,0,78,246,254,222,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,243,254,225,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,249,254,220,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,242,254,224,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,254,219,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,242,254,254,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,209,254,232,83,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,226,254,224,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,254,250,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,244,174,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";

// for (var i=0; i<784; i++)
// {
	// arr[i] = str.split(",")[i];
// }

// const x = tf.tensor1d(arr); 

// const inputTensor = fromPixels(imageData, 1).reshape([1, 28, 28, 1]).cast('float32').div(scalar(255));

// x.print();

// const y = x.reshape([1, 28, 28, 1]);

// y.print();

// console.log('shape x:', x.shape);

// console.log('shape y:', y.shape);



// const values = y.dataSync();
// const arr = Array.from(values);
// console.log(values);
// console.log(arr);


// const predictionResult =  model.predict(y).dataSync();
  
// console.log(JSON.stringify(predictionResult));




//--------
	


import {MnistData} from './data.js';



async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {  


  const data = new MnistData();
  await data.load();
  await showExamples(data);
  
  console.log('data imported');
  
  
  
  
  
  const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  
  
  
  console.log('start training');
	  
  await train(model, data);
  
  console.log('model trained');
  
  await model.save('downloads://my-model');
  
  console.log('model saved');
  
  
  // await showAccuracy(model, data);
  // await showConfusion(model, data);
  
  
  //=================
  
  
  var arr = [];

  var str = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,192,134,32,0,0,0,0,0,0,0,0,15,77,5,0,0,0,0,0,0,0,0,0,0,0,0,17,235,250,169,0,0,0,0,0,0,0,0,15,220,241,37,0,0,0,0,0,0,0,0,0,0,0,20,189,253,147,0,0,0,0,0,0,0,0,0,139,253,100,0,0,0,0,0,0,0,0,0,0,0,0,70,253,253,21,0,0,0,0,0,0,0,0,43,254,173,13,0,0,0,0,0,0,0,0,0,0,0,22,153,253,96,0,0,0,0,0,0,0,0,43,231,254,92,0,0,0,0,0,0,0,0,0,0,0,0,163,255,204,11,0,0,0,0,0,0,0,0,104,254,158,0,0,0,0,0,0,0,0,0,0,0,0,0,162,253,178,5,0,0,0,0,0,0,9,131,237,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,162,253,253,191,175,70,70,70,70,133,197,253,253,169,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,228,253,253,254,253,253,253,253,254,253,253,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,65,137,254,232,137,137,137,44,253,253,161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,254,206,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,253,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,254,241,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,158,254,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,244,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,104,254,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,253,157,0,13,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,253,154,91,204,161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,253,254,253,154,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,190,128,23,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
  
  
  for (var i=0; i<784; i++)
  {
	  arr[i] = parseFloat(str.split(",")[i]) / 255;
  }

  const x = tf.tensor1d(arr); 



  const y = x.reshape([1, 28, 28, 1]);

// console.log('sfsg-1');

  // const predictionResult =  model.predict(y);   //.dataSync();
  
// console.log('sfsg-2');
  
// var prediction;
// tf.tidy(() => { 
// const output = model.predict(x);
// const axis = 1;
// prediction = Array.from(output.argMax(axis).dataSync());
// preds = output.arraySync();
  
// console.log('sfsg-3');
  
  // console.log(JSON.stringify(predictionResult));
  
  
  const output = model.predict(y);
  
  const prediction = Array.from(output.argMax([-1]).dataSync());
  
  console.log(prediction);
  
  // const preds = model.predict(y).argMax([-1]).dataSync());;
  
  // console.log(predictionResult);
  
// console.log('sfsg-4');
  
  
}

document.addEventListener('DOMContentLoaded', run);



function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;  
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}


async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}



const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  
  
  
  // console.log(JSON.stringify(testData));
  // console.log(testDataSize);
  
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  
  // testxs.print();
  
  // console.log(JSON.stringify(JSON.parse(JSON.stringify(testxs))));
  
  // console.log(JSON.stringify(testxs));
  
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}


