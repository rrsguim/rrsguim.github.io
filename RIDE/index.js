/**
 * @license
 *
 * RIDE - Representantion learning for Interpolation, Distribution and Extrapolation of time series by related series.
 *
 * Copyright 2021 Rafael Guimaraes. All Rights Reserved.
 * =============================================================================
 */


async function run() {
   //Data
   const data = await getData();
   const tensorData = convertToTensor(data);
   const {xs, ys} = tensorData;

console.log('xs shape', xs.shape[0],xs.shape[1]);
console.log('ys shape', ys.shape[0],ys.shape[1]);

   // Create the model
   const model = createModel(xs.shape[1]);
  
   //Show model summary
   tfvis.show.modelSummary({name: 'Model Summary', tab:'Model'}, model);

   //Show summary information about layers and a histogram of parameters in each layer. 
//   const nameLayers = ['First', 'Second'];
//   const dataLayers = { values: [model.getLayer('Input', 1), model.getLayer('Dense 1', 2)], nameLayers  }; //TODO all layers ************
   const dataLayers = model.getLayer('Input', 1);
   const surface = { name: 'Layer Summary', tab: 'Layers'};
   tfvis.show.layer(surface, dataLayers);

// It is here because of reload issues, but it is not working properly // TODO ****************
//   tfvis.show.layer( {name: 'Training Performance', tab: 'Performance' });
//   tfvis.visor().setActiveTab('Performance');

   // Train the model  
   await trainModel(model, xs, ys);
 
   //Show Original data X RIDE
   results(ys, model.predict(xs));
 
}



async function getData() {
   const csvUrl = document.getElementById("query-url").value	
console.log('Data source:', csvUrl);
   const csvDataset = tf.data.csv(
	   csvUrl, {
         columnConfigs: {
             TARGET: { 
                 isLabel: true
             }
         }
   });
   const arrays = await csvDataset.toArray();
   return {
	   arrays: arrays 
	};
}



function convertToTensor(data) {
  return tf.tidy(() => {

    //Convert data to Tensor
    const dataSet = (data.arrays)
      .map(({xs, ys}) =>
        {
          // Convert xs(features) and ys(labels) from object form (keyed by
          // column name) to array form.
          return {xs:Object.values(xs), ys:Object.values(ys)};
        });

    /// Create label tensor to visualize it
    const length = dataSet.length;
    const xs = [];
    const ys = [];

    for (let i = 0; i < length; i++) {
       xs.push((dataSet[i]).xs)
       ys.push((dataSet[i]).ys)
    }
	
	//Array to tensor
    const xsTensor = tf.tensor(xs); 
    const ysTensor = tf.tensor(ys); 

    return {
      xs: xsTensor, 
      ys: ysTensor, 
    }
  });  
}



function createModel(input_shape) {
   // Define the model
   const model = tf.sequential();
   // Define some parameters
   const unit = parseInt(document.getElementById("dense-units").value);
   const lambda = parseFloat(document.getElementById("regularizerLambda").value);
   var regularizer = (document.getElementById("regularizer").value == "L1") ? (tf.regularizers.l1({l1: lambda})) : 
                     ((document.getElementById("regularizer").value == "L2") ? (tf.regularizers.l2({l2: lambda})) : 
                     (tf.regularizers.l1l2({l1: lambda, l2: lambda})));
   const activation = document.getElementById("activationFunction").value;
   // Input Layer
   model.add(tf.layers.dense({
     inputShape: [input_shape], 
     units: input_shape,        
//     useBias: false,
//     kernel_regularizer: regularizer,
//     activation: activation,
     name: 'Input',
   }));
   // Dense Layer 1
   model.add(tf.layers.dense({
     units: unit,
     useBias: true, //false
     kernel_regularizer: regularizer,
     activation: activation,
     name: 'Dense_1',
   }));
   // Dense Layer 2
   model.add(tf.layers.dense({
     units: unit,
     useBias: true, //false
     kernel_regularizer: regularizer,
     activation: activation,
     name: 'Dense_2',
   }));
   // Dense Layer 3
   model.add(tf.layers.dense({
     units: unit,
     useBias: true, //false
     kernel_regularizer: regularizer,
     activation: activation,
     name: 'Dense_3',
   }));
   // Output Layer
   model.add(tf.layers.dense({
	 units: 1,
     activation: 'linear',          
     useBias: false,
     name: 'Output',				
   }));
//check
console.log('Hidden units', unit);
console.log('Activation function', activation);
console.log('Regularizer', regularizer);
console.log('Lambda regularizer', lambda);

   return model;
}


async function trainModel(model, xs, ys) {
   // Define some parameters
   var optimizer = (document.getElementById("optimizer").value == 'adam') ? tf.train.adam(parseFloat(document.getElementById("learningRate").value)) : tf.train.adagrad(parseFloat(document.getElementById("learningRate").value));
   const batchSize = (document.getElementById("batchSize").value == '1') ? xs.shape[0] : parseFloat(document.getElementById("batchSize").value);
   const epochs = parseFloat(document.getElementById("epochs").value);
   const validationSplit = parseFloat(document.getElementById("validationSplit").value);
//check
console.log('Optimizer', optimizer);
console.log('Batch size', batchSize);
console.log('Epochs', epochs);
console.log('Validation Split', validationSplit);
   // Prepare the model for training.
   model.compile({
     optimizer: optimizer,
     loss: 'meanSquaredError', 
     metrics: ['mse', 'mae'],  
   });
   // Fit the model using the prepared Dataset
   return await model.fit(xs,ys, {
	 batchSize,
     epochs,
	 validationData: validation_split=validationSplit,
     shuffle: true,
     callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance', tab: 'Performance' },
      ['loss', 'mae'], //mse / 'val_loss'  // see    https://storage.googleapis.com/tfjs-vis/mnist/dist/index.html 
      { 
      height: 200, 
      callbacks: ['onEpochEnd']
      }
     )
   });
}



function results(ys, ride) {
    //Show original data
	const arr_y = ys.dataSync();
	const arr_ride = ride.dataSync();	
	var y_length = [];
	for (var i = 0; i <= arr_y.length; i++) {
   		y_length.push(i);
	}
	let zip = (arr1, arr2) => arr1.map((x, i) => { return {'x':x, 'y':arr2[i]}})
	const user_data = zip(y_length, arr_y)
	const ride_data = zip(y_length, arr_ride)
	const label1 = 'TARGET'
	const label2 = 'RIDE'		
		
	let original_data = { values: [user_data, ride_data], series: [label1, label2] }
console.log(original_data);                          // TODO ************************* data to download
	// Render to visor
	const surface = { name: 'RESULTS', tab: 'Results' };
	tfvis.render.linechart(surface, original_data, 
	  { 
	  width: 500, height: 450,                       // TODO size adjustable *************
	  xLabel: 'Time', yLabel: 'TARGET unit', 
	  }
    )
}



async function startRunning(){
	run();
}



async function init(){

}









