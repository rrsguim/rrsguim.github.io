/**
 * @license
 * Copyright 2021 Rafael Guimaraes. All Rights Reserved.
 * =============================================================================
 */

let model;


async function loadModel(){
    const MODEL_URL = 'https://rrsguim.github.io/DeepDental/model.json';
    const model = await tf.loadLayersModel(MODEL_URL);
	return model
}



async function predict() {

let genero = parseInt(document.getElementById("genero").value, 10);
let categoria = parseInt(document.getElementById("categoria").value, 10);
let tempo = parseInt(document.getElementById("tempo").value, 10);
let total = parseInt(document.getElementById("total").value, 10);
let ultima = parseInt(document.getElementById("ultima").value, 10);
let reagenda = parseInt(document.getElementById("reagenda").value, 10);
let estetica = parseInt(document.getElementById("estetica").value, 10);
let maiorValor = parseInt(document.getElementById("maiorValor").value, 10);
	
let patientData = tf.tensor2d([genero,categoria,tempo,total,ultima,reagenda,estetica,maiorValor], [1,8]);

    const predictedClass = tf.tidy(() => {
      const predictions = model.predict(patientData);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "A";
			break;
		case 1:
			predictionText = "B";
			break;
		case 2:
			predictionText = "C";
			break;
	}
	document.getElementById("result-box").innerText = predictionText;
			
    predictedClass.dispose();
	//TODO class probability

}



function startPredicting(){
	predict();
}



async function init(){
	model = await loadModel();
}



init();







