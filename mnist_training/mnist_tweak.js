// Requirements
var networks = require('./netclass_module.js');
var jsonfile = require('jsonfile')
var util = require('util')
var fs = require("fs")


console.log("Reading from file...");
var file = './data/data_5k.json';
var contents = jsonfile.readFileSync(file);

console.log("Processing inputs...");
var training_data = [];
for (var i = 0; i < contents.length; i++)
{
	training_data.push([ contents[i].data, contents[i].validation ]);
}
console.log("Loaded " + training_data.length + " training patterns");


var outputFile = './results.json';
var results = [];
var middleLayerCounts 	= [ 30 ];
var batchSizes 			= [ 10 ];
var etas 				= [ 3 ];
var epochs 				= 30;
// Try combinations of training parameters
for (var i = 0; i < middleLayerCounts.length; i++)
{
	for (var j = 0; j < batchSizes.length; j++)
	{
		for (var k = 0; k < etas.length; k++)
		{
			console.log("\nRunning with middle size: " + middleLayerCounts[i]
					  + "\nEpochs: " + epochs
					  + "\nBatchSize: " + batchSizes[j]
					  + "\nLearningRate: " + etas[k]
					);
			results.push( trainNetwork(	middleLayerCounts[i], 
										epochs, 
										batchSizes[j], 
										etas[k], 
										training_data) );

			jsonfile.writeFileSync(outputFile, results);
		}
	}
}

console.log(results);



function trainNetwork(middleLayerNeurons, epochs, batchSize, eta, training_data)
{
	console.log("Training started...");
	var dim = [784, middleLayerNeurons, 10];
	var net = new networks.Network(dim);
	var corrects = net.train(training_data, epochs, batchSize, eta, training_data);
	console.log("Training complete...");
	return { 
		"middleLayerNeurons": middleLayerNeurons,
		"epochs": epochs, 
		"batchSize": batchSize, 
		"learningRate": eta,
		"correctPerEpoch" : corrects,
		"totalPatterns": training_data.length,
	};
}
