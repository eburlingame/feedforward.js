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

var results = [];
var batchSize	= 10;
var eta			= 3 ;
var epochs 		= 1;

var dim = [784, 30, 10];
var net = new networks.Network(dim);

console.log("\nEpochs: " + epochs
		  + "\nBatchSize: " + batchSize
		  + "\nLearningRate: " + eta
		);

console.log("Training started...");
net.train(training_data, epochs, batchSize, eta, training_data);
console.log("Training complete...");

// Write the network to a file
var networkFile = './network.json';
jsonfile.writeFileSync(networkFile, net.toObject());
