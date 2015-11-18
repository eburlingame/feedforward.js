// Requirements
var networks = require('./netclass_module.js');
var jsonfile = require('jsonfile');
var util = require('util');
var fs = require("fs");

var synaptic = require('synaptic'); // this line is not needed in the browser
var Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

console.log("Reading from file...");
var file = './data/data_5k.json';
var contents = jsonfile.readFileSync(file);

console.log("Processing inputs...");
var trainingSet = [];
for (var i = 0; i < contents.length; i++)
{
	trainingSet.push({  input:   contents[i].data, 
						output:  contents[i].validation  });
}
console.log("Loaded " + trainingSet.length + " training patterns");


// create the network
console.log("Creating network...");
var inputLayer = new Layer(784);
var hiddenLayer = new Layer(30);
var outputLayer = new Layer(10);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer
});

console.log("Training network...");
var trainer = new Trainer(myNetwork);
trainer.train(trainingSet,{
    rate: 0.1,
    iterations: 10,
    error: .005,
    shuffle: true,
    log: 1,
});

console.log("Testing all training samples...")
var correct = 0;
for (var i = 0; i < trainingSet.length; i++)
{
	var input = trainingSet[i].input;
	var output = myNetwork.activate(input);
	var desiredOutput = trainingSet[i].output;
	if (arrMaxLoc(output) == arrMaxLoc(desiredOutput))
	{
		correct++;
	}
}
console.log("Number correct: " + correct + " / " + trainingSet.length + " (" + (correct / trainingSet.length) + " %)");

function arrMaxLoc(arr)
{
	var maxLoc = 0;
	for (var i = 0; i < arr.length; i++)
	{
		if (arr[i] > arr[maxLoc])
		{
			maxLoc = i;
		}
	}
	return maxLoc;
}