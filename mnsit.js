// Requirements
var networks = require('./netclass_module.js');
var jsonfile = require('jsonfile')
var util = require('util')


var dim = [784, 30, 10];
var net = new networks.Network(dim);

var file = './data/data.json';
var contents = jsonfile.readFileSync(file);

var training_data = [];
for (var i = 0; i < contents.length; i++)
{
	training_data.push([ contents[i].data, contents[i].validation ]);
}

console.log("Training started...");
net.train(training_data, 30, 10, 0.01, training_data.slice(0, 1000));

console.log(net.testData(training_data) + " / " + training_data.length);