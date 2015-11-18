// Requirements
var networks = require('./netclass_module.js');
var jsonfile = require('jsonfile')
var util = require('util')
var fs = require("fs")


console.log("Reading from file...");
var file = './network.json';
var networkObj = jsonfile.readFileSync(file);

var dim = [3, 30, 1];
var net = new networks.Network(dim);
net.fromObject(networkObj);

console.log(net.feedForward([1, 2, 3]));
