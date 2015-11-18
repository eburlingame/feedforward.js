var math = require('mathjs');

// Represents a single neuron with the weights leading to it 
// Pass in the number of neurons in the previous layer or the number
// of inputs for the first layer
function Neuron(previousCount)
{
	// Holds the weights connecting the previous layer to this neuron 
	this.weights = [];
	// Holds the bias for this neuron, initialize it to a random value
	this.bias = randNorm();
	for (var i = 0; i < previousCount; i++)
	{
		this.weights[i] = randNorm();
	}

	this.toObject = function()
	{
		return {
			bias: 		this.bias,
			weights: 	this.weights,
		};
	}

	// Takes a simple javascript object and generate the correct
	// network from it
	this.fromObject = function(obj)
	{
		this.bias = obj.bias;
		this.weights = obj.weights;
	}

	this.calculateZ = function(lastLayerActivations)
	{
		var sum = 1 * this.bias;
		for(var i = 0; i < this.weights.length; i++)
		{
			sum += lastLayerActivations[i] * this.weights[i];
		}
		return sum;
	}

	this.calculateActivation = function(lastLayerActivations)
	{
		var z = this.calculateZ(lastLayerActivations);
		return sigmoid(z);
	}

	// Takes the single delta value for this neuron
	this.previousDeltas = function(delta)
	{
		// Hold one piece of the previous layers delta
		var partPrevDelta = [];
		for (var i = 0; i < this.weights.length; i++)
		{
			// Calculate this piece of the previous layer's delta
			partPrevDelta[i] = delta * this.weights[i];
		}

		return partPrevDelta;
	}

	this.applyDeltas = function(previousActivations, delta, eta, batchSize)
	{
		var nablaB = delta; // Calculate dC/db
		this.bias -= ( eta / batchSize ) * nablaB; // Update the neuron's bias from the delta
		for (var i = 0; i < this.weights.length; i++)
		{
			var nablaW = previousActivations[i] * delta;  // Calculate dC/dw
			this.weights[i] -= ( eta / batchSize ) * nablaW; // Update the neuron's proceeding bias 
		}
	}
}

// Represents a layer of a neuron with the weights leading to it 
// Pass in the number of neurons in this layer
// Pass in the number of neurons in the previous layer or the number
// of inputs for the first layer
function Layer(neuronCount, previousCount)
{
	// Holds the neurons in this layer  
	this.neurons = [];
	for (var i = 0; i < neuronCount; i++)
	{
		this.neurons.push(new Neuron(previousCount));
	}

	this.toObject = function()
	{
		var obj = {};
		obj.neurons = [];
		for (var i = 0; i < this.neurons.length; i++)
		{
			obj.neurons[i] = this.neurons[i].toObject();		
		}
		return obj;
	}

	// Takes a simple javascript object and generate the correct
	// network from it
	this.fromObject = function(obj)
	{
		this.neurons = [];
		for (var i = 0; i < obj.neurons.length; i++)
		{
			var neuron = obj.neurons[i];
			this.neurons[i] = new Neuron(0);
			this.neurons[i].fromObject(neuron);
		}
	}
	
	this.getActivationArray = function(input)
	{
		var arr = [];
		for (var i = 0; i < this.neurons.length; i++)
		{
			arr[i] = this.neurons[i].calculateActivation(input);
		}
		return arr;
	}

	this.getZArray = function(input)
	{
		var arr = [];
		for (var i = 0; i < this.neurons.length; i++)
		{
			arr[i] = this.neurons[i].calculateZ(input);
		}
		return arr;
	}

	// Return an array of deltas for the previous layer of neurons 
	// (backpropigating the weights)
	// Takes an array of deltas for this layer and zs for the previous layer
	this.previousDeltas = function(deltas, zs)
	{
		var prevDeltas = [];
		var numPreviousNeurons = this.neurons[0].weights.length; // The number of neurons in the previous layer

		// For each neuron in this layer
		for (var i = 0; i < this.neurons.length; i++)
		{
			// Get the delta parts for each previous neuron 
			var prevParts = this.neurons[i].previousDeltas(deltas[i]);
			// For each previous layer neuron
			for (var j = 0; j < numPreviousNeurons; j++)
			{
				if (isNaN(prevDeltas[j]))
				{
					prevDeltas[j] = 0;
				}
				// Add the part of that delta from this neuron to the previous one's delta
				prevDeltas[j] += prevParts[j]; 
			}
		}


		// Calculate the sigmoid prime of the previous layer's z values and 
		// multiply the new deltas by it
		for (var i = 0; i < prevDeltas.length; i++)
		{
			// Add the part of that delta from this neuron to the previous one's delta
			prevDeltas[i] *= sigmoidPrime(zs[i]);
		}

		return prevDeltas;
	}

	this.applyDeltas = function(previousActivations, deltas, eta, batchSize)
	{
		// For each neuron
		for (var i = 0; i < this.neurons.length; i++)
		{
			this.neurons[i].applyDeltas(previousActivations, deltas[i], eta, batchSize);
		}
	}
}


exports.Network = function(size)
{
	// Holds an array of layers
	this.layers = []; 
	// Initialize the layers
	for (var i = 1; i < size.length; i++)
	{
		// Create a new layer, passing in the previous layer's size
		var layer = new Layer(size[i], size[i - 1]);
		// Save the layer
		this.layers.push(layer);
	}

	// Returns a standard javascript object with the necessary values
	// to write the network to a file
	this.toObject = function()
	{
		var obj = {};
		obj.layers = [];
		for (var i = 0; i < this.layers.length; i++)
		{
			obj.layers[i] = this.layers[i].toObject();		
		}
		return obj;
	}

	// Takes a simple javascript object and generate the correct
	// network from it
	this.fromObject = function(obj)
	{
		this.layers = [];
		for (var i = 0; i < obj.layers.length; i++)
		{
			var layer = obj.layers[i];
			this.layers[i] = new Layer(0);
			this.layers[i].fromObject(layer);
		}
	}

	// Pass in data to feed forward through the network
	// Takes an array of input matching in length to the first layer
	// Returns the activation array from the last layer
	this.feedForward = function(input)
	{
		// Go forward through the tree
		for (var i = 0; i < this.layers.length; i++)
		{
			input = this.layers[i].getActivationArray(input);
		}
		return input;
	}

	this.feedForwardZ = function(input)
	{
		// Go forward through the tree
		for (var i = 0; i < this.layers.length - 1; i++)
		{
			input = this.layers[i].getActivationArray(input);
		}
		return this.layers[this.layers.length - 1].getZArray(input);
	}


	// TRAINING FUNCTIONS

	this.train = function (data, epochs, batchSize, eta, testData)
	{
		var corrects = [];
		var n = data.length;
		data = shuffle(data); // Shuffle the data
		// For each epoch
		for (var j = 0; j < epochs; j++)
		{
			// For each mini batch
			for (var k = 0; k < data.length; k += batchSize)
			{
				var batch = data.slice(k, k + batchSize); // Take this batch chunk
				this.trainBatch(batch, eta);
			}
			// Test the data at the end of the epoch
			if (testData)
			{
				var correct = this.testData(testData);
				var percent = correct/testData.length * 100; 
				console.log("Epoch " + j + " is complete. " + correct + " / " + testData.length + " correct (" + percent + "%)");
				corrects.push(correct);
			}
			else
			{
				console.log("Epoch " + j + " is complete");
			}
		}
		return corrects;
	}

	this.testData = function(testData)
	{
		var correct = 0;
		var costSum = 0;
		for (var i = 0; i < testData.length; i++)
		{
			var result = this.feedForward(testData[i][0]);
			var score = this.score(result, testData[i][1]);
			if (score > 0)
			{
				correct++;
			}
			costSum += sumCost(result, testData[i][1]);
		}
		console.log("Averate cost: " + costSum / testData.length)
		return correct;
	}

	this.score = function(output, testData)
	{
		var outputMaxLoc = 0;
		var desiredMaxLoc = 0; 
		for (var i = 0; i < output.length; i++)
		{
			// Find the location of the maximum in the output array
			if (output[i] > output[outputMaxLoc] )
			{
				outputMaxLoc = i;
			}

			// Find the location of the maximum in the desired output array
			if (testData[i] > testData[desiredMaxLoc] )
			{
				desiredMaxLoc = i;
			}
		}
		// If the maximums match
		if (desiredMaxLoc == outputMaxLoc)
		{
			return 1;
		}
		return 0;
	}

	this.trainBatch = function(batch, eta)
	{
		// For each input/output in the batch
		var cost = 0;
		for (var i = 0; i < batch.length; i++)
		{
			var input = batch[i][0]; // 0th element is input
			var output = batch[i][1]; // 1st element is output

			this.backprop(input, output, eta, batch.length);
			var calculatedOutput = this.feedForward(input);
			cost += sumCost(calculatedOutput, output);
		}
		cost = 1/2 * cost;
		// console.log("Cost " + cost);
	}

	this.backprop = function(input, output, eta, batchSize)
	{
		var activations = []; // Holds the activations, layer by layer
		var zs = []; // Holds the zs, layer by layer
		var activation = input;
		// Feed forward through the network, saving zs and activations
		for (var i = 0; i < this.layers.length; i++)
		{
			zs[i] = this.layers[i].getZArray(activation);
			activation = this.layers[i].getActivationArray(activation);
			activations[i] = activation;
		}

		var deltas = []; // Holds the deltas
		// Calculate the delta for the last layer
		deltas[this.layers.length - 1] = this.finalDeltas(  
									input, 
									output, 
									zs[zs.length - 1] );

		// For each layer backwards up until the last layer
		for (var i = this.layers.length - 1; i > 0; i--)
		{
			// Calculate the deltas for the previous layer
			deltas[i - 1] = this.layers[i].previousDeltas(deltas[i], zs[i - 1]);
		}


		var previousActivations = input;
		// Apply the deltas, updating the weights and biases
		for (var i = 0; i < this.layers.length; i++)
		{
			// Apply the delta for this layer
			this.layers[i].applyDeltas(previousActivations, deltas[i], eta, batchSize);

			// Set previous activations to the activations of this layer for
			// the next iteration
			previousActivations = activations[i];
		}
	}

	// Calculate the delta (error) in the last output layer
	this.finalDeltas = function(input, output, finalZs)
	{
		var outputActivations = this.feedForward(input);
		var finalDeltas = [];
		for (var i = 0; i < outputActivations.length; i++)
		{
			finalDeltas[i] = costDerivative(outputActivations[i], output[i]);
			finalDeltas[i] *= sigmoidPrime(finalZs[i]);
		}
		return finalDeltas;
	}
}


// Helper math functions: 

function sumCost(output, y)
{
	var sum = 0;
	for (var i = 0; i < output.length; i++)
	{
		sum += (1/2) * math.pow((y[i] - output[i]), 2);
	}
	return sum;
}

// Returns the cost function derivative (output - desired output)
// For a given output activation value (outputActivation) and the desired output (y)
function costDerivative(outputActivation, y)
{
	return outputActivation - y;
}


function getRandomSet(arr, numberOfElements)
{
	var elements = [];
	while (elements.length < numberOfElements)
	{
		var index = math.randomInt(0, numberOfElements);
		elements.push(arr[index]);
	}
	return elements;
}

function shuffle(array) 
{
  var currentIndex = array.length, temporaryValue, randomIndex ;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}


function sigmoid(z)
{
	return ( 1.0 / ( 1.0 + math.exp(-z) ) );
}

function sigmoidPrime(z)
{
	return sigmoid(z) * (1 - sigmoid(z));
} 

function randNorm() 
{
    return ((Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random()) - 3) / 3;
}

