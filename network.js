
// Sizes in a 1D array that represents how many neurons per layer
// The first layer is assumed to be an input layer
function Network(sizes)
{
	this.num_layers = sizes.length;
	this.sizes = sizes;
	this.biases = [];
	this.weights = [];

	this.init = function()
	{
		// For each layer 
		for (var i = 0; i < this.sizes.length - 1; i++)
		{
			this.biases[i] = []; // Each layer needs an array of neurons
			this.weights[i] = []; // Each layer needs an array of neurons

			// For each "connection" to the next layer
			for (var j = 0; j < this.sizes[i + 1]; j++)
			{
				this.biases[i][j] = math.random(-1, 1);
				this.weights[i][j] = []; // We want weights[layer, neuron, input]
				// Each neuron has as many inputs as the previous layer has neurons
				for (var k = 0; k < this.sizes[i]; k++)
				{
					this.weights[i][j][k] = math.random(-1, 1); // Initialize it to a random value
				}
			}
		}		
	}
	this.init();

	// Feeds an array of input value forward through the network
	// a is an array that's length must match the number of input neurons
	this.feedforward = function(a)
	{
		// For each layer
		for (var i = 1; i < this.num_layers - 1; i++)
		{
			// Set array a to be the sigma ( w dot a  + b)
			var weights = this.weights[i]; // 2D array of biases per neuron
			var biases = this.biases[i]; // 1D array of biases 
			a = this.sigmoid( math.add( this.dot( a, weights ),  biases ) );
		}
		return a;
	}


	// Training data is an array of 2-element arrays representing the 
	// desired input and desired output as the 0th and 1st elements
	// The input and output are vectors (arrays) that must match the
	// number of input and out neurons respectively
	this.SGD = function(training_data, epochs, mini_batch_size, eta, test_data)
	{
		var n = training_data.length;
		// For each epoch
		for (var j = 0; j < epochs; j++)
		{
			// For each mini batch
			for (var k = 0; k < mini_batch_size; k++)
			{
				var miniBatch = this.getRandomSet(training_data, mini_batch_size);
				this.updateMiniBatch(miniBatch, eta);
			}
			if (test_data)
			{
				var correct = 0;
				for (var i = 0; i < test_data.length; i++)
				{
					var result = this.feedforward(test_data[i][0]);
					var diff = math.add ( result, math.multiply(-1, test_data[i][1]) );
					diff = math.abs(diff);
					if (diff < 0.5)
					{
						correct++;
					}
				}
				console.log("Epoch " + j + " is complete. " + correct + " / " + test_data.length + " correct");
			}
			else
			{
				console.log("Epoch " + j + " is complete");
			}
		}
	}

	this.updateMiniBatch = function(miniBatch, eta)
	{
		var nablaB = [];
		var nablaW = [];
		// For each layer
		for (var i = 0; i < this.num_layers - 1; i++)
		{
			// Make zero arrays the size and shape of the bias / weights arrays 
			nablaB[i] = math.multiply(this.biases[i], 0);
			nablaW[i] = math.multiply(this.weights[i], 0);
		}

		// For each test pair in the mini-batch
		for (var i = 0; i < miniBatch.length; i++)
		{
			var x = miniBatch[i][0]; // 0th element is input
			var y = miniBatch[i][1]; // 1st element is output

			// Returns array of 2-dimensional arrays
			// with the 0th element as delta_nabla biases
			// and the  1st element as delta_nable weights array
			var delta_nablas = this.backprop(x, y);

			// For each layer 
			for (var j = 0; j < this.num_layers - 1; j++)
			{
				nablaB[j] = math.add( nablaB[j], math.transpose(delta_nablas.B[j]) );
				nablaW[j] = math.add( nablaW[j], math.transpose(delta_nablas.W[j]) );
			}
		}

		// For each layer
		for (var i = 1; i < this.num_layers - 1; i++)
		{
			// Update the weights and biases based on the give nablaB and nablaWs 
			// calculated above
			var scale = -1 * eta / miniBatch.length;
			this.weights[i] = math.add( this.weights[i] , math.multiply(scale, nablaW[i]) );
			this.biases[i] = math.add( this.biases[i] , math.multiply(scale, nablaB[i]) );
		}
	}


	this.backprop = function(x, y)
	{
		var nablaB = []; // Store the gradient of cost / biases
		var nablaW = []; // Store the gradient of cost / weights
		// For each layer
		for (var i = 0; i < this.num_layers - 1; i++)
		{
			// Make zero arrays the size and shape of the bias / weights arrays 
			nablaB[i] = math.multiply(this.biases[i], 0);
			nablaW[i] = math.multiply(this.weights[i], 0);
		}

		var activation = x;
		var activations = [activation]; // To store all the activation values for each layer
		var zs = []; // To store all the z vectors for each layer

		// For each layer, feed-forward the activations
		for (var i = 0; i < this.biases.length; i++)
		{
			var weights = this.weights[i]; // 2D array of biases per neuron
			var biases = this.biases[i]; // 1D array of biases 
			
			// Calculate z = weights dot activations + bias
			var z = math.add( this.dot( activation, weights ), biases ); 
			zs.push(z); // Push onto list of z's

			activation = this.sigmoid(z); // Calculate the activation 
			activations.push(activation); // Push the array of activations for this layer
		}

		// Find the cost derivative for the last layer
		var costDerivative = this.costDerivative( activations[activations.length - 1], y );

		// Matrix will get squeezed if a and b in math.multiply(a, b) only have one element
		// Put them back into an array here
		if (costDerivative.constructor != Array)
		{
			costDerivative = [ costDerivative ];
		}

		var delta =  this.hadamard( costDerivative, this.sigmoidPrime( zs[ zs.length - 1 ] ) );

		// Matrix will get squeezed if a and b in math.multiply(a, b) only have one element
		// Put them back into an array here
		if (delta.constructor != Array)
		{
			delta = [ delta ];
		}

		// Calculate the gradients for the final layer weights and biases (connections to the output layer)
		nablaB[nablaB.length - 1] = delta;
		nablaW[nablaW.length - 1] = this.dot( delta, activations[ activations.length - 2 ] );

		// Backwards propagate the activations backwards
		for (var l = 2; l < this.num_layers; l++)
		{
			// Find z moving backwards
			z = zs[ zs.length - l ]; 
			// Calculate sigmoid prime
			var sp = this.sigmoidPrime(z); 

			var deltaDotWeight = this.deltaDotWeight(this.weights.length - l + 1, delta);
			delta = this.hadamard( deltaDotWeight , sp );
			if (delta.constructor != Array)
			{
				delta = [ delta ];
			}

			nablaB[ nablaB.length - l ] = delta;
			nablaW[ nablaW.length - l ] = this.dot( delta, activations[activations.length - l - 1] );
		}
		return { "B": nablaB, "W": nablaW };
	}

	this.deltaDotWeight = function(l, prevDelta)
	{
		var delta = [];
		// Initialize a new delta that's the size of the origin layer
		for (var i = 0; i < this.weights[l][0].length; i++)
		{
			delta[i] = 0;
		}

		// For each next-layer destination neuron (where the weighting are pointing)
		for (var i = 0; i < this.weights[l].length; i++)
		{
			var originWeight = this.weights[l][i];
			// For each origin-layer neuron
			for (var j = 0; j < originWeight.length; j++)
			{
				// Calculate the dot of previousDelta and the weight 
				delta[j] += originWeight[j] * prevDelta [i];
			}
		}
		return delta;
	}

	this.costDerivative = function(outputActivations, y)
	{
		for (var i = 0; i < outputActivations.length; i++)
		{
			outputActivations[i] = outputActivations[i] - y[i];
		}
		// Return (activations - y)
		return outputActivations;
	}

	// Returns a random set of the elements in arr with 
	// numberOfElements elements
	this.getRandomSet = function(arr, numberOfElements)
	{
		var elements = [];
		while (elements.length < numberOfElements)
		{
			var index = math.randomInt(0, numberOfElements - 1);
			elements.push(arr[index])
		}
		return elements;
	}


	// Takes two arrays a, b and returns an array who's elements
	// are the product of a[i] * b[i]
	this.hadamard = function(a, b)
	{
		for (var i = 0; i < a.length; i++)
		{
			a[i] = math.multiply(a[i], b[i]);
		}
		return a;
	}

	// Returns an array of dot products of a single dimension vector a, 
	// and a 2-dimensional vector b. Returned array will be of size
	// equal to that of a
	this.dot = function(a, b)
	{
		var result = [];
		// if (a.length == 1)
		// {
		// 	return math.multiply(b, a[0]);
		// }
		for (var i = 0; i < b.length; i++)
		{
			result[i] = math.multiply(a, b[i]);
		}
		return result;
	}

	// z is an array and sigmoid returns an array with the function 
	// applied to each element
	this.sigmoid = function(z)
	{
		for (var i = 0; i < z.length; i++ )
		{
			z[i] = this.sigmoidFunc(z[i]);
		}
		return z;
	}

	// z is an array and sigmoid returns an array with the function 
	// applied to each element
	this.sigmoidPrime = function(z)
	{
		for (var i = 0; i < z.length; i++ )
		{
			z[i] = this.sigmoidFunc(z[i]) * (1 - this.sigmoidFunc(z[i]));
		}
		return z;
	}

	this.sigmoidFunc = function(z)
	{
		return ( 1.0 / ( 1.0 + math.exp(-z) ) );
	}

}